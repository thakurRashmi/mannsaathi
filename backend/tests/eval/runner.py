"""
Eval harness for MannSaathi's crisis-detection system.

Two evaluation modes:

  - rules_only: only run the deterministic regex layer. Fast (<1s for 140
    cases). Run on every commit / in CI.

  - full_gate: run the complete crisis_gate (rules + LLM, OR-merged).
    Slow (~5 min for 140 cases on local Gemma 2B). Run nightly / before
    releases.

Both modes compute the same metrics:

  - Precision  : of the cases we flagged, what fraction were truly crisis?
  - Recall     : of the truly-crisis cases, what fraction did we catch?
  - F1         : harmonic mean of precision and recall
  - False negatives (FN)  : MISSED crises. THE SAFETY-CRITICAL METRIC.
                            We gate CI on FN==0 for the rules layer.
  - False positives (FP)  : safe messages flagged as crisis. Mild user-
                            experience cost (helplines shown unnecessarily).

Why the eval harness matters (more than the metrics themselves):
  This file turns "I tested it manually" into "I have a reproducible
  measurement of the safety system, with a CI gate that blocks any
  regression that misses a crisis case."

Usage:
  # Fast rules-only eval (suitable for CI):
  python -m tests.eval.runner --mode rules_only

  # Full gate eval (slower; needs Ollama running):
  python -m tests.eval.runner --mode full_gate --max-concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, List

# Set sane logging when running standalone (don't spam at INFO level).
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s :: %(message)s")

# Make the backend `app/` package importable when running via `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agents.crisis_rules import check_rules  # noqa: E402

# crisis_gate.evaluate is only imported lazily in full_gate mode so the
# rules-only mode doesn't pay LangChain's import cost.


# ---- Dataset loading --------------------------------------------------------

CASES_PATH = Path(__file__).parent / "cases.jsonl"


@dataclass
class Case:
    id: str
    text: str
    expected_crisis: bool
    category: str


def load_cases(path: Path = CASES_PATH) -> List[Case]:
    cases: list[Case] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = json.loads(line)
        cases.append(
            Case(
                id=d["id"],
                text=d["text"],
                expected_crisis=bool(d["expected_crisis"]),
                category=d["category"],
            )
        )
    return cases


# ---- Predictor functions ----------------------------------------------------
# Both take a case's text and return (predicted_is_crisis, latency_ms).

PredictFn = Callable[[str], Awaitable[tuple[bool, float]]]


async def predict_rules_only(text: str) -> tuple[bool, float]:
    t0 = time.perf_counter()
    result = check_rules(text)
    return result.is_crisis, (time.perf_counter() - t0) * 1000.0


async def predict_full_gate(text: str) -> tuple[bool, float]:
    # Lazy import: full_gate mode requires LangChain + Ollama running.
    from app.agents.crisis_gate import evaluate

    t0 = time.perf_counter()
    decision = await evaluate(text)
    return decision.is_crisis, (time.perf_counter() - t0) * 1000.0


# ---- Metrics ----------------------------------------------------------------

@dataclass
class CaseOutcome:
    case: Case
    predicted: bool
    latency_ms: float

    @property
    def is_tp(self) -> bool:
        return self.case.expected_crisis and self.predicted

    @property
    def is_fp(self) -> bool:
        return (not self.case.expected_crisis) and self.predicted

    @property
    def is_fn(self) -> bool:
        return self.case.expected_crisis and not self.predicted

    @property
    def is_tn(self) -> bool:
        return (not self.case.expected_crisis) and not self.predicted


@dataclass
class EvalReport:
    outcomes: List[CaseOutcome]
    mode: str
    duration_s: float

    # --- Top-level metrics

    @property
    def tp(self) -> int:
        return sum(1 for o in self.outcomes if o.is_tp)

    @property
    def fp(self) -> int:
        return sum(1 for o in self.outcomes if o.is_fp)

    @property
    def fn(self) -> int:
        return sum(1 for o in self.outcomes if o.is_fn)

    @property
    def tn(self) -> int:
        return sum(1 for o in self.outcomes if o.is_tn)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    # --- Per-category accuracy (useful for spotting weak spots)

    def by_category(self) -> dict[str, dict[str, float | int]]:
        buckets: dict[str, list[CaseOutcome]] = defaultdict(list)
        for o in self.outcomes:
            buckets[o.case.category].append(o)
        report: dict[str, dict[str, float | int]] = {}
        for cat, items in buckets.items():
            correct = sum(1 for i in items if i.predicted == i.case.expected_crisis)
            report[cat] = {
                "total": len(items),
                "correct": correct,
                "accuracy": correct / len(items) if items else 0.0,
            }
        return report

    def false_negatives(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.is_fn]

    def false_positives(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.is_fp]

    @property
    def avg_latency_ms(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.latency_ms for o in self.outcomes) / len(self.outcomes)

    def to_dict(self) -> dict:
        """Machine-readable summary (for CI artifacts, regression checks)."""
        return {
            "mode": self.mode,
            "duration_s": round(self.duration_s, 2),
            "total": self.total,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "by_category": self.by_category(),
            "false_negatives": [
                {"id": o.case.id, "text": o.case.text, "category": o.case.category}
                for o in self.false_negatives()
            ],
            "false_positives": [
                {"id": o.case.id, "text": o.case.text, "category": o.case.category}
                for o in self.false_positives()
            ],
        }


# ---- Runner -----------------------------------------------------------------

async def run_eval(
    predict_fn: PredictFn,
    cases: List[Case],
    mode: str,
    max_concurrency: int = 1,
) -> EvalReport:
    """
    Run all cases through `predict_fn` with bounded concurrency.

    max_concurrency > 1 is mainly useful for full_gate mode where the
    bottleneck is LLM latency. For rules_only it's irrelevant (everything
    finishes in microseconds).
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(case: Case) -> CaseOutcome:
        async with sem:
            predicted, latency = await predict_fn(case.text)
        return CaseOutcome(case=case, predicted=predicted, latency_ms=latency)

    t0 = time.perf_counter()
    outcomes = await asyncio.gather(*(_one(c) for c in cases))
    duration = time.perf_counter() - t0
    return EvalReport(outcomes=list(outcomes), mode=mode, duration_s=duration)


def print_report(report: EvalReport, *, show_failures: int = 10) -> None:
    """Pretty-printed summary, suitable for terminal viewing."""
    print()
    print("=" * 68)
    print(f" MannSaathi crisis-detection eval — mode: {report.mode}")
    print("=" * 68)
    print(f"  total cases       : {report.total}")
    print(f"  duration          : {report.duration_s:.2f}s "
          f"(avg {report.avg_latency_ms:.1f} ms / case)")
    print()
    print(f"  TP (real crisis caught)   : {report.tp}")
    print(f"  TN (safe correctly safe)  : {report.tn}")
    print(f"  FP (safe flagged crisis)  : {report.fp}")
    print(f"  FN (CRISIS MISSED)        : {report.fn}   "
          f"{'  ← SAFETY-CRITICAL' if report.fn else '✓ zero misses'}")
    print()
    print(f"  precision : {report.precision:.4f}")
    print(f"  recall    : {report.recall:.4f}")
    print(f"  F1        : {report.f1:.4f}")
    print(f"  accuracy  : {report.accuracy:.4f}")
    print()
    print("  Per-category accuracy:")
    for cat, m in sorted(report.by_category().items()):
        print(f"    {cat:20s}  {m['correct']:>3d}/{m['total']:<3d}   "
              f"({m['accuracy']:.1%})")

    if report.false_negatives():
        print()
        print(f"  ⚠️  FALSE NEGATIVES ({len(report.false_negatives())} — "
              f"crises the system missed):")
        for o in report.false_negatives()[:show_failures]:
            print(f"     [{o.case.category}] {o.case.id}: {o.case.text!r}")
        if len(report.false_negatives()) > show_failures:
            print(f"     … +{len(report.false_negatives()) - show_failures} more")

    if report.false_positives():
        print()
        print(f"  ⚠️  False positives ({len(report.false_positives())} — "
              f"safe messages incorrectly flagged):")
        for o in report.false_positives()[:show_failures]:
            print(f"     [{o.case.category}] {o.case.id}: {o.case.text!r}")
        if len(report.false_positives()) > show_failures:
            print(f"     … +{len(report.false_positives()) - show_failures} more")
    print("=" * 68)


# ---- CLI --------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the crisis-detection eval harness")
    parser.add_argument(
        "--mode",
        choices=["rules_only", "full_gate"],
        default="rules_only",
        help="rules_only is fast (<1s); full_gate runs the LLM classifier too",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=1,
        help="Number of concurrent eval calls (full_gate mode only)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print a machine-readable JSON report instead of the pretty one",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Write the JSON report to this file",
    )
    args = parser.parse_args()

    cases = load_cases()
    predict_fn = predict_rules_only if args.mode == "rules_only" else predict_full_gate
    report = asyncio.run(
        run_eval(predict_fn, cases, mode=args.mode, max_concurrency=args.max_concurrency)
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)

    if args.save:
        Path(args.save).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\n[saved JSON report to {args.save}]")

    # Exit code: non-zero if there are any false negatives. This is what
    # CI uses to gate merges — a regression that misses a crisis FAILS.
    return 1 if report.fn > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
