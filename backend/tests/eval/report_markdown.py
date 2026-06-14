"""
Render an eval report as Markdown — for pasting into README / blog posts.

Usage:
  # First run the eval and save JSON:
  python -m tests.eval.runner --mode rules_only --save /tmp/eval.json
  # Then render:
  python -m tests.eval.report_markdown /tmp/eval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def render(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"### Crisis-detection eval — `{report['mode']}` mode\n")
    lines.append(
        f"**{report['total']} cases · "
        f"{report['duration_s']}s · "
        f"avg {report['avg_latency_ms']} ms/case**\n"
    )
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Precision | **{report['precision']:.2%}** |")
    lines.append(f"| Recall    | **{report['recall']:.2%}** |")
    lines.append(f"| F1        | **{report['f1']:.2%}** |")
    lines.append(f"| Accuracy  | **{report['accuracy']:.2%}** |")
    lines.append(f"| False negatives (CRISES MISSED) | **{report['fn']}** |")
    lines.append(f"| False positives (safe flagged)  | **{report['fp']}** |")
    lines.append("")
    lines.append("**Per-category accuracy:**\n")
    lines.append("| Category | Correct / Total | Accuracy |")
    lines.append("|---|---|---|")
    for cat, m in sorted(report["by_category"].items()):
        lines.append(f"| `{cat}` | {m['correct']}/{m['total']} | {m['accuracy']:.1%} |")
    if report["false_negatives"]:
        lines.append("")
        lines.append("**False negatives** (cases the system missed):")
        for fn in report["false_negatives"][:10]:
            lines.append(f"- `[{fn['category']}]` {fn['text']!r}")
        if len(report["false_negatives"]) > 10:
            lines.append(f"- _… +{len(report['false_negatives']) - 10} more_")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: report_markdown.py <report.json>", file=sys.stderr)
        return 2
    report = json.loads(Path(sys.argv[1]).read_text())
    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
