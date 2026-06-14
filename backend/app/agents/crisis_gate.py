"""
Crisis Gate — the top-level safety check that runs before any agent.

This module is the safety contract of MannSaathi. The rest of the agent
graph (Triage, Listener, Reflector, Advice-redirect) ONLY executes if
this gate decides the message is safe to process.

Architecture:

      ┌──────────────────────┐
      │   user_message       │
      └──────────┬───────────┘
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
  [ rules ]             [ LLM ]
    (sync)              (async)
       │                    │
       └─────────┬──────────┘
                 ▼
            OR-merge
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
  is_crisis=True       is_crisis=False
  → CrisisResponse     → continue to Triage

Why parallel:
  The rules layer is essentially instant; the LLM layer takes 1-3s.
  Running them concurrently means the gate's latency = LLM latency
  (not their sum). For a safety-critical path, every second counts.

Why OR-merge (not AND, not majority-vote):
  Both layers are tuned for HIGH RECALL on crisis. Each will produce
  some false positives (the rules layer especially). We accept those
  false positives because:
    - False positive: user sees helpline numbers when they didn't need
      them. Mild annoyance, no harm.
    - False negative: user in crisis gets a chatty LLM response when
      they needed a helpline. Real harm.
  OR-merge biases the system toward false positives. Deliberate.

The canned crisis response:
  We do NOT call any LLM to generate the response in a crisis. The text
  is hard-coded, human-vetted, and includes verified Indian helpline
  numbers. An LLM-generated crisis message could hallucinate the wrong
  number, the wrong tone, or in extreme cases something harmful. We
  refuse to take that risk in the safety path.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from app.agents.crisis_llm import classify_with_llm, LLMResult
from app.agents.crisis_rules import check_rules, RulesResult

log = logging.getLogger("mannsaathi.agents.crisis_gate")


# ---- Canned, human-vetted crisis response -----------------------------------
# This text is what the user sees if the gate fires. It is NEVER generated
# by an LLM. Helpline numbers verified June 2026; review quarterly.

CRISIS_RESPONSE_TEXT = (
    "I hear you, and I'm worried about you right now. What you're feeling "
    "matters, and you don't have to handle this alone.\n\n"
    "Please reach out to someone trained to help:\n\n"
    "📞 iCall (free, confidential): 9152987821\n"
    "📞 Vandrevala Foundation (24/7): 1860-2662-345\n"
    "📞 AASRA (24/7): 9820466726\n\n"
    "If you're in immediate danger, please call 112 (India emergency)."
)


@dataclass
class CrisisDecision:
    """Final decision from the gate, plus diagnostic info."""

    is_crisis: bool
    # Source attribution: which layer(s) flagged it. Useful for eval
    # harnesses and for tuning the rules vs LLM trade-off.
    rules_fired: bool
    llm_fired: bool
    rules_categories: list[str]
    llm_confidence: str  # "low" | "medium" | "high" | "error" | "n/a"
    llm_reason: str
    # The text to show the user IF is_crisis is True. None otherwise.
    response_text: Optional[str]


async def evaluate(user_message: str) -> CrisisDecision:
    """
    Run both safety layers in parallel and OR-merge their results.

    This function is the only entry point the rest of the app should use
    to check whether a message is in crisis.
    """
    # Kick off the LLM classifier in the background — it's the slow one.
    llm_task = asyncio.create_task(classify_with_llm(user_message))

    # Rules layer is synchronous and tiny; run it inline.
    rules: RulesResult = check_rules(user_message)

    # FAST PATH: if rules layer already says crisis, we don't need to
    # wait for the LLM. We still await its task (to avoid orphan
    # warnings + so its result shows in logs), but we know the decision.
    if rules.is_crisis:
        # Don't block on LLM — but also don't leak the task.
        # We schedule cancellation; if it has already started, that's fine.
        llm_task.cancel()
        try:
            await llm_task
        except (asyncio.CancelledError, Exception):
            pass
        llm = LLMResult(is_crisis=False, confidence="n/a", reason="short-circuited by rules")
    else:
        llm = await llm_task

    is_crisis = rules.is_crisis or llm.is_crisis
    log.info(
        "crisis gate :: is_crisis=%s rules=%s llm=%s | rules_cats=%s llm_conf=%s reason=%s",
        is_crisis,
        rules.is_crisis,
        llm.is_crisis,
        rules.matched_categories,
        llm.confidence,
        llm.reason,
    )

    return CrisisDecision(
        is_crisis=is_crisis,
        rules_fired=rules.is_crisis,
        llm_fired=llm.is_crisis,
        rules_categories=rules.matched_categories,
        llm_confidence=llm.confidence,
        llm_reason=llm.reason,
        response_text=CRISIS_RESPONSE_TEXT if is_crisis else None,
    )
