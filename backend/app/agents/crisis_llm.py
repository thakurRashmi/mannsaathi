"""
LLM-based crisis classifier.

This is the SECOND layer of the safety system. Its job is to catch
crisis signals that the deterministic rules layer misses — especially:

  - Euphemisms ("I'm tired of being here", "I just want it to stop")
  - Indirect statements ("Sometimes I think everyone would be relieved")
  - Multi-turn buildup ("I haven't slept in days... I can't take it")

It runs IN PARALLEL with the rules layer (see crisis_gate.py). Their
results are OR-merged: either layer firing is enough.

Failure mode (important):
  If this LLM call fails for any reason — network, timeout, malformed
  output — we return is_crisis=False. We do NOT raise. That's because:
    - The rules layer is the safety floor; it caught what's catchable.
    - Raising would break the entire chat for unrelated network blips.
    - Defaulting to "not crisis" means we'd miss SOME subtle cases,
      but we'd never falsely accuse someone or break their session.
  This is a deliberate availability-vs-safety tradeoff. The rules layer
  is what makes it acceptable.

Why structured JSON output:
  Free-text classification is ambiguous ("kinda?", "maybe", "I think so").
  We force the model into a binary decision with a confidence and reason.
  The reason field is for our logs/observability; it never goes to the user.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_chat_model

log = logging.getLogger("mannsaathi.agents.crisis_llm")


_SYSTEM_PROMPT = """\
You are a binary safety classifier for a mental wellness chat. Your ONLY \
job is to decide whether the user's most recent message indicates an \
active crisis that requires immediate human intervention.

WHEN UNSURE, ERR ON THE SIDE OF is_crisis=true. False alarms cost a user \
seeing helpline numbers (mild). Missed crises cost a real human life. \
The rules layer (separate) catches obvious cases; YOUR job is the subtle ones.

CRISIS — return is_crisis=true (these are all REAL examples that must be caught):
- Direct mentions of suicide, self-harm, ending one's life, killing oneself
- Euphemisms and indirect ideation, e.g.:
    "I'm tired of being here"
    "I just want everything/it to stop"
    "I can't go on"
    "I just want to sleep and never wake up"
    "I don't want to wake up tomorrow"
    "I've made peace with not being around"
    "I gave away all my things"  (preparation behavior)
    "I feel like a burden to everyone"  (burden ideation)
    "Nothing matters anymore" + any sign of finality
    "The pain will stop soon, one way or another"
    "I can't see myself being here next year"
    "My family would be better off without me"
    "I just want a permanent solution"
    "I'm done. With everything. Permanently."
- Active abuse: being hit, hurt, raped, threatened by someone right now
- Imminent danger to others: planning to hurt/kill someone

NOT crisis — return is_crisis=false:
- General sadness, anxiety, stress, loneliness, frustration
- Crying spells, panic attacks, sleep problems — without ideation
- Difficulty at work, school, relationships
- "This is killing me", "I'm dying of laughter", "I'd die for X" — figurative
- Articles, books, news, documentaries ABOUT mental health topics
- Wanting to help someone else (brother, friend, colleague)
- Asking about mental health resources in general

Respond with EXACTLY one line of JSON, no prose, no code fences:
  {"is_crisis": true, "confidence": "high", "reason": "burden ideation + giving away possessions"}

confidence must be one of: "low", "medium", "high".
"""


@dataclass
class LLMResult:
    is_crisis: bool
    confidence: str  # "low" | "medium" | "high" | "error"
    reason: str


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

# Bound the classifier call so it can't hang the request indefinitely.
# Local Ollama (gemma2:2b) is typically 2-5s when warm, but can take
# 15-20s on a cold start (model loading into RAM). We pick a generous
# ceiling so cold-start cases still get classified, while still capping
# request latency at a known worst case.
#
# Trade-off: a higher timeout means a slower worst-case user experience
# in a crisis. We mitigate this with the parallel rules layer — for any
# obvious crisis phrase, rules fire in microseconds and short-circuit
# this classifier entirely (see crisis_gate.py).
_CLASSIFIER_TIMEOUT_S = 30.0


async def classify_with_llm(user_message: str) -> LLMResult:
    """
    Classify a single user message for crisis content.

    Never raises. On any failure, returns is_crisis=False with
    confidence="error" so the rules layer remains our safety floor.
    """
    try:
        return await asyncio.wait_for(
            _classify_inner(user_message), timeout=_CLASSIFIER_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        log.warning("crisis LLM classifier timed out — falling through to rules layer only")
        return LLMResult(is_crisis=False, confidence="error", reason="timeout")
    except Exception as e:
        log.warning("crisis LLM classifier failed: %s", e)
        return LLMResult(is_crisis=False, confidence="error", reason=f"{type(e).__name__}")


async def _classify_inner(user_message: str) -> LLMResult:
    model = get_chat_model()
    response = await model.ainvoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Classify this user message: {user_message!r}"),
        ]
    )
    raw = response.content if isinstance(response.content, str) else str(response.content)
    return _parse(raw)


def _parse(raw: str) -> LLMResult:
    match = _JSON_RE.search(raw)
    if not match:
        return LLMResult(is_crisis=False, confidence="error", reason="no JSON in output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return LLMResult(is_crisis=False, confidence="error", reason="invalid JSON")

    is_crisis = bool(data.get("is_crisis", False))
    confidence = str(data.get("confidence", "low")).lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    reason = str(data.get("reason", ""))[:200]
    return LLMResult(is_crisis=is_crisis, confidence=confidence, reason=reason)
