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

CRISIS includes (return is_crisis=true):
- Direct or indirect mentions of suicide, self-harm, ending one's life
- Euphemisms: "I'm tired of being here", "I just want it all to stop", \
"I can't go on", "no point anymore", "everyone would be better off"
- Mentions of plans, means, or timing for self-harm
- Active abuse: being hit, hurt, raped, threatened by someone now
- Imminent danger to others: planning to hurt someone

NOT crisis (return is_crisis=false):
- General sadness, anxiety, stress, loneliness, frustration
- Difficulty at work, school, relationships
- Saying "I'm dying of laughter", "this is killing me" (figurative)
- Asking about mental health resources in general
- Talking ABOUT crisis abstractly without personal ideation

Respond with EXACTLY one line of JSON, no prose, no code fences:
  {"is_crisis": true, "confidence": "high", "reason": "explicit suicidal ideation"}

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
_CLASSIFIER_TIMEOUT_S = 25.0


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
