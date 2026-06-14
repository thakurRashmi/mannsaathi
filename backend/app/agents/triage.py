"""
Triage agent: classifies the incoming user message into one of a few
conversation lanes, so the right specialized agent handles it.

Design notes:

  Why not just let one big-prompt agent handle everything?
    Specialized prompts produce better responses than mega-prompts.
    Routing also lets us instrument: we can see in logs/metrics how often
    each lane fires — useful for prompt tuning later.

  Why structured output (JSON) instead of free text?
    Routing needs to be deterministic-ish. Free text replies like
    "Hmm, I think they're venting" are ambiguous. We force the model
    to return one of N strings, which we then parse.

  Why the fallback to "listener"?
    Listener is the safe default. If parsing fails or the model returns
    something we don't recognize, defaulting to Listener means we still
    respond gracefully (just maybe not as optimally as Reflector would).
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_chat_model
from app.agents.state import ChatTurn, Route

log = logging.getLogger("mannsaathi.agents.triage")

TRIAGE_SYSTEM_PROMPT = """\
You are the triage layer of a mental wellness chat. Your ONLY job is to \
classify the user's most recent message into one of three lanes:

- "listener"        : the user is venting / sharing feelings; they need a \
warm short reflective response. (Default lane.)
- "reflector"      : the user is going deeper, exploring causes, asking \
"why do I feel this way", or hinting at patterns — they'd benefit from a \
slower, more probing reply.
- "advice_redirect": the user is asking for prescriptive advice ("what \
should I do", "tell me what to do", "give me steps"), or asking for \
medical / legal / financial guidance.

Respond with ONE LINE of JSON only, like:
  {"route": "listener", "reason": "venting about work"}

No prose, no markdown, no code fences. Just one JSON object.
"""

_VALID_ROUTES: set[Route] = {"listener", "reflector", "advice_redirect"}

# Defensive parser: pull the first JSON-looking thing out of model output.
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


async def triage(user_message: str, history: List[ChatTurn]) -> tuple[Route, str]:
    """
    Returns (route, reason).

    On any failure (timeout, parse error, unknown route), returns
    ("listener", "fallback: <why>"). Listener is the safe default.
    """
    model = get_chat_model()

    # We give the model only the LAST few turns to keep this fast and
    # focused. Routing doesn't need full history.
    recent = history[-4:]
    recent_text = "\n".join(
        f"{t['role']}: {t['content']}" for t in recent
    )
    classification_prompt = (
        f"Recent conversation:\n{recent_text or '(no prior messages)'}\n\n"
        f"New user message: {user_message!r}\n\n"
        "Classify it. Return only the JSON object."
    )

    try:
        response = await model.ainvoke(
            [
                SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
                HumanMessage(content=classification_prompt),
            ]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        return _parse_route(raw)
    except Exception as e:
        log.warning("triage failed, defaulting to listener: %s", e)
        return "listener", f"fallback: {type(e).__name__}"


def _parse_route(raw: str) -> tuple[Route, str]:
    """Best-effort parse of the model's JSON output."""
    match = _JSON_RE.search(raw)
    if not match:
        log.warning("triage: no JSON found in output: %s", raw[:200])
        return "listener", "fallback: no JSON in output"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        log.warning("triage: JSON parse failed: %s | raw=%s", e, raw[:200])
        return "listener", "fallback: invalid JSON"

    route = data.get("route", "listener")
    reason = str(data.get("reason", ""))[:200]

    if route not in _VALID_ROUTES:
        log.warning("triage: unknown route %r, defaulting to listener", route)
        return "listener", f"fallback: unknown route {route!r}"

    return route, reason  # type: ignore[return-value]
