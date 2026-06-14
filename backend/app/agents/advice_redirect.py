"""
Advice-redirect node.

When the user asks for prescriptive advice ("what should I do?", "tell me
the steps"), MannSaathi should NOT pretend to be a therapist / doctor /
lawyer / financial advisor. We gently redirect.

Design choice: this node uses a small LLM call to vary the wording (so it
doesn't feel canned), but the SYSTEM PROMPT is tight — it lists the exact
constraints and we keep `num_predict` low. This is a deliberate
LLM-where-it-helps / rules-where-it-matters tradeoff.

(We could make this purely template-based with no LLM call. The reason
not to: a robotic "I can't give advice" repeated verbatim feels cold.
A slightly-different empathetic redirect each time feels human.)
"""
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage

from app.agents.llm import get_chat_model
from app.agents.state import ChatTurn

ADVICE_REDIRECT_PROMPT = """\
You are MannSaathi. The user just asked for prescriptive advice (what to \
do, what steps, medical / legal / financial guidance). You are an AI \
companion, NOT a therapist / doctor / lawyer / financial advisor — and \
you say so kindly.

Respond in 2-3 short sentences:
1. Acknowledge what they're trying to figure out, in their own words.
2. Gently explain you're not the right resource for prescriptive advice.
3. Offer to keep listening to how this is affecting them.

No bullet points. No emojis. Warm but honest.
"""


def _to_lc_messages(
    history: List[ChatTurn], user_message: str
) -> List[BaseMessage]:
    msgs: List[BaseMessage] = [SystemMessage(content=ADVICE_REDIRECT_PROMPT)]
    for h in history:
        if h.get("role") == "user":
            msgs.append(HumanMessage(content=h.get("content", "")))
        elif h.get("role") == "assistant":
            msgs.append(AIMessage(content=h.get("content", "")))
    msgs.append(HumanMessage(content=user_message))
    return msgs


async def redirect_for_advice(
    history: List[ChatTurn], user_message: str
) -> str:
    model = get_chat_model()
    messages = _to_lc_messages(history, user_message)
    response = await model.ainvoke(messages)
    text = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    return text.strip()
