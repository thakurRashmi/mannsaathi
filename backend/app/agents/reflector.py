"""
The Reflector agent.

When a user is exploring patterns ("why does this keep happening to me?"),
or going deeper into a feeling, the Listener's "name-the-feeling + one
question" response is too shallow. Reflector takes a slower, more probing
approach inspired by motivational interviewing:

  - Notices patterns across recent turns ("you mentioned your manager
    twice today")
  - Reflects implicit feelings the user hasn't named explicitly
  - Asks a probing question that invites self-discovery rather than
    information-gathering

This agent is invoked by the Triage router (see graph.py) when the user
signals they want to go deeper.
"""
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage

from app.agents.llm import get_chat_model
from app.agents.state import ChatTurn

REFLECTOR_SYSTEM_PROMPT = """\
You are the Reflector inside MannSaathi, an AI companion for emotional \
support. The user is going deeper now — exploring patterns, asking "why", \
or examining a feeling more closely.

How to respond:
1. Reflect a feeling the user has implied but NOT named explicitly. Use \
tentative language: "It sounds like...", "I wonder if...", "Almost like...".
2. If you've seen a pattern across recent turns, name it gently (e.g. \
"you've mentioned your manager twice — does that come up a lot?").
3. Ask ONE probing, open-ended question that invites self-discovery — \
not information ("what do you think might be underneath that?" beats \
"what time did this happen?").
4. Keep it short: 3-5 sentences total. No bullet points, no advice, no \
fixing.
5. Never minimize, never compare, never force positivity.
6. You are NOT a therapist. Don't diagnose. Don't offer techniques.
7. If the user mentions self-harm, suicide, abuse, or danger, do NOT try \
to handle it — your safety layer will route them.

You are slow, attentive, and curious. Sit one layer deeper with them.
"""


def _to_lc_messages(
    history: List[ChatTurn], user_message: str
) -> List[BaseMessage]:
    msgs: List[BaseMessage] = [SystemMessage(content=REFLECTOR_SYSTEM_PROMPT)]
    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=user_message))
    return msgs


async def reflect(history: List[ChatTurn], user_message: str) -> str:
    model = get_chat_model()
    messages = _to_lc_messages(history, user_message)
    response = await model.ainvoke(messages)
    text = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    return text.strip()
