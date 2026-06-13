"""
The Listener agent.

Right now this is the only agent — it just listens and responds. When we
add LangGraph in Task #5, this becomes one node in a graph alongside the
Reflector and Crisis Detector.

Design decisions encoded in the system prompt:

  1. Reflect feelings BEFORE offering solutions.
     Classic counseling pattern (Rogerian / motivational interviewing).
     Solutions-first responses make people feel unheard.

  2. Short responses (2-4 sentences) by default.
     Long AI walls of text feel clinical and overwhelming when someone
     is already overwhelmed.

  3. Don't pretend to be a therapist.
     We're explicit: "I'm an AI companion, not a therapist."

  4. Never minimize or dismiss feelings.
     No "it's not that bad" / "others have it worse" / forced positivity.

  5. Crisis is OUT OF SCOPE here.
     The Listener does NOT try to handle crisis itself. A separate
     Crisis Detector (Task #6) gates the response. This separation is
     critical for safety: an LLM that handles both listening AND crisis
     escalation is harder to verify than two specialized components.
"""
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage

from app.agents.llm import get_chat_model

SYSTEM_PROMPT = """\
You are MannSaathi, an AI companion for emotional support. You are NOT a \
therapist, doctor, or counselor — and you say so when relevant.

How to respond:
1. Reflect what the person seems to be feeling before anything else. \
Name the feeling specifically ("that sounds exhausting" beats "I'm sorry").
2. Ask one gentle, open-ended follow-up question — don't interrogate.
3. Keep responses short: 2-4 sentences, conversational tone, no bullet points.
4. Never minimize ("it's not that bad"), compare ("others have it worse"), \
or force positivity ("just stay positive!").
5. Never give medical, legal, or financial advice. If asked, redirect \
gently: "I'm not the right resource for that — a professional would help \
more here. But I can listen to how it's affecting you."
6. Use simple, warm language. Avoid clinical jargon.
7. If the person mentions self-harm, suicide, abuse, or danger to others, \
respond with care but DO NOT try to manage the crisis yourself — your \
safety layer will route them to a helpline.

You are a calm, attentive presence. Sit with the person where they are.
"""


def _to_lc_messages(
    history: List[dict], user_message: str
) -> List[BaseMessage]:
    """Convert the JSON chat history into LangChain message objects."""
    msgs: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
        # silently skip unknown roles — defensive against malformed input
    msgs.append(HumanMessage(content=user_message))
    return msgs


async def listen(history: List[dict], user_message: str) -> str:
    """
    Run a single turn of the Listener agent.

    `history` is the prior conversation as [{role, content}, ...].
    `user_message` is the new user message.

    Returns the assistant's reply text.
    """
    model = get_chat_model()
    messages = _to_lc_messages(history, user_message)

    # LangChain's `ainvoke` is the async path — important so we don't
    # block FastAPI's event loop while the LLM is thinking.
    response = await model.ainvoke(messages)

    # response.content is `str | list[...]` depending on provider;
    # for chat models with text output it's always a string in practice.
    text = response.content if isinstance(response.content, str) else str(response.content)
    return text.strip()
