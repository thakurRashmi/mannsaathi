"""
Chat endpoint. Right now it echoes the user message back so we can verify
the full stack (frontend -> backend -> response) works end-to-end before
we plug in the LLM.

Day 2 will replace `_echo_response` with a real LangGraph agent call.
"""
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["chat"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    # When crisis detection kicks in (Day 2+), this flag tells the frontend
    # to show the helpline escalation UI instead of a normal chat bubble.
    is_crisis: bool = False


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    return ChatResponse(reply=_echo_response(req.message))


def _echo_response(user_msg: str) -> str:
    return (
        "I hear you. (This is a placeholder response — the AI brain is being "
        f"wired up next.) You said: “{user_msg}”"
    )
