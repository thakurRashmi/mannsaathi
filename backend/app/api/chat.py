"""
Chat endpoint. Calls the Listener agent to generate a real response.

In Task #5, the single-agent call will be replaced by a LangGraph run
that coordinates Listener + Reflector + Crisis Detector. The HTTP
contract on this endpoint stays the same — that's the point of putting
agent logic behind `agents/`.
"""
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.listener import listen

log = logging.getLogger("mannsaathi.api.chat")

router = APIRouter(prefix="/api", tags=["chat"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    # When crisis detection kicks in (Task #6), this flag tells the
    # frontend to show the helpline escalation UI instead of a normal
    # chat bubble.
    is_crisis: bool = False


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        # Pydantic models -> plain dicts for the agent layer.
        history = [m.model_dump() for m in req.history]
        reply = await listen(history=history, user_message=req.message)
        return ChatResponse(reply=reply)
    except Exception as e:
        # Don't leak stack traces to the client — show a calm fallback.
        # We log the real error server-side for debugging.
        log.exception("chat endpoint failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=(
                "I'm having trouble thinking right now. "
                "Please try again in a moment."
            ),
        ) from e
