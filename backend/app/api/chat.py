"""
Chat endpoint. Runs the LangGraph conversation graph for each turn.

The HTTP contract here is intentionally simple — request in, reply out.
All the multi-agent complexity lives behind `agents.graph.run_conversation`.
Swapping the agent topology (adding a Crisis Detector in Task #6, adding
memory layers later) does NOT change this file.
"""
import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.graph import run_conversation

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
    # When crisis detection kicks in (Task #6), this flag tells the frontend
    # to show the helpline escalation UI instead of a normal chat bubble.
    is_crisis: bool = False
    # Observability fields — useful in dev / interview demos, ignored by the
    # current frontend. Lets you see "which agent answered me and why".
    route: Optional[str] = None
    route_reason: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        history = [m.model_dump() for m in req.history]
        result = await run_conversation(
            user_message=req.message, history=history
        )
        log.info(
            "chat :: route=%s reason=%s",
            result.get("route"),
            result.get("route_reason"),
        )
        return ChatResponse(
            reply=result["reply"],
            route=result.get("route"),
            route_reason=result.get("route_reason"),
        )
    except Exception as e:
        log.exception("chat endpoint failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=(
                "I'm having trouble thinking right now. "
                "Please try again in a moment."
            ),
        ) from e
