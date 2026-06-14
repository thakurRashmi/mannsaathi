"""
Chat endpoint. Runs the LangGraph conversation graph for each turn.

The HTTP contract here is intentionally simple — request in, reply out
plus a few observability fields. All the multi-agent + safety complexity
lives behind `agents.graph.run_conversation`.
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
    # Tells the frontend to render the crisis escalation UI (helpline
    # numbers, "you matter" framing) instead of a normal chat bubble.
    is_crisis: bool = False
    # Observability — useful in dev, ignored by current frontend.
    route: Optional[str] = None
    route_reason: Optional[str] = None
    crisis_rules_fired: Optional[bool] = None
    crisis_llm_fired: Optional[bool] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        history = [m.model_dump() for m in req.history]
        result = await run_conversation(
            user_message=req.message, history=history
        )
        log.info(
            "chat :: is_crisis=%s route=%s reason=%s rules=%s llm=%s",
            result.get("is_crisis"),
            result.get("route"),
            result.get("route_reason"),
            result.get("crisis_rules_fired"),
            result.get("crisis_llm_fired"),
        )
        return ChatResponse(
            reply=result["reply"],
            is_crisis=bool(result.get("is_crisis", False)),
            route=result.get("route"),
            route_reason=result.get("route_reason"),
            crisis_rules_fired=result.get("crisis_rules_fired"),
            crisis_llm_fired=result.get("crisis_llm_fired"),
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
