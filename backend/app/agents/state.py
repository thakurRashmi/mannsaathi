"""
Typed state object that flows through the LangGraph.

Every node in the graph receives this state, can read from it, and returns
a partial update that LangGraph merges back in. Think of it as a typed
"context dict" — but with explicit fields, validated by Pydantic.

Why a typed state instead of a free-form dict:
  - You can see at a glance what data each node has access to
  - Adding a new field forces you to think about who sets it and who reads it
  - LangGraph's graph compiler can verify the shape end-to-end
"""
from typing import List, Literal, Optional, TypedDict


class ChatTurn(TypedDict):
    """One turn in the conversation history."""

    role: Literal["user", "assistant"]
    content: str


# The lanes Triage can route a message into.
# Each value maps to one agent node in graph.py.
Route = Literal["listener", "reflector", "advice_redirect"]


class GraphState(TypedDict, total=False):
    """
    The mutable state object that travels through every node in the graph.

    `total=False` means every field is optional — nodes can leave fields
    unset and downstream nodes will see them as missing. LangGraph merges
    each node's returned partial state into the running state.

    Fields:
      user_message: the new user input for this turn (set once, read by all)
      history:      prior conversation (set once, read by all)
      route:        which agent to invoke (set by Triage)
      route_reason: why Triage chose that route (debugging / observability)
      reply:        the final assistant response (set by whichever agent ran)
    """

    user_message: str
    history: List[ChatTurn]
    route: Route
    route_reason: str
    reply: str


def empty_state(user_message: str, history: List[ChatTurn]) -> GraphState:
    """Helper to construct the initial state for a new turn."""
    return GraphState(user_message=user_message, history=history)
