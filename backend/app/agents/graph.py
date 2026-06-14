"""
The MannSaathi conversation graph.

Visual:
                   ┌──────────┐
                   │  START   │
                   └────┬─────┘
                        ▼
                   ┌──────────┐
                   │  triage  │   (1 LLM call — fast classifier)
                   └────┬─────┘
                        │
       ┌────────────────┼─────────────────┐
       ▼                ▼                 ▼
  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐
  │ listener │   │  reflector   │   │ advice_redirect  │
  └────┬─────┘   └──────┬───────┘   └─────────┬────────┘
       └────────────────┴─────────────────────┘
                        ▼
                   ┌──────────┐
                   │   END    │
                   └──────────┘

In Task #6 we'll insert a Crisis Detector node BEFORE triage as a
non-negotiable safety gate — if it fires, the whole graph short-circuits
to the crisis-response path and bypasses every LLM downstream.

Why a graph instead of just `if/else` in Python:
  - LangGraph gives us free observability (you can trace which nodes ran)
  - State updates are explicit and typed
  - Easy to add new nodes / branches without restructuring code
  - Composable: the graph itself is a runnable that we can wrap with
    checkpointing, retries, streaming, etc., in later tasks
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.advice_redirect import redirect_for_advice
from app.agents.listener import listen
from app.agents.reflector import reflect
from app.agents.state import ChatTurn, GraphState, Route, empty_state
from app.agents.triage import triage

log = logging.getLogger("mannsaathi.agents.graph")


# ---- Node implementations ----------------------------------------------------
# Each node receives the current GraphState and returns a PARTIAL update.
# LangGraph merges the partial back into the running state automatically.


async def _triage_node(state: GraphState) -> dict[str, Any]:
    route, reason = await triage(
        user_message=state["user_message"],
        history=state.get("history", []),
    )
    log.info("triage :: route=%s reason=%s", route, reason)
    return {"route": route, "route_reason": reason}


async def _listener_node(state: GraphState) -> dict[str, Any]:
    reply = await listen(
        history=state.get("history", []),
        user_message=state["user_message"],
    )
    return {"reply": reply}


async def _reflector_node(state: GraphState) -> dict[str, Any]:
    reply = await reflect(
        history=state.get("history", []),
        user_message=state["user_message"],
    )
    return {"reply": reply}


async def _advice_redirect_node(state: GraphState) -> dict[str, Any]:
    reply = await redirect_for_advice(
        history=state.get("history", []),
        user_message=state["user_message"],
    )
    return {"reply": reply}


# ---- Routing function -------------------------------------------------------
# After `triage`, LangGraph calls this to decide which node runs next.


def _select_agent(state: GraphState) -> Route:
    return state.get("route", "listener")


# ---- Graph assembly ---------------------------------------------------------


def _build_graph():
    g = StateGraph(GraphState)

    g.add_node("triage", _triage_node)
    g.add_node("listener", _listener_node)
    g.add_node("reflector", _reflector_node)
    g.add_node("advice_redirect", _advice_redirect_node)

    g.add_edge(START, "triage")

    # Conditional edge: after triage runs, pick which agent node executes.
    # The dict maps the Route enum value -> destination node name.
    g.add_conditional_edges(
        "triage",
        _select_agent,
        {
            "listener": "listener",
            "reflector": "reflector",
            "advice_redirect": "advice_redirect",
        },
    )

    # All agent nodes converge to END.
    g.add_edge("listener", END)
    g.add_edge("reflector", END)
    g.add_edge("advice_redirect", END)

    # `.compile()` returns a Runnable we can invoke per-request.
    return g.compile()


# Compile the graph once at module load — same lifecycle as the LLM client.
# (Each request reuses the compiled graph; only the state is per-request.)
_GRAPH = _build_graph()


async def run_conversation(
    user_message: str, history: list[ChatTurn]
) -> dict[str, Any]:
    """
    Entry point used by the chat API.

    Returns a dict with at least:
      - reply (str)        : the assistant response
      - route (str)        : which agent ran (for logging / future telemetry)
      - route_reason (str) : why the router picked that agent
    """
    initial = empty_state(user_message=user_message, history=history)
    final_state = await _GRAPH.ainvoke(initial)
    return {
        "reply": final_state.get("reply", ""),
        "route": final_state.get("route", "listener"),
        "route_reason": final_state.get("route_reason", ""),
    }
