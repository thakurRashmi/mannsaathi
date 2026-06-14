"""
The MannSaathi conversation graph.

Visual:
                   ┌──────────┐
                   │  START   │
                   └────┬─────┘
                        ▼
                ┌───────────────┐
                │ crisis_gate   │   safety-critical, ALWAYS first
                └───────┬───────┘
                        │
            ┌───────────┴───────────┐
            │ is_crisis?            │
            └───────┬───────────────┘
        true ┌──────┴──────┐ false
             ▼             ▼
       ┌──────────┐   ┌──────────┐
       │  crisis  │   │  triage  │
       │ response │   └────┬─────┘
       └────┬─────┘        │
            │       ┌──────┼──────────────┐
            │       ▼      ▼              ▼
            │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐
            │  │ listener │ │ reflector│ │ advice_redirect  │
            │  └────┬─────┘ └────┬─────┘ └─────────┬────────┘
            │       └────────────┼─────────────────┘
            └────────────────────┼─────────────────────────┐
                                 ▼                         │
                            ┌────────┐                     │
                            │  END   │ ◄───────────────────┘
                            └────────┘

Safety contract:
  When `crisis_gate` returns is_crisis=True, the graph short-circuits to
  `crisis_response`, which returns a HARD-CODED helpline message. No
  LLM is consulted to generate the user-facing reply. This is the entire
  point of the gate — to make it architecturally impossible for an LLM
  hallucination to reach a user in crisis.
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.advice_redirect import redirect_for_advice
from app.agents.crisis_gate import evaluate as crisis_evaluate
from app.agents.listener import listen
from app.agents.reflector import reflect
from app.agents.state import ChatTurn, GraphState, Route, empty_state
from app.agents.triage import triage

log = logging.getLogger("mannsaathi.agents.graph")


# ---- Node implementations ----------------------------------------------------


async def _crisis_gate_node(state: GraphState) -> dict[str, Any]:
    """Run the hybrid rules+LLM crisis check before anything else."""
    decision = await crisis_evaluate(state["user_message"])
    update: dict[str, Any] = {
        "is_crisis": decision.is_crisis,
        "crisis_rules_fired": decision.rules_fired,
        "crisis_llm_fired": decision.llm_fired,
    }
    # If it's crisis, also set the response now — it's a hard-coded string.
    # This makes the `crisis_response` node essentially a no-op pass-through,
    # but keeping it as a separate node makes the graph diagram readable.
    if decision.is_crisis and decision.response_text:
        update["reply"] = decision.response_text
    return update


async def _crisis_response_node(state: GraphState) -> dict[str, Any]:
    """
    Crisis path terminal node. We keep it as a separate node (rather than
    going directly from the gate to END) so the graph topology clearly
    shows crisis as a distinct branch.

    IMPORTANT: every node MUST return at least one state field. LangGraph
    raises InvalidUpdateError on an empty {} update (with a confusingly
    worded error message). So we always return `reply` here — either
    re-affirming what the gate set, or falling back to the canned text.
    """
    from app.agents.crisis_gate import CRISIS_RESPONSE_TEXT

    return {"reply": state.get("reply") or CRISIS_RESPONSE_TEXT}


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


# ---- Routing functions ------------------------------------------------------


def _after_crisis_gate(state: GraphState) -> str:
    """Branch from crisis_gate: 'crisis_response' if flagged, else 'triage'."""
    return "crisis_response" if state.get("is_crisis") else "triage"


def _select_agent(state: GraphState) -> Route:
    """Branch from triage: route to the picked specialized agent."""
    return state.get("route", "listener")


# ---- Graph assembly ---------------------------------------------------------


def _build_graph():
    g = StateGraph(GraphState)

    # Nodes
    g.add_node("crisis_gate", _crisis_gate_node)
    g.add_node("crisis_response", _crisis_response_node)
    g.add_node("triage", _triage_node)
    g.add_node("listener", _listener_node)
    g.add_node("reflector", _reflector_node)
    g.add_node("advice_redirect", _advice_redirect_node)

    # Entry: always go through the crisis gate first.
    g.add_edge(START, "crisis_gate")

    # After the gate: either short-circuit to crisis_response, or proceed.
    g.add_conditional_edges(
        "crisis_gate",
        _after_crisis_gate,
        {
            "crisis_response": "crisis_response",
            "triage": "triage",
        },
    )

    # Crisis path goes straight to END (no LLM in the user-facing reply).
    g.add_edge("crisis_response", END)

    # Normal path: triage -> one of three agents -> END.
    g.add_conditional_edges(
        "triage",
        _select_agent,
        {
            "listener": "listener",
            "reflector": "reflector",
            "advice_redirect": "advice_redirect",
        },
    )
    g.add_edge("listener", END)
    g.add_edge("reflector", END)
    g.add_edge("advice_redirect", END)

    return g.compile()


_GRAPH = _build_graph()


async def run_conversation(
    user_message: str, history: list[ChatTurn]
) -> dict[str, Any]:
    """
    Entry point used by the chat API.

    Returns a dict with at least:
      reply           (str)  : the assistant response
      is_crisis       (bool) : whether the safety gate fired
      route           (str)  : which agent answered (or "crisis_response")
      route_reason    (str)  : why
      crisis_rules_fired (bool)
      crisis_llm_fired   (bool)
    """
    initial = empty_state(user_message=user_message, history=history)
    final_state = await _GRAPH.ainvoke(initial)

    is_crisis = bool(final_state.get("is_crisis", False))
    return {
        "reply": final_state.get("reply", ""),
        "is_crisis": is_crisis,
        "route": "crisis_response" if is_crisis else final_state.get("route", "listener"),
        "route_reason": final_state.get("route_reason", ""),
        "crisis_rules_fired": bool(final_state.get("crisis_rules_fired", False)),
        "crisis_llm_fired": bool(final_state.get("crisis_llm_fired", False)),
    }
