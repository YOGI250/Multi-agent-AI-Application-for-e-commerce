# subgraphs/escalation_handler.py

import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from tools.support_tools import (
    check_user_history_tool,
    create_ticket_tool
)
from langfuse_helpers.tracing import create_span, end_span

logger = logging.getLogger(__name__)


class EscalationState(TypedDict):
    user_id:                 str
    issue_type:              str
    order_id:                Optional[str]
    severity:                str
    user_history:            Optional[dict]
    is_repeat_complainant:   Optional[bool]
    ticket_priority:         Optional[str]
    ticket_id:               Optional[str]
    ticket_created:          Optional[bool]
    langfuse_trace_id:       Optional[str]
    langfuse_parent_span_id: Optional[str]


def check_user_history_node(
    state: EscalationState
) -> EscalationState:
    logger.info("Subgraph node: check_user_history")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    user_id   = state.get("user_id", "")

    span = create_span(
        trace_id              = trace_id,
        name                  = "check_user_history",
        parent_observation_id = parent_id,
        input_data            = {"user_id": user_id}
    ) if trace_id else None

    history = check_user_history_tool.invoke({
        "user_id":  user_id,
        "order_id": state.get("order_id")
    })
    state["user_history"]          = history
    state["is_repeat_complainant"] = history.get(
        "is_repeat_complainant", False
    )

    if span:
        end_span(span, {
            "total_complaints":    history.get("total_complaints", 0),
            "is_repeat":           state["is_repeat_complainant"]
        })

    return state


def assign_priority(state: EscalationState) -> EscalationState:
    logger.info("Subgraph node: assign_priority")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    severity  = state.get("severity", "LOW")
    is_repeat = state.get("is_repeat_complainant", False)

    span = create_span(
        trace_id              = trace_id,
        name                  = "assign_priority",
        parent_observation_id = parent_id,
        input_data            = {"severity": severity, "is_repeat": is_repeat}
    ) if trace_id else None

    if severity == "HIGH" and is_repeat:
        priority = "URGENT"
    elif severity == "HIGH":
        priority = "HIGH"
    elif severity == "MEDIUM" and is_repeat:
        priority = "HIGH"
    elif severity == "MEDIUM":
        priority = "MEDIUM"
    else:
        priority = "LOW"

    state["ticket_priority"] = priority

    if span:
        end_span(span, {"priority": priority})

    return state


def create_ticket_node(state: EscalationState) -> EscalationState:
    logger.info("Subgraph node: create_ticket")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "create_ticket",
        parent_observation_id = parent_id,
        input_data            = {
            "user_id":    state.get("user_id"),
            "issue_type": state.get("issue_type"),
            "priority":   state.get("ticket_priority")
        }
    ) if trace_id else None

    ticket = create_ticket_tool.invoke({
        "user_id":    state.get("user_id"),
        "issue_type": state.get("issue_type"),
        "priority":   state.get("ticket_priority"),
        "order_id":   state.get("order_id")
    })

    if ticket:
        state["ticket_id"]      = ticket.get("ticket_id")
        state["ticket_created"] = True
    else:
        state["ticket_id"]      = None
        state["ticket_created"] = False

    if span:
        end_span(span, {
            "ticket_id":      state["ticket_id"],
            "ticket_created": state["ticket_created"]
        })

    return state


def build_escalation_handler_subgraph():
    graph = StateGraph(EscalationState)

    graph.add_node("check_user_history", check_user_history_node)
    graph.add_node("assign_priority",    assign_priority)
    graph.add_node("create_ticket",      create_ticket_node)

    graph.set_entry_point("check_user_history")
    graph.add_edge("check_user_history", "assign_priority")
    graph.add_edge("assign_priority",    "create_ticket")
    graph.add_edge("create_ticket",      END)

    return graph.compile()


escalation_handler_subgraph = build_escalation_handler_subgraph()