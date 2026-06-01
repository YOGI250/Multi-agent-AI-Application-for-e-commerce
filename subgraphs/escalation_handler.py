# subgraphs/escalation_handler.py

import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from tools.support_tools import check_user_history_tool, create_ticket_tool
from langfuse_helpers.tracing import create_span, end_span

logger = logging.getLogger(__name__)


class EscalationState(TypedDict):
    user_id: str
    issue_type: str
    order_id: Optional[str]
    severity: str
    is_duplicate: Optional[bool]
    days_open: Optional[int]
    ticket_id: Optional[str]
    ticket_created: Optional[bool]
    langfuse_trace_id: Optional[str]
    langfuse_parent_span_id: Optional[str]


def check_user_history_node(state: EscalationState) -> EscalationState:
    logger.info("Subgraph node: check_user_history")

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    user_id = state.get("user_id", "")

    span = (
        create_span(
            trace_id=trace_id,
            name="check_user_history",
            parent_observation_id=parent_id,
            input_data={"user_id": user_id, "order_id": state.get("order_id")},
        )
        if trace_id
        else None
    )

    history = check_user_history_tool.invoke({"user_id": user_id, "order_id": state.get("order_id")})

    state["is_duplicate"] = history.get("is_duplicate", False)
    state["days_open"] = history.get("days_open", 0)

    # Pre-populate ticket_id so create_ticket_node can use it without a second DB hit
    if history.get("is_duplicate"):
        state["ticket_id"] = history.get("existing_ticket_id")

    if span:
        end_span(span, {"is_duplicate": state["is_duplicate"], "days_open": state["days_open"]})

    return state


def create_ticket_node(state: EscalationState) -> EscalationState:
    logger.info("Subgraph node: create_ticket")

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = (
        create_span(
            trace_id=trace_id,
            name="create_ticket",
            parent_observation_id=parent_id,
            input_data={
                "user_id": state.get("user_id"),
                "issue_type": state.get("issue_type"),
                "is_duplicate": state.get("is_duplicate"),
            },
        )
        if trace_id
        else None
    )

    if state.get("is_duplicate"):
        # Ticket already open — do not create a second one
        state["ticket_created"] = False
    else:
        # New complaint — create ticket using severity directly as priority
        ticket = create_ticket_tool.invoke(
            {
                "user_id": state.get("user_id"),
                "issue_type": state.get("issue_type"),
                "priority": state.get("severity"),
                "order_id": state.get("order_id"),
            }
        )

        if ticket:
            state["ticket_id"] = ticket.get("ticket_id")
            state["ticket_created"] = True
        else:
            state["ticket_id"] = None
            state["ticket_created"] = False

    if span:
        end_span(
            span,
            {
                "ticket_id": state.get("ticket_id"),
                "ticket_created": state.get("ticket_created"),
                "is_duplicate": state.get("is_duplicate"),
            },
        )

    return state


def build_escalation_handler_subgraph():
    graph = StateGraph(EscalationState)

    graph.add_node("check_user_history", check_user_history_node)
    graph.add_node("create_ticket", create_ticket_node)

    graph.set_entry_point("check_user_history")
    graph.add_edge("check_user_history", "create_ticket")
    graph.add_edge("create_ticket", END)

    return graph.compile()


escalation_handler_subgraph = build_escalation_handler_subgraph()
