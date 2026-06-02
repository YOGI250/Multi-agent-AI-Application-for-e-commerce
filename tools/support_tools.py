# tools/support_tools.py
import logging
from typing import Optional
from langchain_core.tools import tool
from services.mock_support_api import get_policy, get_user_complaint_history, create_ticket
from utils.langfuse_context import get_trace_context
from langfuse_helpers.tracing import create_span, end_span

logger = logging.getLogger(__name__)


@tool
def lookup_policy_tool(issue_type: str) -> dict:
    """
    Fetches the company policy text for a given issue type.
    Issue types: damaged_product, wrong_item, refund, cancellation.
    The policy text is injected into the LLM resolution prompt
    to prevent hallucination of policy details.
    Returns dict with issue_type and policy_text.
    """
    logger.info(f"Tool called: lookup_policy_tool for issue_type={issue_type}")
    trace_id, parent_span_id = get_trace_context()
    span = create_span(trace_id, "tool:lookup_policy", parent_observation_id=parent_span_id,
                       input_data={"issue_type": issue_type}) if trace_id else None

    result = get_policy(issue_type)

    if not result:
        logger.warning(f"No policy found for: {issue_type}")
        result = {
            "issue_type": issue_type,
            "policy_text": "Standard return policy applies. Contact support for assistance.",
        }

    logger.info(f"Policy fetched for: {issue_type}")
    if span:
        end_span(span, {"issue_type": issue_type, "policy_length": len(result.get("policy_text", ""))})
    return result


@tool
def check_user_history_tool(user_id: str, order_id: Optional[str] = None) -> dict:
    """
    Checks if an open ticket already exists for this user + order.
    Returns is_duplicate (bool), days_open (int), existing_ticket_id (str or None).
    Used by escalation_handler subgraph to detect duplicate complaints.
    """
    logger.info(f"Tool called: check_user_history_tool for user_id={user_id} order_id={order_id}")
    trace_id, parent_span_id = get_trace_context()
    span = create_span(trace_id, "tool:check_user_history", parent_observation_id=parent_span_id,
                       input_data={"user_id": user_id, "order_id": order_id}) if trace_id else None

    result = get_user_complaint_history(user_id, order_id=order_id)

    logger.info(f"History for {user_id}: is_duplicate={result['is_duplicate']}, days_open={result['days_open']}")
    if span:
        end_span(span, {"is_duplicate": result.get("is_duplicate"), "days_open": result.get("days_open")})
    return result


@tool
def create_ticket_tool(user_id: str, issue_type: str, priority: str, order_id: Optional[str] = None) -> dict:
    """
    Creates a new support ticket in the database.
    Called for HIGH and MEDIUM severity complaints.
    Returns the ticket_id which is included in the
    final response to the user as a reference number.
    Priority levels: URGENT, HIGH, MEDIUM, LOW.
    """
    logger.info(f"Tool called: create_ticket_tool for user={user_id} issue={issue_type} priority={priority}")
    trace_id, parent_span_id = get_trace_context()
    span = create_span(trace_id, "tool:create_ticket", parent_observation_id=parent_span_id,
                       input_data={"user_id": user_id, "issue_type": issue_type,
                                   "priority": priority, "order_id": order_id}) if trace_id else None

    result = create_ticket(user_id=user_id, issue_type=issue_type, priority=priority, order_id=order_id)

    if result:
        logger.info(f"Ticket created successfully: {result.get('ticket_id')}")
        if span:
            end_span(span, {"ticket_id": result.get("ticket_id"), "success": True})
    else:
        logger.error(f"Failed to create ticket for user {user_id}")
        if span:
            end_span(span, {"success": False})

    return result
