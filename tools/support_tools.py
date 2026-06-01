# tools/support_tools.py
import logging
from typing import Optional
from langchain_core.tools import tool
from services.mock_support_api import (
    get_policy,
    get_user_complaint_history,
    create_ticket
)

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

    result = get_policy(issue_type)

    if not result:
        logger.warning(f"No policy found for: {issue_type}")
        return {
            "issue_type": issue_type,
            "policy_text": "Standard return policy applies. Contact support for assistance."
        }

    logger.info(f"Policy fetched for: {issue_type}")
    return result


@tool
def check_user_history_tool(user_id: str, order_id: Optional[str] = None) -> dict:
    """
    Checks if an open ticket already exists for this user + order.
    Returns is_duplicate (bool), days_open (int), existing_ticket_id (str or None).
    Used by escalation_handler subgraph to detect duplicate complaints.
    """
    logger.info(
        f"Tool called: check_user_history_tool for "
        f"user_id={user_id} order_id={order_id}"
    )

    result = get_user_complaint_history(user_id, order_id=order_id)

    logger.info(
        f"History for {user_id}: is_duplicate={result['is_duplicate']}, "
        f"days_open={result['days_open']}"
    )

    return result


@tool
def create_ticket_tool(
    user_id: str,
    issue_type: str,
    priority: str,
    order_id: Optional[str] = None
) -> dict:
    """
    Creates a new support ticket in the database.
    Only called for HIGH severity complaints.
    Returns the ticket_id which is included in the
    final response to the user as a reference number.
    Priority levels: URGENT, HIGH, MEDIUM, LOW.
    """
    logger.info(
        f"Tool called: create_ticket_tool for user={user_id} "
        f"issue={issue_type} priority={priority}"
    )

    result = create_ticket(
        user_id=user_id,
        issue_type=issue_type,
        priority=priority,
        order_id=order_id
    )

    if result:
        logger.info(f"Ticket created successfully: {result.get('ticket_id')}")
    else:
        logger.error(f"Failed to create ticket for user {user_id}")

    return result