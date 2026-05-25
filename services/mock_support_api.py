# services/mock_support_api.py

import logging
import uuid
from typing import Optional
from datetime import datetime
from database.connection import SessionLocal
from database.models import Policy, SupportTicket

logger = logging.getLogger(__name__)


def get_policy(issue_type: str) -> Optional[dict]:
    """
    Fetches the policy text for a given issue type.
    Called by lookup_policy tool node in Support Agent.
    Injects real policy text into LLM prompt to prevent
    hallucination of policy details.
    """
    db = SessionLocal()
    try:
        policy = db.query(Policy).filter(
            Policy.issue_type == issue_type
        ).first()

        if not policy:
            logger.warning(f"Policy not found for issue_type: {issue_type}")
            # Return a generic policy if specific one not found
            return {
                "issue_type":   issue_type,
                "policy_text":  "Please contact our support team for assistance. We will resolve your issue within 5-7 business days.",
                "updated_at":   str(datetime.utcnow())
            }

        return {
            "issue_type":   policy.issue_type,
            "policy_text":  policy.policy_text,
            "updated_at":   str(policy.updated_at)
        }

    except Exception as e:
        logger.error(f"Error fetching policy for {issue_type}: {e}")
        return None
    finally:
        db.close()


def get_user_complaint_history(user_id: str) -> dict:
    """
    Returns complaint history for a user.
    Called by check_user_history tool node inside
    escalation_handler subgraph.
    Used to determine if user is a repeat complainant
    and set ticket priority accordingly.
    """
    db = SessionLocal()
    try:
        # Count past support tickets
        tickets = db.query(SupportTicket).filter(
            SupportTicket.user_id == user_id
        ).order_by(SupportTicket.created_at.desc()).all()

        total_complaints = len(tickets)

        recent_tickets = [
            {
                "ticket_id":  t.ticket_id,
                "issue_type": t.issue_type,
                "priority":   t.priority,
                "status":     t.status,
                "created_at": str(t.created_at)
            }
            for t in tickets[:5]
        ]

        is_repeat_complainant = total_complaints >= 3

        logger.info(
            f"User {user_id} has {total_complaints} past complaints. "
            f"Repeat: {is_repeat_complainant}"
        )

        return {
            "user_id":               user_id,
            "total_complaints":      total_complaints,
            "recent_tickets":        recent_tickets,
            "is_repeat_complainant": is_repeat_complainant
        }

    except Exception as e:
        logger.error(f"Error fetching complaint history for {user_id}: {e}")
        return {
            "user_id":               user_id,
            "total_complaints":      0,
            "recent_tickets":        [],
            "is_repeat_complainant": False
        }
    finally:
        db.close()


def create_ticket(
    user_id: str,
    issue_type: str,
    priority: str,
    order_id: Optional[str] = None
) -> dict:
    """
    Creates a new support ticket in the database.
    Called by create_ticket tool node inside
    escalation_handler subgraph.
    Returns ticket_id which is included in the
    final response to the user.
    """
    db = SessionLocal()
    try:
        ticket = SupportTicket(
            ticket_id=str(uuid.uuid4())[:8].upper(),
            user_id=user_id,
            order_id=order_id,
            issue_type=issue_type,
            priority=priority,
            status="open",
            created_at=datetime.utcnow()
        )
        db.add(ticket)
        db.commit()

        logger.info(
            f"Ticket created: {ticket.ticket_id} "
            f"for user {user_id} with priority {priority}"
        )

        return {
            "ticket_id":  ticket.ticket_id,
            "user_id":    ticket.user_id,
            "issue_type": ticket.issue_type,
            "priority":   ticket.priority,
            "status":     ticket.status,
            "created_at": str(ticket.created_at)
        }

    except Exception as e:
        logger.error(f"Error creating ticket for {user_id}: {e}")
        db.rollback()
        return {}
    finally:
        db.close()