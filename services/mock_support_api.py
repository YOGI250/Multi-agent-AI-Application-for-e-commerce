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


def get_user_complaint_history(user_id: str, order_id: Optional[str] = None) -> dict:
    """
    Returns complaint history scoped to a specific order.
    A user is a repeat complainant only if the same order
    already has an open unresolved ticket — not based on
    total lifetime complaint count.
    A user who complains about 10 different orders is just
    an active customer. A user coming back about the SAME
    unresolved order is the one who needs escalation.
    """
    db = SessionLocal()
    try:
        query = db.query(SupportTicket).filter(
            SupportTicket.user_id == user_id
        )
        if order_id:
            query = query.filter(SupportTicket.order_id == order_id)

        tickets = query.order_by(SupportTicket.created_at.desc()).all()

        total_complaints = len(tickets)

        recent_tickets = [
            {
                "ticket_id":  t.ticket_id,
                "issue_type": t.issue_type,
                "priority":   t.priority,
                "status":     t.status,
                "created_at": str(t.created_at),
                "order_id":   t.order_id
            }
            for t in tickets[:5]
        ]

        # Repeat means this specific order has an existing open ticket
        has_open_ticket = any(
            t.status in ("open", "in_progress")
            for t in tickets
        )
        is_repeat_complainant = bool(order_id and has_open_ticket)

        logger.info(
            f"User {user_id} has {total_complaints} complaints "
            f"{'for order ' + order_id if order_id else 'total'}. "
            f"Repeat: {is_repeat_complainant}"
        )

        return {
            "user_id":               user_id,
            "order_id":              order_id,
            "total_complaints":      total_complaints,
            "recent_tickets":        recent_tickets,
            "is_repeat_complainant": is_repeat_complainant
        }

    except Exception as e:
        logger.error(f"Error fetching complaint history for {user_id}: {e}")
        return {
            "user_id":               user_id,
            "order_id":              order_id,
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
    Before creating, checks if an open ticket already exists
    for this user + order + issue combination to prevent duplicates.
    If one exists, returns the existing ticket with is_duplicate=True.
    """
    db = SessionLocal()
    try:
        # Check for existing open ticket for same user + order + issue
        existing_query = db.query(SupportTicket).filter(
            SupportTicket.user_id    == user_id,
            SupportTicket.issue_type == issue_type,
            SupportTicket.status     == "open"
        )
        if order_id:
            existing_query = existing_query.filter(
                SupportTicket.order_id == order_id
            )
        existing_ticket = existing_query.first()

        if existing_ticket:
            logger.info(
                f"Open ticket already exists: {existing_ticket.ticket_id} "
                f"for user {user_id} — skipping duplicate creation"
            )
            return {
                "ticket_id":    existing_ticket.ticket_id,
                "user_id":      existing_ticket.user_id,
                "issue_type":   existing_ticket.issue_type,
                "priority":     existing_ticket.priority,
                "status":       existing_ticket.status,
                "created_at":   str(existing_ticket.created_at),
                "is_duplicate": True
            }

        # No existing open ticket — create a new one
        ticket = SupportTicket(
            ticket_id  = str(uuid.uuid4())[:8].upper(),
            user_id    = user_id,
            order_id   = order_id,
            issue_type = issue_type,
            priority   = priority,
            status     = "open",
            created_at = datetime.utcnow()
        )
        db.add(ticket)
        db.commit()

        logger.info(
            f"Ticket created: {ticket.ticket_id} "
            f"for user {user_id} with priority {priority}"
        )

        return {
            "ticket_id":    ticket.ticket_id,
            "user_id":      ticket.user_id,
            "issue_type":   ticket.issue_type,
            "priority":     ticket.priority,
            "status":       ticket.status,
            "created_at":   str(ticket.created_at),
            "is_duplicate": False
        }

    except Exception as e:
        logger.error(f"Error creating ticket for {user_id}: {e}")
        db.rollback()
        return {}
    finally:
        db.close()