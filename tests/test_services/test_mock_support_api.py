# tests/test_services/test_mock_support_api.py

import pytest
from unittest.mock import patch, MagicMock


class TestGetPolicy:

    def test_get_policy_returns_dict(self):
        mock_policy           = MagicMock()
        mock_policy.issue_type   = "damaged_product"
        mock_policy.policy_text  = "Full refund for damaged items."
        mock_policy.resolution_steps = ["Step 1", "Step 2"]
        mock_policy.escalate_after_days = 3

        mock_db = MagicMock()
        mock_db.query.return_value \
            .filter.return_value.first.return_value = mock_policy

        with patch("services.mock_support_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_support_api import get_policy
            result = get_policy("damaged_product")

        assert isinstance(result, dict)

    def test_get_policy_returns_dict_for_unknown(self):
        mock_db = MagicMock()
        mock_db.query.return_value \
            .filter.return_value.first.return_value = None

        with patch("services.mock_support_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_support_api import get_policy
            result = get_policy("unknown_issue")

        assert isinstance(result, dict)


class TestGetUserComplaintHistory:

    def test_returns_history_dict(self):
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value           = mock_query
        mock_query.filter.return_value       = mock_query
        mock_query.order_by.return_value     = mock_query
        mock_query.limit.return_value        = mock_query
        mock_query.count.return_value        = 2
        mock_query.all.return_value          = []

        with patch("services.mock_support_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_support_api import get_user_complaint_history
            result = get_user_complaint_history("test_user_123")

        assert isinstance(result, dict)

    def test_returns_zero_complaints_for_new_user(self):
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value       = mock_query
        mock_query.filter.return_value   = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value    = mock_query
        mock_query.count.return_value    = 0
        mock_query.all.return_value      = []

        with patch("services.mock_support_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_support_api import get_user_complaint_history
            result = get_user_complaint_history("new_user_123")

        assert isinstance(result, dict)


    def test_get_policy_exception_returns_none(self):
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("DB error")
        with patch("services.mock_support_api.SessionLocal", return_value=mock_db):
            from services.mock_support_api import get_policy
            result = get_policy("damaged_product")
        assert result is None


class TestGetUserComplaintHistory:

    def test_returns_history_dict(self):
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value           = mock_query
        mock_query.filter.return_value       = mock_query
        mock_query.order_by.return_value     = mock_query
        mock_query.limit.return_value        = mock_query
        mock_query.count.return_value        = 2
        mock_query.all.return_value          = []

        with patch("services.mock_support_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_support_api import get_user_complaint_history
            result = get_user_complaint_history("test_user_123")

        assert isinstance(result, dict)

    def test_returns_zero_complaints_for_new_user(self):
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value       = mock_query
        mock_query.filter.return_value   = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value    = mock_query
        mock_query.count.return_value    = 0
        mock_query.all.return_value      = []

        with patch("services.mock_support_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_support_api import get_user_complaint_history
            result = get_user_complaint_history("new_user_123")

        assert isinstance(result, dict)

    def test_filters_by_order_id_when_provided(self):
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value       = mock_query
        mock_query.filter.return_value   = mock_query
        mock_query.first.return_value    = None

        with patch("services.mock_support_api.SessionLocal", return_value=mock_db):
            from services.mock_support_api import get_user_complaint_history
            result = get_user_complaint_history("u1", order_id="ORD-001")

        assert result["is_duplicate"] == False

    def test_returns_existing_ticket_when_found(self):
        from datetime import datetime, timezone
        mock_ticket            = MagicMock()
        mock_ticket.ticket_id  = "TK-001"
        mock_ticket.created_at = datetime(2026, 5, 1, tzinfo=timezone.utc)

        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value       = mock_query
        mock_query.filter.return_value   = mock_query
        mock_query.first.return_value    = mock_ticket

        with patch("services.mock_support_api.SessionLocal", return_value=mock_db):
            from services.mock_support_api import get_user_complaint_history
            result = get_user_complaint_history("u1")

        assert result["is_duplicate"]       == True
        assert result["existing_ticket_id"] == "TK-001"
        assert result["days_open"]          >= 0

    def test_handles_naive_datetime(self):
        from datetime import datetime
        mock_ticket            = MagicMock()
        mock_ticket.ticket_id  = "TK-002"
        # naive datetime (no tzinfo)
        mock_ticket.created_at = datetime(2026, 5, 1)

        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value    = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value  = mock_ticket

        with patch("services.mock_support_api.SessionLocal", return_value=mock_db):
            from services.mock_support_api import get_user_complaint_history
            result = get_user_complaint_history("u2")

        assert result["is_duplicate"] == True

    def test_exception_returns_no_duplicate(self):
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("DB error")
        with patch("services.mock_support_api.SessionLocal", return_value=mock_db):
            from services.mock_support_api import get_user_complaint_history
            result = get_user_complaint_history("u3")
        assert result["is_duplicate"] == False


class TestCreateTicket:

    def test_returns_existing_ticket_when_duplicate(self):
        mock_existing            = MagicMock()
        mock_existing.ticket_id  = "TK-DUP"
        mock_existing.user_id    = "test_user"
        mock_existing.issue_type = "damaged_product"
        mock_existing.priority   = "HIGH"
        mock_existing.status     = "open"
        mock_existing.created_at = "2026-05-01"

        # Use a self-returning mock_query so chained .filter() calls work
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value  = mock_existing
        mock_db.query.return_value     = mock_query

        with patch("services.mock_support_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_support_api import create_ticket
            result = create_ticket(
                user_id    = "test_user",
                issue_type = "damaged_product",
                priority   = "HIGH",
                order_id   = "ORD-1001"
            )

        assert result["is_duplicate"] == True
        assert result["ticket_id"]    == "TK-DUP"

    def test_creates_new_ticket_when_no_duplicate(self):
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value     = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value  = None  # no existing ticket

        with patch("services.mock_support_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_support_api import create_ticket
            result = create_ticket(
                user_id    = "new_user",
                issue_type = "missing_item",
                priority   = "MEDIUM"
            )

        assert isinstance(result, dict)
        assert result.get("is_duplicate") == False

    def test_create_ticket_exception_returns_empty_dict(self):
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("DB error")
        with patch("services.mock_support_api.SessionLocal", return_value=mock_db):
            from services.mock_support_api import create_ticket
            result = create_ticket("u1", "damaged_product", "HIGH")
        assert result == {}
