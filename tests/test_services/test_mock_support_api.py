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


class TestCreateTicket:

    def test_create_ticket_returns_dict(self):
        mock_db = MagicMock()
        mock_db.query.return_value \
            .filter.return_value.first.return_value = MagicMock()

        with patch("services.mock_support_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_support_api import create_ticket
            result = create_ticket(
                user_id    = "test_user_123",
                issue_type = "damaged_product",
                priority   = "HIGH",
                order_id   = "ORD-1001"
            )

        assert isinstance(result, dict)
