# tests/test_tools/test_support_tools.py

import pytest
from unittest.mock import patch, MagicMock
from utils.langfuse_context import set_trace_context


class TestLookupPolicyTool:

    def test_lookup_returns_policy(self):
        mock_policy = {"issue_type": "damaged_product", "policy_text": "We offer full refund for damaged products."}
        with patch("tools.support_tools.get_policy", return_value=mock_policy):
            from tools.support_tools import lookup_policy_tool

            result = lookup_policy_tool.invoke({"issue_type": "damaged_product"})
        assert isinstance(result, dict)

    def test_lookup_returns_dict_for_unknown(self):
        with patch("tools.support_tools.get_policy", return_value=None):
            from tools.support_tools import lookup_policy_tool

            result = lookup_policy_tool.invoke({"issue_type": "unknown_issue"})
        assert isinstance(result, dict)


class TestCheckUserHistoryTool:

    def test_check_history_returns_dict(self):
        mock_history = {"existing_ticket_id": "TKT-001", "is_duplicate": True, "days_open": 2}
        with patch("tools.support_tools.get_user_complaint_history", return_value=mock_history):
            from tools.support_tools import check_user_history_tool

            result = check_user_history_tool.invoke({"user_id": "test_user_123"})
        assert isinstance(result, dict)

    def test_check_history_returns_dict_for_new_user(self):
        mock_history = {"existing_ticket_id": None, "is_duplicate": False, "days_open": 0}
        with patch("tools.support_tools.get_user_complaint_history", return_value=mock_history):
            from tools.support_tools import check_user_history_tool

            result = check_user_history_tool.invoke({"user_id": "new_user_123"})
        assert isinstance(result, dict)


class TestCreateTicketTool:

    def test_create_ticket_returns_result(self):
        mock_result = {"ticket_id": "TKT-001", "status": "open"}
        with patch("tools.support_tools.create_ticket", return_value=mock_result):
            from tools.support_tools import create_ticket_tool

            result = create_ticket_tool.invoke(
                {
                    "user_id": "test_user_123",
                    "issue_type": "damaged_product",
                    "priority": "HIGH",
                    "order_id": "ORD-1001",
                }
            )
        assert isinstance(result, dict)

    def test_create_ticket_without_order_id(self):
        mock_result = {"ticket_id": "TKT-002", "status": "open"}
        with patch("tools.support_tools.create_ticket", return_value=mock_result):
            from tools.support_tools import create_ticket_tool

            result = create_ticket_tool.invoke(
                {"user_id": "test_user_123", "issue_type": "refund", "priority": "MEDIUM"}
            )
        assert isinstance(result, dict)


class TestSupportToolsWithSpan:

    def setup_method(self):
        set_trace_context("test-trace-id", "test-span-id")

    def teardown_method(self):
        set_trace_context(None, None)

    def test_lookup_policy_with_span(self):
        mock_policy = {"issue_type": "refund", "policy_text": "Full refund within 7 days."}
        mock_span = MagicMock()
        with patch("tools.support_tools.get_policy", return_value=mock_policy), patch(
            "tools.support_tools.create_span", return_value=mock_span
        ), patch("tools.support_tools.end_span") as mock_end:
            from tools.support_tools import lookup_policy_tool

            result = lookup_policy_tool.invoke({"issue_type": "refund"})

        assert isinstance(result, dict)
        mock_end.assert_called_once()

    def test_check_user_history_with_span(self):
        mock_history = {"existing_ticket_id": None, "is_duplicate": False, "days_open": 0}
        mock_span = MagicMock()
        with patch("tools.support_tools.get_user_complaint_history", return_value=mock_history), patch(
            "tools.support_tools.create_span", return_value=mock_span
        ), patch("tools.support_tools.end_span") as mock_end:
            from tools.support_tools import check_user_history_tool

            result = check_user_history_tool.invoke({"user_id": "test_user_123"})

        assert isinstance(result, dict)
        mock_end.assert_called_once()

    def test_create_ticket_success_with_span(self):
        mock_result = {"ticket_id": "TKT-001", "status": "open"}
        mock_span = MagicMock()
        with patch("tools.support_tools.create_ticket", return_value=mock_result), patch(
            "tools.support_tools.create_span", return_value=mock_span
        ), patch("tools.support_tools.end_span") as mock_end:
            from tools.support_tools import create_ticket_tool

            result = create_ticket_tool.invoke(
                {
                    "user_id": "test_user_123",
                    "issue_type": "damaged_product",
                    "priority": "HIGH",
                    "order_id": "ORD-1001",
                }
            )

        assert isinstance(result, dict)
        mock_end.assert_called_once()

    def test_create_ticket_failure_with_span(self):
        mock_span = MagicMock()
        with patch("tools.support_tools.create_ticket", return_value=None), patch(
            "tools.support_tools.create_span", return_value=mock_span
        ), patch("tools.support_tools.end_span") as mock_end:
            from tools.support_tools import create_ticket_tool

            result = create_ticket_tool.invoke(
                {"user_id": "test_user_123", "issue_type": "refund", "priority": "MEDIUM"}
            )

        assert result is None
        mock_end.assert_called_once()
