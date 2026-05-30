# tests/test_subgraphs/test_escalation_handler.py

import pytest
from unittest.mock import patch


def _state(**kwargs):
    base = {
        "user_id":                 "test_user",
        "issue_type":              "damaged_product",
        "order_id":                "ORD-001",
        "severity":                "HIGH",
        "is_duplicate":            None,
        "days_open":               None,
        "ticket_id":               None,
        "ticket_created":          None,
        "langfuse_trace_id":       None,
        "langfuse_parent_span_id": None,
    }
    base.update(kwargs)
    return base


class TestCheckUserHistoryNode:

    def test_marks_duplicate_when_ticket_exists(self):
        from subgraphs.escalation_handler import check_user_history_node
        mock_history = {
            "existing_ticket_id": "TKT-001",
            "is_duplicate":       True,
            "days_open":          3
        }
        with patch("subgraphs.escalation_handler.check_user_history_tool") as mock_tool:
            mock_tool.invoke.return_value = mock_history
            result = check_user_history_node(_state())
        assert result["is_duplicate"] is True
        assert result["ticket_id"] == "TKT-001"
        assert result["days_open"] == 3

    def test_marks_not_duplicate_for_new_user(self):
        from subgraphs.escalation_handler import check_user_history_node
        mock_history = {
            "existing_ticket_id": None,
            "is_duplicate":       False,
            "days_open":          0
        }
        with patch("subgraphs.escalation_handler.check_user_history_tool") as mock_tool:
            mock_tool.invoke.return_value = mock_history
            result = check_user_history_node(_state())
        assert result["is_duplicate"] is False
        assert result["days_open"] == 0

    def test_does_not_overwrite_ticket_id_when_not_duplicate(self):
        from subgraphs.escalation_handler import check_user_history_node
        mock_history = {
            "existing_ticket_id": None,
            "is_duplicate":       False,
            "days_open":          0
        }
        with patch("subgraphs.escalation_handler.check_user_history_tool") as mock_tool:
            mock_tool.invoke.return_value = mock_history
            result = check_user_history_node(_state(ticket_id=None))
        assert result["ticket_id"] is None


class TestCreateTicketNode:

    def test_skips_creation_for_duplicate(self):
        from subgraphs.escalation_handler import create_ticket_node
        result = create_ticket_node(_state(is_duplicate=True, ticket_id="TKT-001"))
        assert result["ticket_created"] is False

    def test_creates_ticket_for_new_complaint(self):
        from subgraphs.escalation_handler import create_ticket_node
        mock_ticket = {"ticket_id": "TKT-999", "status": "open"}
        with patch("subgraphs.escalation_handler.create_ticket_tool") as mock_tool:
            mock_tool.invoke.return_value = mock_ticket
            result = create_ticket_node(_state(is_duplicate=False))
        assert result["ticket_created"] is True
        assert result["ticket_id"] == "TKT-999"

    def test_handles_empty_ticket_response(self):
        from subgraphs.escalation_handler import create_ticket_node
        with patch("subgraphs.escalation_handler.create_ticket_tool") as mock_tool:
            mock_tool.invoke.return_value = {}
            result = create_ticket_node(_state(is_duplicate=False))
        assert result["ticket_created"] is False
        assert result["ticket_id"] is None


class TestEscalationHandlerSubgraph:

    def test_subgraph_compiles(self):
        from subgraphs.escalation_handler import escalation_handler_subgraph
        assert escalation_handler_subgraph is not None

    def test_full_flow_new_complaint(self):
        from subgraphs.escalation_handler import escalation_handler_subgraph
        with patch("subgraphs.escalation_handler.check_user_history_tool") as mock_hist, \
             patch("subgraphs.escalation_handler.create_ticket_tool") as mock_create:
            mock_hist.invoke.return_value = {
                "existing_ticket_id": None,
                "is_duplicate":       False,
                "days_open":          0
            }
            mock_create.invoke.return_value = {
                "ticket_id": "TKT-NEW",
                "status":    "open"
            }
            result = escalation_handler_subgraph.invoke(_state())
        assert result["ticket_created"] is True
        assert result["ticket_id"] == "TKT-NEW"

    def test_full_flow_duplicate_complaint(self):
        from subgraphs.escalation_handler import escalation_handler_subgraph
        with patch("subgraphs.escalation_handler.check_user_history_tool") as mock_hist:
            mock_hist.invoke.return_value = {
                "existing_ticket_id": "TKT-EXISTING",
                "is_duplicate":       True,
                "days_open":          2
            }
            result = escalation_handler_subgraph.invoke(_state())
        assert result["ticket_created"] is False
        assert result["ticket_id"] == "TKT-EXISTING"
