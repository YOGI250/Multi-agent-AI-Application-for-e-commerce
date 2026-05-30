# tests/test_agents/test_support_agent.py

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def base_state():
    return {
        "message":                 "my product arrived damaged",
        "user_id":                 "test_user_123",
        "session_id":              "test_session_123",
        "history":                 [],
        "is_authenticated":        True,
        "langfuse_trace_id":       None,
        "langfuse_parent_span_id": None,
        "issue_type":              None,
        "severity":                None,
        "policy_data":             {},
        "user_history":            {},
        "resolution":              None,
        "response":                None,
        "agent_used":              None
    }

@pytest.fixture
def state_with_issue(base_state):
    base_state["issue_type"] = "damaged_product"
    base_state["severity"]   = "HIGH"
    base_state["policy_data"] = {
        "issue_type":  "damaged_product",
        "policy_text": "Full refund for damaged items within 30 days."
    }
    return base_state


# ==========================================
# TESTS — classify_issue node
# ==========================================
class TestClassifyIssue:

    def test_classifies_damaged_product(self, base_state):
        from agents.support_agent import classify_issue

        mock_response         = MagicMock()
        mock_response.content = '{"issue_type": "damaged_product", "confidence": "high"}'
        mock_response.usage_metadata = {
            "input_tokens": 100, "output_tokens": 20
        }

        with patch("agents.support_agent.ChatGroq") as mock_groq, \
             patch("agents.support_agent.get_prompt",
                   return_value=("classify issue", 1)), \
             patch("agents.support_agent.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_response
            result = classify_issue(base_state)

        assert result["issue_type"] is not None

    def test_handles_malformed_response(self, base_state):
        from agents.support_agent import classify_issue

        mock_response         = MagicMock()
        mock_response.content = "not valid json"
        mock_response.usage_metadata = {
            "input_tokens": 100, "output_tokens": 5
        }

        with patch("agents.support_agent.ChatGroq") as mock_groq, \
             patch("agents.support_agent.get_prompt",
                   return_value=("classify issue", 1)), \
             patch("agents.support_agent.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_response
            result = classify_issue(base_state)

        assert "issue_type" in result


# ==========================================
# TESTS — assess_severity node
# ==========================================
class TestAssessSeverity:

    def test_damaged_product_is_high(self, base_state):
        from agents.support_agent import assess_severity
        base_state["issue_type"] = "damaged_product"
        result = assess_severity(base_state)
        assert result["severity"] in ["HIGH", "MEDIUM", "LOW", "URGENT"]

    def test_sets_severity_field(self, base_state):
        from agents.support_agent import assess_severity
        base_state["issue_type"] = "refund"
        result = assess_severity(base_state)
        assert result["severity"] is not None


# ==========================================
# TESTS — route_by_severity edge
# ==========================================
class TestRouteBySeverity:

    def test_high_severity_routes_to_escalation(self, base_state):
        from agents.support_agent import route_severity
        base_state["severity"] = "HIGH"
        result = route_severity(base_state)
        assert result == "escalation_handler_node"

    def test_low_severity_routes_to_resolution(self, base_state):
        from agents.support_agent import route_severity
        base_state["severity"] = "LOW"
        result = route_severity(base_state)
        assert result == "draft_resolution"

    def test_medium_severity_routes_to_escalation(self, base_state):
        from agents.support_agent import route_severity
        base_state["severity"] = "MEDIUM"
        result = route_severity(base_state)
        assert result == "escalation_handler_node"


class TestFormatResponse:

    def test_format_sets_response(self, state_with_issue):
        from agents.support_agent import format_response
        state_with_issue["resolution"] = "We will process your refund."
        result = format_response(state_with_issue)
        assert result["response"] is not None
        assert len(result["response"]) > 0

    def test_format_includes_ticket_id(self, state_with_issue):
        from agents.support_agent import format_response
        state_with_issue["resolution"] = "We will process your refund."
        state_with_issue["ticket_id"]  = "TKT-001"
        result = format_response(state_with_issue)
        assert "TKT-001" in result["response"]