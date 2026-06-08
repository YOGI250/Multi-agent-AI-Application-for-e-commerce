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


# ==========================================
# TESTS — route_after_classify edge
# ==========================================
class TestRouteAfterClassify:

    def test_routes_to_request_order_id_when_issue_needs_order(self, base_state):
        from agents.support_agent import route_after_classify
        base_state["issue_type"] = "damaged_product"
        base_state["order_id"]   = None
        result = route_after_classify(base_state)
        assert result == "request_order_id"

    def test_routes_to_assess_severity_when_order_id_present(self, base_state):
        from agents.support_agent import route_after_classify
        base_state["issue_type"] = "damaged_product"
        base_state["order_id"]   = "ORD-1001"
        result = route_after_classify(base_state)
        assert result == "assess_severity"

    def test_routes_to_assess_severity_for_general_query(self, base_state):
        from agents.support_agent import route_after_classify
        base_state["issue_type"] = "general_query"
        base_state["order_id"]   = None
        result = route_after_classify(base_state)
        assert result == "assess_severity"


# ==========================================
# TESTS — request_order_id node
# ==========================================
class TestRequestOrderId:

    def test_sets_response_asking_for_order_id(self, base_state):
        from agents.support_agent import request_order_id
        result = request_order_id(base_state)
        assert result["response"] is not None
        assert "order" in result["response"].lower()

    def test_response_contains_order_id_format_hint(self, base_state):
        from agents.support_agent import request_order_id
        result = request_order_id(base_state)
        assert "ORD" in result["response"]


# ==========================================
# TESTS — lookup_policy_node node
# ==========================================
class TestLookupPolicyNode:

    def test_sets_tool_call_message(self, base_state):
        from agents.support_agent import lookup_policy_node
        base_state["issue_type"] = "damaged_product"
        result = lookup_policy_node(base_state)
        assert result["messages"] is not None
        tc = result["messages"][0].tool_calls[0]
        assert tc["name"] == "lookup_policy_tool"
        assert tc["args"]["issue_type"] == "damaged_product"

    def test_sets_tool_call_for_general_query(self, base_state):
        from agents.support_agent import lookup_policy_node
        base_state["issue_type"] = "general_query"
        result = lookup_policy_node(base_state)
        tc = result["messages"][0].tool_calls[0]
        assert tc["name"] == "lookup_policy_tool"


# ==========================================
# TESTS — process_policy_result node
# ==========================================
class TestProcessPolicyResult:

    def test_extracts_policy_text_from_tool_message(self, base_state):
        import json
        from langchain_core.messages import ToolMessage
        from agents.support_agent import process_policy_result
        policy = {"policy_text": "Full refund within 30 days for damaged items."}
        base_state["messages"] = [ToolMessage(content=json.dumps(policy), tool_call_id="policy_fetch_1")]
        result = process_policy_result(base_state)
        assert result["policy_text"] == "Full refund within 30 days for damaged items."

    def test_sets_default_policy_when_no_tool_message(self, base_state):
        from agents.support_agent import process_policy_result
        base_state["messages"] = []
        result = process_policy_result(base_state)
        assert result["policy_text"] is not None
        assert len(result["policy_text"]) > 0

    def test_handles_invalid_json_in_tool_message(self, base_state):
        from langchain_core.messages import ToolMessage
        from agents.support_agent import process_policy_result
        base_state["messages"] = [ToolMessage(content="not json", tool_call_id="policy_fetch_1")]
        result = process_policy_result(base_state)
        assert result["policy_text"] is not None


# ==========================================
# TESTS — escalation_handler_node node
# ==========================================
class TestEscalationHandlerNode:

    def test_sets_ticket_fields_from_subgraph(self, state_with_issue):
        from agents.support_agent import escalation_handler_node
        mock_result = {
            "ticket_id":      "TKT-001",
            "ticket_created": True,
            "is_duplicate":   False,
            "days_open":      0,
        }
        with patch("agents.support_agent.escalation_handler_subgraph") as mock_sg:
            mock_sg.invoke.return_value = mock_result
            result = escalation_handler_node(state_with_issue)
        assert result["ticket_id"] == "TKT-001"
        assert result["ticket_created"] is True
        assert result["is_duplicate"] is False

    def test_sets_duplicate_flag_when_ticket_already_exists(self, state_with_issue):
        from agents.support_agent import escalation_handler_node
        mock_result = {
            "ticket_id":      "TKT-EXISTING",
            "ticket_created": False,
            "is_duplicate":   True,
            "days_open":      2,
        }
        with patch("agents.support_agent.escalation_handler_subgraph") as mock_sg:
            mock_sg.invoke.return_value = mock_result
            result = escalation_handler_node(state_with_issue)
        assert result["is_duplicate"] is True
        assert result["days_open"] == 2


# ==========================================
# TESTS — draft_resolution node
# ==========================================
class TestDraftResolution:

    def test_sets_resolution_from_llm(self, state_with_issue):
        from agents.support_agent import draft_resolution
        state_with_issue["policy_text"]   = "Full refund within 30 days."
        state_with_issue["issue_type"]    = "damaged_product"
        state_with_issue["issue_details"] = "Screen cracked on arrival"
        state_with_issue["order_id"]      = "ORD-1001"
        state_with_issue["order_status"]  = "delivered"
        state_with_issue["ticket_id"]     = None
        state_with_issue["ticket_created"] = False
        state_with_issue["is_duplicate"]  = False
        state_with_issue["days_open"]     = 0
        mock_resp = MagicMock()
        mock_resp.content = "We sincerely apologise. A full refund has been initiated."
        mock_resp.usage_metadata = {"input_tokens": 300, "output_tokens": 50}
        with patch("agents.support_agent.ChatGroq") as mock_groq, \
             patch("agents.support_agent.get_prompt", return_value=("prompt", 1)):
            mock_groq.return_value.invoke.return_value = mock_resp
            result = draft_resolution(state_with_issue)
        assert result["resolution"] is not None
        assert len(result["resolution"]) > 0

    def test_handles_llm_failure(self, state_with_issue):
        from agents.support_agent import draft_resolution
        state_with_issue["policy_text"]    = "Full refund within 30 days."
        state_with_issue["issue_type"]     = "damaged_product"
        state_with_issue["issue_details"]  = "Screen cracked"
        state_with_issue["order_id"]       = None
        state_with_issue["order_status"]   = None
        state_with_issue["ticket_id"]      = None
        state_with_issue["ticket_created"] = False
        state_with_issue["is_duplicate"]   = False
        state_with_issue["days_open"]      = 0
        with patch("agents.support_agent.ChatGroq") as mock_groq, \
             patch("agents.support_agent.get_prompt", return_value=("prompt", 1)):
            mock_groq.return_value.invoke.side_effect = Exception("LLM error")
            result = draft_resolution(state_with_issue)
        assert result["response"] is not None