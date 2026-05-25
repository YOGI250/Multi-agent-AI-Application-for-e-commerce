# tests/test_agents/test_order_agent.py

import pytest
from unittest.mock import patch, MagicMock


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture
def base_state():
    return {
        "message":                 "where is my order ORD-1001?",
        "user_id":                 "test_user_123",
        "session_id":              "test_session_123",
        "history":                 [],
        "is_authenticated":        True,
        "langfuse_trace_id":       None,
        "langfuse_parent_span_id": None,
        "order_id":                None,
        "order_id_found":          False,
        "order_data":              None,
        "order_analysis":          {},
        "tracking_info":           {},
        "all_orders":              None,
        "show_all_orders":         False,
        "response":                None
    }

@pytest.fixture
def state_with_order(base_state):
    base_state["order_id"]       = "ORD-1001"
    base_state["order_id_found"] = True
    base_state["order_data"]     = {
        "order_id":          "ORD-1001",
        "user_id":           "test_user_123",
        "status":            "shipped",
        "carrier":           "BlueDart",
        "tracking_number":   "BD123456789",
        "order_value":       1299.0,
        "expected_delivery": "2026-05-28",
        "order_date":        "2026-05-22",
        "items":             "Laptop Stand"
    }
    return base_state


# ==========================================
# TESTS — validate_input node
# ==========================================
class TestValidateInput:

    def test_extracts_order_id(self, base_state):
        from agents.order_agent import validate_input
        base_state["message"] = "where is my order ORD-1001?"
        result = validate_input(base_state)
        assert result["order_id"]       == "ORD-1001"
        assert result["order_id_found"] == True

    def test_no_order_id_sets_false(self, base_state):
        from agents.order_agent import validate_input
        base_state["message"] = "show me my orders"
        result = validate_input(base_state)
        assert result["order_id"] is None

    def test_show_all_orders_keyword(self, base_state):
        from agents.order_agent import validate_input
        base_state["message"] = "SHOW ME MY ORDERS"
        result = validate_input(base_state)
        assert result["show_all_orders"] == True

    def test_order_id_pattern_case_insensitive(self, base_state):
        from agents.order_agent import validate_input
        base_state["message"] = "where is ord-1001"
        result = validate_input(base_state)
        assert result["order_id"] is not None


# ==========================================
# TESTS — route_order_found edge
# ==========================================
class TestRouteOrderFound:

    def test_routes_to_fetch_when_found(self, base_state):
        from agents.order_agent import route_order_found
        base_state["order_id_found"] = True
        result = route_order_found(base_state)
        assert result == "fetch_order_data_node"

    def test_routes_to_error_when_not_found(self, base_state):
        from agents.order_agent import route_order_found
        base_state["order_id_found"] = False
        result = route_order_found(base_state)
        assert result == "error_response"


# ==========================================
# TESTS — error_response node
# ==========================================
class TestErrorResponse:

    def test_error_response_sets_response(self, base_state):
        from agents.order_agent import error_response
        base_state["order_id_found"] = False
        result = error_response(base_state)
        assert result["response"] is not None
        assert len(result["response"]) > 0

    def test_error_response_sets_message_without_order_id(self, base_state):
        from agents.order_agent import error_response
        base_state["order_id"] = None
        result = error_response(base_state)
        assert "order ID" in result["response"]

    def test_error_response_includes_order_id_in_message(self, base_state):
        from agents.order_agent import error_response
        base_state["order_id"] = "ORD-9999"
        result = error_response(base_state)
        assert "ORD-9999" in result["response"]


# ==========================================
# TESTS — generate_response node
# ==========================================
class TestGenerateResponse:

    def test_generate_response_with_order(self, state_with_order):
        from agents.order_agent import generate_response

        mock_llm_response         = MagicMock()
        mock_llm_response.content = "Your order ORD-1001 is shipped."
        mock_llm_response.usage_metadata = {
            "input_tokens": 200, "output_tokens": 50
        }

        with patch("agents.order_agent.ChatGroq") as mock_groq, \
             patch("agents.order_agent.get_prompt",
                   return_value=("fallback prompt", 1)):
            mock_groq.return_value.invoke.return_value = mock_llm_response
            result = generate_response(state_with_order)

        assert result["response"] is not None
        assert "ORD-1001" in result["response"] or \
               len(result["response"]) > 0

    def test_generate_response_all_orders(self, base_state):
        from agents.order_agent import generate_response

        base_state["all_orders"] = [
            {
                "order_id":          "ORD-1001",
                "items":             "Laptop Stand",
                "status":            "shipped",
                "order_value":       1299.0,
                "expected_delivery": "2026-05-28"
            },
            {
                "order_id":          "ORD-1002",
                "items":             "USB Hub",
                "status":            "delivered",
                "order_value":       499.0,
                "expected_delivery": "2026-05-10"
            }
        ]
        base_state["order_id"]       = None
        base_state["order_id_found"] = True

        mock_llm_response         = MagicMock()
        mock_llm_response.content = "Here are your 2 orders."
        mock_llm_response.usage_metadata = {
            "input_tokens": 300, "output_tokens": 80
        }

        with patch("agents.order_agent.ChatGroq") as mock_groq:
            mock_groq.return_value.invoke.return_value = mock_llm_response
            result = generate_response(base_state)

        assert result["response"] is not None