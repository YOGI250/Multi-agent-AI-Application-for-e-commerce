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


# ==========================================
# TESTS — prepare_order_fetch node
# ==========================================
class TestPrepareOrderFetch:

    def test_sets_fetch_order_tool_call_when_order_id(self, base_state):
        from agents.order_agent import prepare_order_fetch
        base_state["order_id"]      = "ORD-1001"
        base_state["show_all_orders"] = False
        result = prepare_order_fetch(base_state)
        assert result["messages"] is not None
        assert len(result["messages"]) == 1
        tc = result["messages"][0].tool_calls[0]
        assert tc["name"] == "fetch_order_data"
        assert tc["args"]["order_id"] == "ORD-1001"

    def test_sets_fetch_all_orders_tool_call_when_show_all(self, base_state):
        from agents.order_agent import prepare_order_fetch
        base_state["show_all_orders"] = True
        base_state["user_id"]         = "test_user_123"
        result = prepare_order_fetch(base_state)
        tc = result["messages"][0].tool_calls[0]
        assert tc["name"] == "fetch_all_orders_for_user"
        assert tc["args"]["user_id"] == "test_user_123"

    def test_sets_fetch_all_orders_when_no_order_id(self, base_state):
        from agents.order_agent import prepare_order_fetch
        base_state["order_id"]        = None
        base_state["show_all_orders"] = False
        result = prepare_order_fetch(base_state)
        tc = result["messages"][0].tool_calls[0]
        assert tc["name"] == "fetch_all_orders_for_user"


# ==========================================
# TESTS — process_order_result node
# ==========================================
class TestProcessOrderResult:

    def test_sets_order_data_from_tool_message(self, base_state):
        import json
        from langchain_core.messages import ToolMessage
        from agents.order_agent import process_order_result
        order = {"order_id": "ORD-1001", "user_id": "test_user_123", "status": "shipped"}
        base_state["messages"] = [ToolMessage(content=json.dumps(order), tool_call_id="order_fetch_1")]
        result = process_order_result(base_state)
        assert result["order_data"] == order
        assert result["order_id_found"] is True

    def test_rejects_order_belonging_to_different_user(self, base_state):
        import json
        from langchain_core.messages import ToolMessage
        from agents.order_agent import process_order_result
        order = {"order_id": "ORD-1001", "user_id": "other_user", "status": "shipped"}
        base_state["messages"] = [ToolMessage(content=json.dumps(order), tool_call_id="order_fetch_1")]
        result = process_order_result(base_state)
        assert result["order_data"] is None
        assert result["order_id_found"] is False

    def test_handles_empty_tool_result(self, base_state):
        from agents.order_agent import process_order_result
        base_state["messages"] = []
        result = process_order_result(base_state)
        assert result["order_id_found"] is False

    def test_handles_list_result_sets_all_orders(self, base_state):
        import json
        from langchain_core.messages import ToolMessage
        from agents.order_agent import process_order_result
        orders = [
            {"order_id": "ORD-1001", "user_id": "test_user_123", "status": "shipped"},
            {"order_id": "ORD-1002", "user_id": "test_user_123", "status": "delivered"},
        ]
        base_state["messages"] = [ToolMessage(content=json.dumps(orders), tool_call_id="order_fetch_1")]
        result = process_order_result(base_state)
        assert result["all_orders"] == orders
        assert result["order_id_found"] is True


# ==========================================
# TESTS — route_order_data_found edge
# ==========================================
class TestRouteOrderDataFound:

    def test_routes_to_analyze_when_order_found(self, base_state):
        from agents.order_agent import route_order_data_found
        base_state["order_data"] = {"order_id": "ORD-1001", "status": "shipped"}
        result = route_order_data_found(base_state)
        assert result == "analyze_order_status"

    def test_routes_to_error_when_no_order(self, base_state):
        from agents.order_agent import route_order_data_found
        base_state["order_data"] = None
        result = route_order_data_found(base_state)
        assert result == "error_response"


# ==========================================
# TESTS — analyze_order_status node
# ==========================================
class TestAnalyzeOrderStatus:

    def test_parses_llm_json_response(self, state_with_order):
        from agents.order_agent import analyze_order_status
        mock_resp = MagicMock()
        mock_resp.content = '{"issue_type": "on_track", "summary": "Order is on its way."}'
        mock_resp.usage_metadata = {"input_tokens": 100, "output_tokens": 20}
        with patch("agents.order_agent.ChatGroq") as mock_groq, \
             patch("agents.order_agent.get_prompt", return_value=("prompt", 1)):
            mock_groq.return_value.invoke.return_value = mock_resp
            result = analyze_order_status(state_with_order)
        assert result["order_analysis"]["issue_type"] == "on_track"

    def test_handles_malformed_json(self, state_with_order):
        from agents.order_agent import analyze_order_status
        mock_resp = MagicMock()
        mock_resp.content = "The order is on track."
        mock_resp.usage_metadata = {"input_tokens": 100, "output_tokens": 10}
        with patch("agents.order_agent.ChatGroq") as mock_groq, \
             patch("agents.order_agent.get_prompt", return_value=("prompt", 1)):
            mock_groq.return_value.invoke.return_value = mock_resp
            result = analyze_order_status(state_with_order)
        assert "order_analysis" in result
        assert result["order_analysis"]["issue_type"] == "unknown"

    def test_handles_llm_failure(self, state_with_order):
        from agents.order_agent import analyze_order_status
        with patch("agents.order_agent.ChatGroq") as mock_groq, \
             patch("agents.order_agent.get_prompt", return_value=("prompt", 1)):
            mock_groq.return_value.invoke.side_effect = Exception("LLM error")
            result = analyze_order_status(state_with_order)
        assert result["response"] is not None


# ==========================================
# TESTS — shipment_tracking_node node
# ==========================================
class TestShipmentTrackingNode:

    def test_skips_tracking_when_no_tracking_number(self, state_with_order):
        from agents.order_agent import shipment_tracking_node
        state_with_order["order_data"]["tracking_number"] = None
        result = shipment_tracking_node(state_with_order)
        assert result["tracking_info"] == {}

    def test_invokes_subgraph_when_tracking_number_present(self, state_with_order):
        from agents.order_agent import shipment_tracking_node
        mock_subgraph_result = {
            "carrier_name": "BlueDart",
            "eta":          "2026-05-28",
            "current_location": "Mumbai",
            "tracking_events": [],
        }
        with patch("agents.order_agent.shipment_tracking_subgraph") as mock_sg:
            mock_sg.invoke.return_value = mock_subgraph_result
            result = shipment_tracking_node(state_with_order)
        assert result["tracking_info"]["carrier_name"] == "BlueDart"
        assert result["tracking_info"]["eta"] == "2026-05-28"