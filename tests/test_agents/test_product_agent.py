# tests/test_agents/test_product_agent.py

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def base_state():
    return {
        "message":                 "show me laptops under 50000",
        "user_id":                 "test_user_123",
        "session_id":              "test_session_123",
        "history":                 [],
        "is_authenticated":        True,
        "langfuse_trace_id":       None,
        "langfuse_parent_span_id": None,
        "preferences":             {},
        "search_results":          [],
        "enriched_products":       [],
        "response":                None,
        "agent_used":              None
    }


@pytest.fixture
def state_with_results(base_state):
    base_state["search_results"] = [
        {
            "product_id": "P001",
            "name":       "Dell Laptop",
            "category":   "Computers",
            "price":      45000.0,
            "brand":      "Dell",
            "rating":     4.2,
            "in_stock":   True
        },
        {
            "product_id": "P002",
            "name":       "HP Laptop",
            "category":   "Computers",
            "price":      48000.0,
            "brand":      "HP",
            "rating":     4.0,
            "in_stock":   True
        }
    ]
    return base_state


# ==========================================
# TESTS — extract_preferences node
# ==========================================
class TestExtractPreferences:

    def test_extracts_max_price(self, base_state):
        from agents.product_agent import extract_preferences

        mock_response         = MagicMock()
        mock_response.content = '{"category": "Computers", "max_price": 50000, "min_price": 0, "brand": "", "keyword": "laptop"}'
        mock_response.usage_metadata = {
            "input_tokens": 100, "output_tokens": 30
        }

        with patch("agents.product_agent.ChatGroq") as mock_groq, \
             patch("agents.product_agent.get_prompt",
                   return_value=("extract preferences", 1)), \
             patch("agents.product_agent.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_response
            result = extract_preferences(base_state)

        assert "preferences" in result
        assert isinstance(result["preferences"], dict)

    def test_handles_malformed_response(self, base_state):
        from agents.product_agent import extract_preferences

        mock_response         = MagicMock()
        mock_response.content = "not valid json"
        mock_response.usage_metadata = {
            "input_tokens": 100, "output_tokens": 5
        }

        with patch("agents.product_agent.ChatGroq") as mock_groq, \
             patch("agents.product_agent.get_prompt",
                   return_value=("extract preferences", 1)), \
             patch("agents.product_agent.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_response
            result = extract_preferences(base_state)

        assert "preferences" in result


# ==========================================
# TESTS — route_search_results edge
# ==========================================
# fix route function name
class TestRouteSearchResults:

    def test_routes_to_rank_when_results_found(self, state_with_results):
        from agents.product_agent import route_results_found
        result = route_results_found(state_with_results)
        assert result == "rank_and_filter"

    def test_routes_to_broaden_when_no_results(self, base_state):
        from agents.product_agent import route_results_found
        base_state["search_results"]   = []
        base_state["broaden_attempts"] = 0
        result = route_results_found(base_state)
        assert result == "broaden_search"

    def test_routes_to_no_results_after_attempts(self, base_state):
        from agents.product_agent import route_results_found
        base_state["search_results"]   = []
        base_state["broaden_attempts"] = 3
        result = route_results_found(base_state)
        assert result == "no_results_response"


# fix no_results_response — does not set agent_used
class TestNoResultsResponse:

    def test_no_results_sets_response(self, base_state):
        from agents.product_agent import no_results_response
        result = no_results_response(base_state)
        assert result["response"] is not None
        assert len(result["response"]) > 0

    def test_no_results_mentions_products(self, base_state):
        from agents.product_agent import no_results_response
        result = no_results_response(base_state)
        assert "catalog" in result["response"].lower() or "available" in result["response"].lower()


# ==========================================
# TESTS — format_recommendations node
# ==========================================
class TestFormatRecommendations:

    def test_format_sets_response_with_products(self, base_state):
        from agents.product_agent import format_recommendations
        base_state["final_recommendations"] = [
            {
                "name":         "Dell Laptop",
                "price":        45000.0,
                "rating":       4.2,
                "rating_count": 150,
                "brand":        "Dell",
                "features":     ["8GB RAM", "512GB SSD"],
                "score":        0.85
            }
        ]
        result = format_recommendations(base_state)
        assert result["response"] is not None
        assert len(result["response"]) > 0

    def test_format_handles_empty_products(self, base_state):
        from agents.product_agent import format_recommendations
        base_state["final_recommendations"] = []
        result = format_recommendations(base_state)
        assert result["response"] is not None


# ==========================================
# TESTS — search_products_node node
# ==========================================
class TestSearchProductsNode:

    def test_sets_tool_call_message(self, base_state):
        from agents.product_agent import search_products_node
        base_state["filters"] = {"category": "Computers", "max_price": 50000}
        result = search_products_node(base_state)
        assert result["messages"] is not None
        tc = result["messages"][0].tool_calls[0]
        assert tc["name"] == "search_products_tool"
        assert tc["args"]["filters"] == {"category": "Computers", "max_price": 50000}

    def test_sets_tool_call_with_empty_filters(self, base_state):
        from agents.product_agent import search_products_node
        base_state["filters"] = {}
        result = search_products_node(base_state)
        tc = result["messages"][0].tool_calls[0]
        assert tc["name"] == "search_products_tool"


# ==========================================
# TESTS — process_search_result node
# ==========================================
class TestProcessSearchResult:

    def test_sets_search_results_from_tool_message(self, state_with_results):
        import json
        from langchain_core.messages import ToolMessage
        from agents.product_agent import process_search_result
        products = [{"product_id": "P001", "name": "Dell Laptop", "price": 45000}]
        state_with_results["messages"] = [ToolMessage(content=json.dumps(products), tool_call_id="product_search_1")]
        result = process_search_result(state_with_results)
        assert result["search_results"] == products

    def test_sets_empty_list_on_no_tool_message(self, base_state):
        from agents.product_agent import process_search_result
        base_state["messages"] = []
        result = process_search_result(base_state)
        assert result["search_results"] == []

    def test_handles_invalid_json_in_tool_message(self, base_state):
        from langchain_core.messages import ToolMessage
        from agents.product_agent import process_search_result
        base_state["messages"] = [ToolMessage(content="not json", tool_call_id="product_search_1")]
        result = process_search_result(base_state)
        assert result["search_results"] == []


# ==========================================
# TESTS — broaden_search node
# ==========================================
class TestBroadenSearch:

    def test_first_attempt_drops_brand(self, base_state):
        from agents.product_agent import broaden_search
        base_state["filters"]          = {"brand": "Dell", "max_price": 50000}
        base_state["original_filters"] = {"brand": "Dell", "max_price": 50000}
        base_state["broaden_attempts"] = 0
        result = broaden_search(base_state)
        assert result["filters"]["brand"] is None
        assert result["broaden_attempts"] == 1

    def test_second_attempt_drops_product_type(self, base_state):
        from agents.product_agent import broaden_search
        base_state["filters"]          = {"product_type": "gaming_laptop", "category": "Computers"}
        base_state["original_filters"] = {"category": "Computers"}
        base_state["broaden_attempts"] = 1
        result = broaden_search(base_state)
        assert result["filters"]["product_type"] is None

    def test_third_attempt_drops_all_filters(self, base_state):
        from agents.product_agent import broaden_search
        base_state["filters"]          = {"brand": "Dell", "max_price": 50000, "product_type": "laptop"}
        base_state["original_filters"] = {}
        base_state["broaden_attempts"] = 2
        result = broaden_search(base_state)
        assert result["filters"]["brand"] is None
        assert result["filters"]["max_price"] is None
        assert result["filters"]["product_type"] is None

    def test_increments_broaden_attempts(self, base_state):
        from agents.product_agent import broaden_search
        base_state["filters"]          = {}
        base_state["original_filters"] = {}
        base_state["broaden_attempts"] = 0
        result = broaden_search(base_state)
        assert result["broaden_attempts"] == 1


# ==========================================
# TESTS — rank_and_filter node
# ==========================================
class TestRankAndFilter:

    def test_ranks_products_by_llm_indices(self, state_with_results):
        from agents.product_agent import rank_and_filter
        mock_resp = MagicMock()
        mock_resp.content = "[2, 1]"
        mock_resp.usage_metadata = {"input_tokens": 200, "output_tokens": 10}
        with patch("agents.product_agent.ChatGroq") as mock_groq, \
             patch("agents.product_agent.get_prompt", return_value=("prompt", 1)), \
             patch("agents.product_agent.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_resp
            result = rank_and_filter(state_with_results)
        assert result["ranked_products"] is not None
        assert len(result["ranked_products"]) == 2
        assert result["ranked_products"][0]["name"] == "HP Laptop"

    def test_empty_llm_index_array_sets_empty_ranked(self, state_with_results):
        from agents.product_agent import rank_and_filter
        mock_resp = MagicMock()
        mock_resp.content = "[]"
        mock_resp.usage_metadata = {"input_tokens": 200, "output_tokens": 5}
        with patch("agents.product_agent.ChatGroq") as mock_groq, \
             patch("agents.product_agent.get_prompt", return_value=("prompt", 1)), \
             patch("agents.product_agent.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_resp
            result = rank_and_filter(state_with_results)
        assert result["ranked_products"] == []

    def test_handles_llm_failure(self, state_with_results):
        from agents.product_agent import rank_and_filter
        with patch("agents.product_agent.ChatGroq") as mock_groq, \
             patch("agents.product_agent.get_prompt", return_value=("prompt", 1)):
            mock_groq.return_value.invoke.side_effect = Exception("LLM error")
            result = rank_and_filter(state_with_results)
        assert result["response"] is not None
