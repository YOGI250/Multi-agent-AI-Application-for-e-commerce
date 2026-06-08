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
