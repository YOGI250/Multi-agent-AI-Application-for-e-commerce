# tests/test_agents/test_intent_router.py

import pytest
import json
from unittest.mock import patch, MagicMock


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture
def base_state():
    return {
        "message":                 "show me laptops",
        "user_id":                 "test_user_123",
        "session_id":              "test_session_123",
        "history":                 [],
        "is_authenticated":        True,
        "langfuse_trace_id":       None,
        "langfuse_parent_span_id": None,
        "intent":                  None,
        "confidence":              None,
        "access_granted":          None
    }


# ==========================================
# TESTS — intent_router node
# ==========================================
class TestIntentRouter:

    def test_classifies_product_query(self, base_state):
        from agents.intent_router import intent_router

        mock_response         = MagicMock()
        mock_response.content = json.dumps({
            "intent":     "product_query",
            "confidence": "high",
            "reason":     "asking about products"
        })
        mock_response.usage_metadata = {
            "input_tokens": 100, "output_tokens": 20
        }

        with patch("agents.intent_router.ChatGroq") as mock_groq, \
             patch("agents.intent_router.get_prompt",
                   return_value=("classify intent", 1)), \
             patch("agents.intent_router.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_response
            base_state["message"] = "show me laptops under 50000"
            result = intent_router(base_state)

        assert result["intent"] == "product_query"

    def test_classifies_order_query(self, base_state):
        from agents.intent_router import intent_router

        mock_response         = MagicMock()
        mock_response.content = json.dumps({
            "intent":     "order_query",
            "confidence": "high",
            "reason":     "asking about order"
        })
        mock_response.usage_metadata = {
            "input_tokens": 100, "output_tokens": 20
        }

        with patch("agents.intent_router.ChatGroq") as mock_groq, \
             patch("agents.intent_router.get_prompt",
                   return_value=("classify intent", 1)), \
             patch("agents.intent_router.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_response
            base_state["message"] = "where is my order ORD-1001"
            result = intent_router(base_state)

        assert result["intent"] == "order_query"

    def test_classifies_support_query(self, base_state):
        from agents.intent_router import intent_router

        mock_response         = MagicMock()
        mock_response.content = json.dumps({
            "intent":     "support_query",
            "confidence": "high",
            "reason":     "complaint about product"
        })
        mock_response.usage_metadata = {
            "input_tokens": 100, "output_tokens": 20
        }

        with patch("agents.intent_router.ChatGroq") as mock_groq, \
             patch("agents.intent_router.get_prompt",
                   return_value=("classify intent", 1)), \
             patch("agents.intent_router.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_response
            base_state["message"] = "my product arrived damaged"
            result = intent_router(base_state)

        assert result["intent"] == "support_query"

    def test_classifies_unknown_intent(self, base_state):
        from agents.intent_router import intent_router

        mock_response         = MagicMock()
        mock_response.content = json.dumps({
            "intent":     "unknown",
            "confidence": "low",
            "reason":     "cannot classify"
        })
        mock_response.usage_metadata = {
            "input_tokens": 100, "output_tokens": 20
        }

        with patch("agents.intent_router.ChatGroq") as mock_groq, \
             patch("agents.intent_router.get_prompt",
                   return_value=("classify intent", 1)), \
             patch("agents.intent_router.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_response
            base_state["message"] = "hello"
            result = intent_router(base_state)

        assert result["intent"] == "unknown"

    def test_handles_malformed_llm_response(self, base_state):
        from agents.intent_router import intent_router

        mock_response         = MagicMock()
        mock_response.content = "not valid json"
        mock_response.usage_metadata = {
            "input_tokens": 100, "output_tokens": 5
        }

        with patch("agents.intent_router.ChatGroq") as mock_groq, \
             patch("agents.intent_router.get_prompt",
                   return_value=("classify intent", 1)), \
             patch("agents.intent_router.create_generation"):
            mock_groq.return_value.invoke.return_value = mock_response
            result = intent_router(base_state)

        assert result["intent"] == "unknown"


# ==========================================
# TESTS — route_to_agent edge function
# ==========================================
class TestRouteToAgent:

    def test_routes_order_query_authenticated(self, base_state):
        from agents.intent_router import route_to_agent
        base_state["intent"]           = "order_query"
        base_state["is_authenticated"] = True
        result = route_to_agent(base_state)
        assert result == "run_order_agent"

    def test_routes_product_query(self, base_state):
        from agents.intent_router import route_to_agent
        base_state["intent"]           = "product_query"
        base_state["is_authenticated"] = False
        result = route_to_agent(base_state)
        assert result == "run_product_agent"

    def test_routes_support_query_authenticated(self, base_state):
        from agents.intent_router import route_to_agent
        base_state["intent"]           = "support_query"
        base_state["is_authenticated"] = True
        result = route_to_agent(base_state)
        assert result == "run_support_agent"

    def test_denies_order_query_guest(self, base_state):
        from agents.intent_router import route_to_agent
        base_state["intent"]           = "order_query"
        base_state["is_authenticated"] = False
        result = route_to_agent(base_state)
        assert result == "access_denied"

    def test_denies_support_query_guest(self, base_state):
        from agents.intent_router import route_to_agent
        base_state["intent"]           = "support_query"
        base_state["is_authenticated"] = False
        result = route_to_agent(base_state)
        assert result == "access_denied"

    def test_routes_unknown_to_fallback(self, base_state):
        from agents.intent_router import route_to_agent
        base_state["intent"]           = "unknown"
        base_state["is_authenticated"] = True
        result = route_to_agent(base_state)
        assert result == "fallback_response"


# ==========================================
# TESTS — access_denied node
# ==========================================
class TestAccessDenied:

    def test_access_denied_sets_response(self, base_state):
        from agents.intent_router import access_denied
        result = access_denied(base_state)
        assert result["response"] is not None
        assert len(result["response"]) > 0

    def test_access_denied_sets_agent_used(self, base_state):
        from agents.intent_router import access_denied
        result = access_denied(base_state)
        assert result.get("agent_used") == "access_control"


# ==========================================
# TESTS — fallback_response node
# ==========================================
class TestFallbackResponse:

    def test_fallback_sets_response(self, base_state):
        from agents.intent_router import fallback_response
        result = fallback_response(base_state)
        assert result["response"] is not None
        assert len(result["response"]) > 0

    def test_fallback_sets_agent_used(self, base_state):
        from agents.intent_router import fallback_response
        result = fallback_response(base_state)
        assert result.get("agent_used") == "fallback"