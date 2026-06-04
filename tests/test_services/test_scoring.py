# tests/test_services/test_scoring.py

import pytest
from unittest.mock import patch, MagicMock


_DUMMY_SCORES = {
    "answer_relevancy": 0.8,
    "faithfulness": 0.9,
    "completeness": 0.7,
    "task_completion": 0.8,
    "hallucination": 0.1,
}


class TestScoreResponse:

    def test_score_product_agent(self):
        with patch("langfuse_helpers.scoring.langfuse_client") as mock_lf, \
             patch("langfuse_helpers.scoring._llm_judge", return_value=_DUMMY_SCORES):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "product_agent",
                message    = "show me laptops",
                response   = "Here are top 3 laptops for you."
            )
        assert mock_lf.create_score.called

    def test_score_order_agent(self):
        with patch("langfuse_helpers.scoring.langfuse_client") as mock_lf, \
             patch("langfuse_helpers.scoring._llm_judge", return_value=_DUMMY_SCORES):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "order_agent",
                message    = "where is my order ORD-1001",
                response   = "Your order ORD-1001 is shipped."
            )
        assert mock_lf.create_score.called

    def test_score_support_agent(self):
        with patch("langfuse_helpers.scoring.langfuse_client") as mock_lf, \
             patch("langfuse_helpers.scoring._llm_judge", return_value=_DUMMY_SCORES):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "support_agent",
                message    = "my product arrived damaged",
                response   = "We apologize for the damaged product."
            )
        assert mock_lf.create_score.called

    def test_score_access_control(self):
        with patch("langfuse_helpers.scoring.langfuse_client") as mock_lf, \
             patch("langfuse_helpers.scoring._llm_judge", return_value=_DUMMY_SCORES):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "access_control",
                message    = "where is my order",
                response   = "Please log in to track orders."
            )
        assert mock_lf.create_score.called

    def test_score_handles_exception(self):
        with patch("langfuse_helpers.scoring.langfuse_client") as mock_lf, \
             patch("langfuse_helpers.scoring.record_error") as mock_err:
            mock_lf.create_score.side_effect = Exception("LangFuse error")
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "product_agent",
                message    = "show me laptops",
                response   = "Here are some laptops."
            )
        mock_err.assert_called_once()

    def test_score_empty_message_returns_default_relevancy(self):
        with patch("langfuse_helpers.scoring.langfuse_client"):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "product_agent",
                message    = "the a an",  # all stop words → empty content set
                response   = "Here are laptops ₹40,000 ⭐ 4.5"
            )

    def test_score_order_agent_partial_faithfulness(self):
        # only has_status, no order_id → faithfulness = 0.7
        with patch("langfuse_helpers.scoring.langfuse_client"):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "order_agent",
                message    = "where is my package",
                response   = "Your package has been shipped and is in transit."
            )

    def test_score_order_agent_no_data_faithfulness(self):
        # no order_id, no status → faithfulness = 0.4
        with patch("langfuse_helpers.scoring.langfuse_client"):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "order_agent",
                message    = "where is my order",
                response   = "Please provide your order ID."
            )

    def test_score_product_agent_partial_faithfulness(self):
        # has price but no rating → faithfulness = 0.7
        with patch("langfuse_helpers.scoring.langfuse_client"):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "product_agent",
                message    = "show me laptops",
                response   = "Laptop available at ₹45,000."
            )

    def test_score_product_agent_no_data_faithfulness(self):
        # no price, no rating → faithfulness = 0.4
        with patch("langfuse_helpers.scoring.langfuse_client"):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "product_agent",
                message    = "show me laptops",
                response   = "We have many laptops available."
            )

    def test_score_support_agent_partial_faithfulness(self):
        # has ticket id but no policy → faithfulness = 0.7
        with patch("langfuse_helpers.scoring.langfuse_client"):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "support_agent",
                message    = "my order is damaged",
                response   = "Ticket ABCD1234 has been created for your complaint."
            )

    def test_score_unknown_agent(self):
        with patch("langfuse_helpers.scoring.langfuse_client"):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "unknown_agent",
                message    = "help me",
                response   = "How can I assist you today?"
            )

    def test_score_short_response_completeness(self):
        # word_count <= 20 → completeness = 0.3
        with patch("langfuse_helpers.scoring.langfuse_client"):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "product_agent",
                message    = "show me laptops",
                response   = "No laptops found."
            )

    def test_score_medium_response_completeness(self):
        # 20 < word_count <= 40 → completeness = 0.6
        with patch("langfuse_helpers.scoring.langfuse_client"):
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "product_agent",
                message    = "show me laptops",
                response   = " ".join(["word"] * 30)
            )