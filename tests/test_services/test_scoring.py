# tests/test_services/test_scoring.py

import pytest
from unittest.mock import patch, MagicMock


class TestScoreResponse:

    def test_score_product_agent(self):
        with patch("langfuse_helpers.scoring.langfuse_client") as mock_lf:
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "product_agent",
                message    = "show me laptops",
                response   = "Here are top 3 laptops for you."
            )
        assert mock_lf.create_score.called

    def test_score_order_agent(self):
        with patch("langfuse_helpers.scoring.langfuse_client") as mock_lf:
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "order_agent",
                message    = "where is my order ORD-1001",
                response   = "Your order ORD-1001 is shipped."
            )
        assert mock_lf.create_score.called

    def test_score_support_agent(self):
        with patch("langfuse_helpers.scoring.langfuse_client") as mock_lf:
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "support_agent",
                message    = "my product arrived damaged",
                response   = "We apologize for the damaged product."
            )
        assert mock_lf.create_score.called

    def test_score_access_control(self):
        with patch("langfuse_helpers.scoring.langfuse_client") as mock_lf:
            from langfuse_helpers.scoring import score_response
            score_response(
                trace_id   = "trace_123",
                agent_used = "access_control",
                message    = "where is my order",
                response   = "Please log in to track orders."
            )
        assert mock_lf.create_score.called

    def test_score_handles_exception(self):
        with patch("langfuse_helpers.scoring.langfuse_client") as mock_lf:
            mock_lf.create_score.side_effect = Exception("LangFuse error")
            from langfuse_helpers.scoring import score_response
            # should not raise
            score_response(
                trace_id   = "trace_123",
                agent_used = "product_agent",
                message    = "show me laptops",
                response   = "Here are some laptops."
            )