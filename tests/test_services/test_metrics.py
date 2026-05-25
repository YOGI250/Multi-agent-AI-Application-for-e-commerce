# tests/test_services/test_metrics.py

import pytest
from prometheus_client import REGISTRY


class TestRecordRequestMetrics:

    def test_record_success_request(self):
        from monitoring.metrics import record_request_metrics
        # should not raise
        record_request_metrics(
            agent_used      = "product_agent",
            status          = "success",
            latency_seconds = 1.5,
            input_tokens    = 100,
            output_tokens   = 50
        )

    def test_record_without_tokens(self):
        from monitoring.metrics import record_request_metrics
        record_request_metrics(
            agent_used      = "order_agent",
            status          = "success",
            latency_seconds = 2.0
        )

    def test_record_error_status(self):
        from monitoring.metrics import record_request_metrics
        record_request_metrics(
            agent_used      = "support_agent",
            status          = "error",
            latency_seconds = 0.5
        )


class TestRecordError:

    def test_record_error(self):
        from monitoring.metrics import record_error
        record_error(
            error_type = "ValueError",
            agent_used = "order_agent"
        )

    def test_record_error_unknown_agent(self):
        from monitoring.metrics import record_error
        record_error(error_type="Exception")