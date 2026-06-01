# monitoring/metrics.py

from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter("chat_requests_total", "Total number of chat requests", ["agent_used", "status"])

REQUEST_LATENCY = Histogram(
    "chat_request_latency_seconds",
    "Request latency in seconds",
    ["agent_used"],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0],
)

TOKEN_COUNTER = Counter("llm_tokens_total", "Total LLM tokens consumed", ["agent_used", "token_type"])

ERROR_COUNTER = Counter("chat_errors_total", "Total number of errors", ["error_type", "agent_used"])


def record_request_metrics(
    agent_used: str, status: str, latency_seconds: float, input_tokens: int = 0, output_tokens: int = 0
):
    REQUEST_COUNTER.labels(agent_used=agent_used, status=status).inc()

    REQUEST_LATENCY.labels(agent_used=agent_used).observe(latency_seconds)

    if input_tokens > 0:
        TOKEN_COUNTER.labels(agent_used=agent_used, token_type="input").inc(input_tokens)

    if output_tokens > 0:
        TOKEN_COUNTER.labels(agent_used=agent_used, token_type="output").inc(output_tokens)


def record_error(error_type: str, agent_used: str = "unknown"):
    ERROR_COUNTER.labels(error_type=error_type, agent_used=agent_used).inc()
