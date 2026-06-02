# langfuse_helpers/tracing.py
import logging
from typing import Optional
from langfuse import Langfuse
from langfuse.types import TraceContext
from config.settings import settings
from monitoring.metrics import TOKEN_COUNTER

_prompt_cache: dict = {}

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LangFuse client — single instance per process
# ─────────────────────────────────────────────────────────────
langfuse_client = Langfuse(
    secret_key=settings.langfuse_secret_key, public_key=settings.langfuse_public_key, host=settings.langfuse_host
)

# ─────────────────────────────────────────────────────────────
# Groq pricing (llama-3.3-70b-versatile)
# ─────────────────────────────────────────────────────────────
GROQ_INPUT_COST_PER_MILLION = settings.groq_input_cost_per_million
GROQ_OUTPUT_COST_PER_MILLION = settings.groq_output_cost_per_million


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    cost = (
        input_tokens / 1_000_000 * GROQ_INPUT_COST_PER_MILLION
        + output_tokens / 1_000_000 * GROQ_OUTPUT_COST_PER_MILLION
    )
    return round(cost, 8)


def extract_token_usage(response) -> dict:
    """Extracts token counts from a ChatGroq response object."""
    try:
        usage = response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        return {"input": input_tokens, "output": output_tokens, "total": total_tokens}
    except Exception as e:
        logger.warning(f"Could not extract token usage: {e}")
        return {"input": 0, "output": 0, "total": 0}


# ─────────────────────────────────────────────────────────────
# TraceHandle — exposes `.id` as the trace_id and `.update()` on the root span
# ─────────────────────────────────────────────────────────────
class TraceHandle:
    def __init__(self, span, trace_id: str):
        self.id = trace_id  # the 32-hex trace ID
        self._span = span

    def update(self, **kwargs):
        self._span.update(**kwargs)

    def end(self):
        self._span.end()


# ─────────────────────────────────────────────────────────────
# TRACE
# ─────────────────────────────────────────────────────────────
def create_trace(session_id: str, user_id: str, is_authenticated: bool, message: str) -> TraceHandle:
    """Creates a top-level trace for one request."""
    trace_id = langfuse_client.create_trace_id()

    span = langfuse_client.start_observation(
        trace_context=TraceContext(trace_id=trace_id, parent_span_id=None),
        name="chat_request",
        as_type="span",
        input={"message": message},
        metadata={
            "is_authenticated": is_authenticated,
            "message_preview": message[:100],
        },
    )

    # set native session_id and user_id on the trace so they appear
    # in LangFuse UI Sessions tab and User filters
    span.set_trace_io(input={"message": message, "session_id": session_id, "user_id": user_id})
    span.update_trace(session_id=session_id, user_id=user_id)

    logger.info(f"LangFuse trace created: {trace_id}")
    return TraceHandle(span, trace_id)


# ─────────────────────────────────────────────────────────────
# SPAN
# ─────────────────────────────────────────────────────────────
def create_span(
    trace_id: str, name: str, parent_observation_id: Optional[str] = None, input_data: Optional[dict] = None
):
    """Creates a span inside a trace, optionally nested under a parent span."""
    span = langfuse_client.start_observation(
        trace_context=TraceContext(trace_id=trace_id, parent_span_id=parent_observation_id),
        name=name,
        as_type="span",
        input=input_data or {},
    )
    logger.info(f"LangFuse span created: {name}")
    return span


def end_span(span, output_data: Optional[dict] = None):
    """Ends a span and records its output."""
    try:
        if output_data:
            span.update(output=output_data)
        span.end()
        logger.info("LangFuse span ended")
    except Exception as e:
        logger.warning(f"Could not end span: {e}")


# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────
def create_generation(
    trace_id: str,
    name: str,
    model: str,
    prompt: str,
    response: str,
    usage: Optional[dict] = None,
    parent_observation_id: Optional[str] = None,
    prompt_name: Optional[str] = None,
    prompt_version: Optional[int] = None,
    agent_used: str = "unknown",
):
    """Creates and immediately ends a generation span for one LLM call."""
    input_tokens = (usage or {}).get("input", 0)
    output_tokens = (usage or {}).get("output", 0)
    cost = calculate_cost(input_tokens, output_tokens)

    metadata = {}
    if prompt_name:
        metadata["prompt_name"] = prompt_name
        metadata["prompt_version"] = prompt_version

    gen = langfuse_client.start_observation(
        trace_context=TraceContext(trace_id=trace_id, parent_span_id=parent_observation_id),
        name=name,
        as_type="generation",
        model=model,
        input=prompt,
        output=response,
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        },
        cost_details={"total": cost},
        metadata=metadata if metadata else None,
    )
    gen.end()

    if input_tokens > 0:
        TOKEN_COUNTER.labels(agent_used=agent_used, token_type="input").inc(input_tokens)
    if output_tokens > 0:
        TOKEN_COUNTER.labels(agent_used=agent_used, token_type="output").inc(output_tokens)

    logger.info(f"LangFuse generation: {name} | " f"tokens={input_tokens}+{output_tokens} | cost=${cost}")
    return gen


# ─────────────────────────────────────────────────────────────
# PROMPT MANAGEMENT
# ─────────────────────────────────────────────────────────────
def get_prompt(name: str, version: int = None, label: str = None, fallback: str = None) -> tuple:
    """Fetches a prompt from LangFuse with in-memory caching. Falls back to hardcoded fallback if LangFuse is unavailable."""
    if label:
        cache_key = f"{name}::{label}"
    elif version:
        cache_key = f"{name}::v{version}"
    else:
        cache_key = f"{name}::latest"

    if cache_key in _prompt_cache:
        logger.debug(f"Prompt cache hit: {cache_key}")
        return _prompt_cache[cache_key]

    try:
        if label:
            prompt_obj = langfuse_client.get_prompt(name, label=label)
            prompt_version = label
        elif version:
            prompt_obj = langfuse_client.get_prompt(name, version=version)
            prompt_version = version
        else:
            prompt_obj = langfuse_client.get_prompt(name)
            prompt_version = "latest"

        prompt_text = prompt_obj.prompt  # .prompt still exists on TextPromptClient in v4
        result = (prompt_text, prompt_version)
        _prompt_cache[cache_key] = result
        logger.info(f"LangFuse prompt fetched: {name} {cache_key}")
        return result

    except Exception as e:
        logger.warning(f"LangFuse prompt fetch failed for '{name}': {e} — using fallback")
        return (fallback or "", "fallback")


def compile_prompt(template: str, **kwargs) -> str:
    """
    Safe {{variable}} replacement for LangFuse prompt templates.
    Avoids Python .format() which breaks on JSON curly braces.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", str(value or ""))
    return result


# ─────────────────────────────────────────────────────────────
# FLUSH
# ─────────────────────────────────────────────────────────────
def flush():
    """Flushes all pending LangFuse OTEL spans. Called at end of each request."""
    langfuse_client.flush()
