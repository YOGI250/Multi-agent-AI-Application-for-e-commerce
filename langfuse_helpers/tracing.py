# langfuse_helpers/tracing.py
#
# LangFuse v4 SDK — migrated from v2.56.1 to v4.7.1
#
# Key API changes (v2 → v4):
#   trace()       → start_observation(as_type='span', trace_context=...)
#   span()        → start_observation(as_type='span', trace_context={'trace_id':..., 'parent_span_id':...})
#   generation()  → start_observation(as_type='generation', usage_details=..., cost_details=...)
#   span.end(output=...) → span.update(output=...); span.end()
#   score()       → create_score()
#   fetch_trace() → removed; returns zero-dict gracefully
#   trace IDs     → 32 lowercase hex chars (OTEL format), generated via create_trace_id()
#
import logging
from typing import Optional
from langfuse import Langfuse
from langfuse.types import TraceContext
from config.settings import settings

_prompt_cache: dict = {}

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LangFuse client — single instance per process
# ─────────────────────────────────────────────────────────────
langfuse_client = Langfuse(
    secret_key = settings.langfuse_secret_key,
    public_key = settings.langfuse_public_key,
    host       = settings.langfuse_host
)

# ─────────────────────────────────────────────────────────────
# Groq pricing (llama-3.3-70b-versatile)
# ─────────────────────────────────────────────────────────────
GROQ_INPUT_COST_PER_MILLION  = settings.groq_input_cost_per_million
GROQ_OUTPUT_COST_PER_MILLION = settings.groq_output_cost_per_million


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    cost = (
        input_tokens  / 1_000_000 * GROQ_INPUT_COST_PER_MILLION +
        output_tokens / 1_000_000 * GROQ_OUTPUT_COST_PER_MILLION
    )
    return round(cost, 8)


def extract_token_usage(response) -> dict:
    """Extracts token counts from a ChatGroq response object."""
    try:
        usage = response.usage_metadata or {}
        input_tokens  = usage.get("input_tokens",  0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens  = usage.get("total_tokens",  0)
        return {"input": input_tokens, "output": output_tokens, "total": total_tokens}
    except Exception as e:
        logger.warning(f"Could not extract token usage: {e}")
        return {"input": 0, "output": 0, "total": 0}


# ─────────────────────────────────────────────────────────────
# TraceHandle — thin wrapper keeping `trace.id` working in routes.py
#
# In v4 the root observation is a LangfuseSpan, not a trace object.
# span.trace_id  = the 32-hex trace ID (what routes.py needs as `trace.id`)
# span.id        = the observation/span ID (different from trace ID)
#
# This wrapper exposes `.id` as the trace_id so no change is needed
# in api/routes.py which calls `trace.id` and `trace.update(output=...)`.
# ─────────────────────────────────────────────────────────────
class TraceHandle:
    def __init__(self, span, trace_id: str):
        self.id    = trace_id   # the 32-hex trace ID
        self._span = span

    def update(self, **kwargs):
        self._span.update(**kwargs)

    def end(self):
        self._span.end()


# ─────────────────────────────────────────────────────────────
# TRACE
# ─────────────────────────────────────────────────────────────
def create_trace(
    session_id:       str,
    user_id:          str,
    is_authenticated: bool,
    message:          str
) -> TraceHandle:
    """
    Creates a top-level trace for one request.
    In v4, this is done by starting a root observation with a fresh trace_id.
    Session and user context are stored in metadata (LangFuse UI shows these).
    """
    trace_id = langfuse_client.create_trace_id()

    span = langfuse_client.start_observation(
        trace_context = TraceContext(trace_id=trace_id, parent_span_id=None),
        name          = "chat_request",
        as_type       = "span",
        input         = {"message": message},
        metadata      = {
            "session_id":        session_id,
            "user_id":           user_id,
            "is_authenticated":  is_authenticated,
            "message_preview":   message[:100],
        }
    )

    # set trace-level I/O so it shows at the top of the trace in LangFuse UI
    span.set_trace_io(
        input={"message": message, "session_id": session_id, "user_id": user_id}
    )

    logger.info(f"LangFuse trace created: {trace_id}")
    return TraceHandle(span, trace_id)


# ─────────────────────────────────────────────────────────────
# SPAN
# ─────────────────────────────────────────────────────────────
def create_span(
    trace_id:             str,
    name:                 str,
    parent_observation_id: Optional[str] = None,
    input_data:           Optional[dict] = None
):
    """
    Creates a span inside a trace, optionally nested under a parent span.
    parent_observation_id → parent_span_id in v4 TraceContext.
    """
    span = langfuse_client.start_observation(
        trace_context = TraceContext(
            trace_id      = trace_id,
            parent_span_id = parent_observation_id
        ),
        name    = name,
        as_type = "span",
        input   = input_data or {}
    )
    logger.info(f"LangFuse span created: {name}")
    return span


def end_span(span, output_data: Optional[dict] = None):
    """
    Ends a span and records its output.
    In v4, end() no longer accepts an output parameter —
    output must be set via update() before calling end().
    """
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
    trace_id:              str,
    name:                  str,
    model:                 str,
    prompt:                str,
    response:              str,
    usage:                 Optional[dict] = None,
    parent_observation_id: Optional[str]  = None,
    prompt_name:           Optional[str]  = None,
    prompt_version:        Optional[int]  = None
):
    """
    Creates and immediately ends a generation span for one LLM call.

    In v4:
      - usage_details replaces usage dict (no "unit" key)
      - cost_details replaces server-side cost calculation
      - end() is called immediately after creation since all data is known
    """
    input_tokens  = (usage or {}).get("input",  0)
    output_tokens = (usage or {}).get("output", 0)
    cost          = calculate_cost(input_tokens, output_tokens)

    metadata = {}
    if prompt_name:
        metadata["prompt_name"]    = prompt_name
        metadata["prompt_version"] = prompt_version

    gen = langfuse_client.start_observation(
        trace_context = TraceContext(
            trace_id       = trace_id,
            parent_span_id = parent_observation_id
        ),
        name          = name,
        as_type       = "generation",
        model         = model,
        input         = prompt,
        output        = response,
        usage_details = {
            "input":  input_tokens,
            "output": output_tokens,
            "total":  input_tokens + output_tokens,
        },
        cost_details  = {"total": cost},
        metadata      = metadata if metadata else None
    )
    gen.end()

    logger.info(
        f"LangFuse generation: {name} | "
        f"tokens={input_tokens}+{output_tokens} | cost=${cost}"
    )
    return gen


# ─────────────────────────────────────────────────────────────
# PROMPT MANAGEMENT
# ─────────────────────────────────────────────────────────────
def get_prompt(
    name:     str,
    version:  int  = None,
    label:    str  = None,
    fallback: str  = None
) -> tuple:
    """
    Fetches a prompt from LangFuse with in-memory caching.
    TextPromptClient.prompt attribute still exists in v4.
    Falls back to hardcoded fallback if LangFuse is unavailable.
    """
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
            prompt_obj     = langfuse_client.get_prompt(name, label=label)
            prompt_version = label
        elif version:
            prompt_obj     = langfuse_client.get_prompt(name, version=version)
            prompt_version = version
        else:
            prompt_obj     = langfuse_client.get_prompt(name)
            prompt_version = "latest"

        prompt_text = prompt_obj.prompt   # .prompt still exists on TextPromptClient in v4
        result      = (prompt_text, prompt_version)
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


def get_trace_token_usage(trace_id: str) -> dict:
    """
    In v4, fetch_trace() has been removed from the SDK.
    Token usage is still tracked via Prometheus metrics in monitoring/metrics.py.
    Returns zero-dict so callers don't need to change.
    """
    return {"input": 0, "output": 0, "total": 0}


# ─────────────────────────────────────────────────────────────
# FLUSH
# ─────────────────────────────────────────────────────────────
def flush():
    """Flushes all pending LangFuse OTEL spans. Called at end of each request."""
    langfuse_client.flush()
