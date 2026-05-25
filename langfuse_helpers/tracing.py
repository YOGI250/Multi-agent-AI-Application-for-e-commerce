# langfuse_helpers/tracing.py

import logging
from typing import Optional
from langfuse import Langfuse
from config.settings import settings

logger = logging.getLogger(__name__)

# ==========================================
# LANGFUSE CLIENT — single instance
# ==========================================
langfuse_client = Langfuse(
    secret_key = settings.langfuse_secret_key,
    public_key = settings.langfuse_public_key,
    host       = settings.langfuse_host
)

# ==========================================
# GROQ PRICING
# llama-3.3-70b-versatile
# ==========================================
import os
GROQ_INPUT_COST_PER_MILLION  = settings.groq_input_cost_per_million
GROQ_OUTPUT_COST_PER_MILLION = settings.groq_output_cost_per_million


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Calculate USD cost for llama-3.3-70b-versatile on Groq.
    Called inside every create_generation call.
    """
    cost = (
        input_tokens  / 1_000_000 * GROQ_INPUT_COST_PER_MILLION +
        output_tokens / 1_000_000 * GROQ_OUTPUT_COST_PER_MILLION
    )
    return round(cost, 8)


def extract_token_usage(response) -> dict:
    """
    Extracts token counts from a ChatGroq response object.
    Returns dict with input, output, total keys.
    Called after every llm.invoke() call.
    """
    try:
        usage = response.usage_metadata or {}
        input_tokens  = usage.get("input_tokens",  0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens  = usage.get("total_tokens",  0)
        return {
            "input":  input_tokens,
            "output": output_tokens,
            "total":  total_tokens
        }
    except Exception as e:
        logger.warning(f"Could not extract token usage: {e}")
        return {"input": 0, "output": 0, "total": 0}


# ==========================================
# TRACE
# ==========================================
def create_trace(
    session_id:       str,
    user_id:          str,
    is_authenticated: bool,
    message:          str
):
    """
    Creates a top-level trace for one request.
    Called once per request in api/routes.py.
    Everything else is nested under this trace.
    """
    trace = langfuse_client.trace(
        name       = "chat_request",
        session_id = session_id,
        user_id    = user_id,
        input      = {"message": message},
        metadata   = {
            "is_authenticated": is_authenticated,
            "message_preview":  message[:100]
        },
        tags = ["authenticated" if is_authenticated else "guest"]
    )
    logger.info(f"LangFuse trace created: {trace.id}")
    return trace


# ==========================================
# SPAN
# Creates a span under a trace or parent span.
# Use for: pure code nodes, tool nodes,
#          subgraph containers.
# ==========================================
def create_span(
    trace_id:            str,
    name:                str,
    parent_observation_id: Optional[str] = None,
    input_data:          Optional[dict]  = None
):
    """
    Creates a span inside a trace.
    parent_observation_id links it to a parent span
    so the hierarchy is correctly nested in LangFuse.
    """
    span = langfuse_client.span(
        trace_id              = trace_id,
        parent_observation_id = parent_observation_id,
        name                  = name,
        input                 = input_data or {}
    )
    logger.info(f"LangFuse span created: {name}")
    return span


def end_span(span, output_data: Optional[dict] = None):
    """
    Ends a span and records its output.
    Must be called after the node completes.
    """
    try:
        span.end(output=output_data or {})
        logger.info("LangFuse span ended")
    except Exception as e:
        logger.warning(f"Could not end span: {e}")


# ==========================================
# GENERATION
# Creates a generation span for every LLM call.
# Records full prompt, full response,
# token usage, and cost.
# ==========================================
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
    Creates a generation span for one LLM call.
    Associates with trace and optional parent span.
    Calculates and records cost automatically.

    usage dict: {"input": N, "output": N, "total": N}
    """
    input_tokens  = (usage or {}).get("input",  0)
    output_tokens = (usage or {}).get("output", 0)
    cost          = calculate_cost(input_tokens, output_tokens)

    metadata = {}
    if prompt_name:
        metadata["prompt_name"]    = prompt_name
        metadata["prompt_version"] = prompt_version

    generation = langfuse_client.generation(
        trace_id              = trace_id,
        parent_observation_id = parent_observation_id,
        name                  = name,
        model                 = model,
        input                 = prompt,
        output                = response,
        usage                 = {
            "input":  input_tokens,
            "output": output_tokens,
            "total":  input_tokens + output_tokens,
            "unit":   "TOKENS"
        },
        metadata              = metadata if metadata else None
    )

    logger.info(
        f"LangFuse generation created: {name} | "
        f"tokens={input_tokens}+{output_tokens} | "
        f"cost=${cost}"
    )
    return generation


# ==========================================
# PROMPT MANAGEMENT
# Pull prompts from LangFuse at runtime.
# Falls back to a default if not found.
# ==========================================
def get_prompt(
    prompt_name: str,
    version:     Optional[int] = None,
    fallback:    str           = ""
) -> tuple[str, Optional[int]]:
    """
    Fetches a prompt template from LangFuse
    prompt management.

    Returns (prompt_text, version_number).
    Falls back to hardcoded prompt if fetch fails.

    Called in LLM nodes to pull prompts at runtime
    instead of hardcoding them in code.
    """
    try:
        if version:
            prompt_obj = langfuse_client.get_prompt(
                prompt_name, version=version
            )
        else:
            prompt_obj = langfuse_client.get_prompt(prompt_name)

        logger.info(
            f"LangFuse prompt fetched: {prompt_name} "
            f"v{prompt_obj.version}"
        )
        return prompt_obj.prompt, prompt_obj.version

    except Exception as e:
        logger.warning(
            f"Could not fetch prompt '{prompt_name}' "
            f"from LangFuse: {e}. Using fallback."
        )
        return fallback, None
    
def compile_prompt(template: str, **kwargs) -> str:
    """
    Safe prompt compilation for LangFuse {{variable}} syntax.
    Replaces {{variable}} with actual values.
    Avoids Python .format() which breaks on JSON curly braces.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", str(value or ""))
    return result

def get_trace_token_usage(trace_id: str) -> dict:
    """
    Fetches actual token usage for a completed trace from LangFuse.
    Used to record accurate token counts in Prometheus.
    """
    try:
        trace = langfuse_client.fetch_trace(trace_id)
        data = trace.data
        return {
            "input":  getattr(data, "prompt_tokens",     None) or getattr(data, "input_tokens",  None) or 0,
            "output": getattr(data, "completion_tokens", None) or getattr(data, "output_tokens", None) or 0,
            "total":  getattr(data, "total_tokens",      None) or 0
        }
        return {"input": 0, "output": 0, "total": 0}
    except Exception as e:
        logger.warning(f"Could not fetch trace token usage: {e}")
        return {"input": 0, "output": 0, "total": 0}


# ==========================================
# FLUSH
# Must be called at end of every request
# to ensure all events are sent to LangFuse.
# ==========================================
def flush():
    """
    Flushes all pending LangFuse events.
    Called at end of each request in api/routes.py.
    """
    langfuse_client.flush()