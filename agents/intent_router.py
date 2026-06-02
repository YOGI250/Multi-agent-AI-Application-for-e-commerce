# agents/intent_router.py

import logging
import re
import json
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from config.settings import settings
from agents.order_agent import order_agent
from agents.product_agent import product_agent
from agents.support_agent import support_agent
from langfuse_helpers.tracing import (
    create_span,
    end_span,
    create_generation,
    get_prompt,
    compile_prompt,
    extract_token_usage,
)
from utils.memory import format_context, format_recent_messages

logger = logging.getLogger(__name__)


# ==========================================
# PENDING SUPPORT CONTEXT DETECTION
# Phrases that the support agent uses when asking for an order ID.
# If any of these appear in recent assistant history, there's an
# unresolved support issue waiting for the user's order ID.
# ==========================================
_SUPPORT_NEEDS_ORDER_PHRASES = [
    "i'll need your order id",
    "typing 'show my orders'",
]

_GREETINGS = {"hi", "hello", "hey", "hiya", "greetings", "howdy", "sup", "yo", "helo", "hai"}
_HELP_PREFIXES = (
    "help",
    "what can you do",
    "what can you help",
    "what do you do",
    "how can you help",
    "what are you",
    "who are you",
    "what is this",
)
_PRODUCT_PREFIXES = (
    "show me",
    "find me",
    "search for",
    "i want",
    "i need",
    "looking for",
    "do you have",
    "give me",
    "can you show",
    "can you find",
    "get me",
)


def _has_pending_support_context(history: list) -> bool:
    for msg in reversed(history[-8:]):
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            if any(phrase in content for phrase in _SUPPORT_NEEDS_ORDER_PHRASES):
                return True
    return False


# ==========================================
# STATE
# ==========================================
class RouterState(TypedDict):
    message: str
    user_id: str
    session_id: str
    history: list
    is_authenticated: bool
    intent: Optional[str]
    confidence: Optional[str]
    reason: Optional[str]
    response: Optional[str]
    agent_used: Optional[str]
    products: Optional[list]
    session_context: Optional[dict]
    langfuse_trace_id: Optional[str]
    langfuse_parent_span_id: Optional[str]
    total_input_tokens: Optional[int]
    total_output_tokens: Optional[int]


# ==========================================
# NODE 1 — intent_router (LLM)
# ==========================================
def intent_router(state: RouterState) -> RouterState:
    logger.info("Intent Router: classifying message", extra={"node_name": "intent_router"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    message = state.get("message", "")
    history = state.get("history", [])

    # ── Pre-check 1: greetings and generic help — always unknown, never context-biased ──
    msg_lower = message.strip().lower()

    if any(msg_lower.startswith(p) for p in _PRODUCT_PREFIXES):
        logger.info(
            "Intent Router: product-search phrase detected — bypassing LLM, routing to product_query",
            extra={"node_name": "intent_router"},
        )
        state["intent"] = "product_query"
        state["confidence"] = "1.0"
        state["reason"] = "Message starts with explicit product-search phrase"
        return state

    if msg_lower in _GREETINGS or any(msg_lower.startswith(p) for p in _HELP_PREFIXES):
        logger.info(
            "Intent Router: greeting/help detected — bypassing LLM, routing to unknown",
            extra={"node_name": "intent_router"},
        )
        state["intent"] = "unknown"
        state["confidence"] = "1.0"
        state["reason"] = "Greeting or generic help request"
        return state

    # ── Pre-check 2: pending support context ──
    # If the support agent previously asked for an order ID and the user
    # is now providing one, skip the LLM and force-route to support.
    if _has_pending_support_context(history):
        order_id_in_message = re.search(r"ORD-[A-Z0-9-]+", message.upper())
        if order_id_in_message:
            logger.info(
                "Intent Router: pending support context detected + order ID provided — "
                "bypassing LLM, routing to support_query",
                extra={"node_name": "intent_router"},
            )
            state["intent"] = "support_query"
            state["confidence"] = "1.0"
            state["reason"] = "User providing order ID in response to pending support request"
            return state

    context_str = format_context(state.get("session_context") or {})
    recent_msgs = format_recent_messages(history, n=2)

    fallback_prompt = f"""You are an intent classifier for an e-commerce customer support system.
Classify the user message into exactly one of these intents.

Intents:
- order_query: questions about orders, delivery, tracking, shipment status
- product_query: searching for products, recommendations, product comparisons
- support_query: complaints, refunds, damaged items, wrong items, cancellations

ALWAYS classify as unknown (never map to another intent based on session context):
- Greetings: "hi", "hello", "hey", "good morning", "how are you"
- Generic help: "what can you help me with?", "what do you do?", "help"
- Unrelated small talk or unclear one-word messages

Session context: {context_str}
Recent messages:
{recent_msgs}

User message: "{message}"

Respond ONLY with a JSON object. No explanation.
{{
  "intent": "order_query or product_query or support_query or unknown",
  "confidence": 0.0 to 1.0,
  "reason": "one sentence explaining why"
}}"""

    prompt_text, prompt_version = get_prompt(
        "intent_router", label=settings.intent_router_prompt_label, fallback=fallback_prompt
    )

    if "{{message}}" in prompt_text:
        prompt_text = compile_prompt(prompt_text, message=message, history=recent_msgs)

    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
    try:
        response = llm.invoke(prompt_text)
    except Exception as e:
        logger.error(f"LLM invoke failed in intent_router: {e}")
        state["intent"] = "unknown"
        state["confidence"] = 0.0
        state["reason"] = "Service temporarily unavailable"
        return state

    usage = extract_token_usage(response)

    if trace_id:
        create_generation(
            trace_id=trace_id,
            name="intent_router",
            model=settings.llm_model_name,
            prompt=prompt_text,
            response=response.content,
            usage=usage,
            parent_observation_id=parent_id,
            prompt_name="intent_router",
            prompt_version=prompt_version,
            agent_used="intent_router",
        )

    try:
        text = response.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        result = (
            json.loads(match.group())
            if match
            else {"intent": "unknown", "confidence": 0.0, "reason": "Could not classify intent"}
        )
    except Exception:
        result = {"intent": "unknown", "confidence": 0.0, "reason": "Could not classify intent"}

    state["intent"] = result.get("intent", "unknown")
    state["confidence"] = result.get("confidence", "low")
    state["reason"] = result.get("reason", "")
    state["total_input_tokens"] = (state.get("total_input_tokens") or 0) + usage.get("input", 0)
    state["total_output_tokens"] = (state.get("total_output_tokens") or 0) + usage.get("output", 0)

    logger.info(f"Intent: {state['intent']} | Confidence: {state['confidence']}", extra={"node_name": "intent_router"})
    return state


# ==========================================
# EDGE — route to correct agent
# ==========================================
def route_to_agent(state: RouterState) -> str:
    intent = state.get("intent", "unknown")
    confidence = state.get("confidence", "low")
    is_authenticated = state.get("is_authenticated", False)

    valid_intents = ["order_query", "product_query", "support_query"]

    if intent not in valid_intents:
        return "fallback_response"

    _STRING_CONFIDENCE = {"high": 1.0, "medium": 0.5, "low": 0.2}
    try:
        confidence_score = float(confidence)
    except (ValueError, TypeError):
        confidence_score = _STRING_CONFIDENCE.get(str(confidence).lower(), 0.5)

    if confidence_score < settings.intent_confidence_threshold:
        return "ask_clarification"

    if not is_authenticated and intent != "product_query":
        return "access_denied"

    if intent == "order_query":
        return "run_order_agent"
    elif intent == "support_query":
        return "run_support_agent"
    else:
        return "run_product_agent"


# ==========================================
# NODE 2 — run_order_agent
# ==========================================
def run_order_agent(state: RouterState) -> RouterState:
    logger.info("Router: running Order Agent", extra={"node_name": "run_order_agent"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = (
        create_span(
            trace_id=trace_id,
            name="order_agent",
            parent_observation_id=parent_id,
            input_data={"message": state["message"]},
        )
        if trace_id
        else None
    )

    result = order_agent.invoke(
        {
            "message": state["message"],
            "user_id": state["user_id"],
            "session_id": state["session_id"],
            "history": state["history"],
            "session_context": state.get("session_context") or {},
            "langfuse_trace_id": trace_id,
            "langfuse_parent_span_id": span.id if span else parent_id,
        }
    )

    state["response"] = result.get("response", "")
    state["agent_used"] = "order_agent"
    state["session_context"] = result.get("session_context") or state.get("session_context") or {}

    if span:
        end_span(span, {"response_length": len(state["response"])})

    return state


# ==========================================
# NODE 3 — run_product_agent
# ==========================================
def run_product_agent(state: RouterState) -> RouterState:
    logger.info("Router: running Product Agent", extra={"node_name": "run_product_agent"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = (
        create_span(
            trace_id=trace_id,
            name="product_agent",
            parent_observation_id=parent_id,
            input_data={"message": state["message"]},
        )
        if trace_id
        else None
    )

    result = product_agent.invoke(
        {
            "message": state["message"],
            "user_id": state["user_id"],
            "session_id": state["session_id"],
            "history": state["history"],
            "session_context": state.get("session_context") or {},
            "langfuse_trace_id": trace_id,
            "langfuse_parent_span_id": span.id if span else parent_id,
        }
    )

    state["response"] = result.get("response", "")
    state["products"] = result.get("products", None)
    state["agent_used"] = "product_agent"
    state["session_context"] = result.get("session_context") or state.get("session_context") or {}

    if span:
        end_span(span, {"response_length": len(state["response"])})

    return state


# ==========================================
# NODE 4 — run_support_agent
# ==========================================
def run_support_agent(state: RouterState) -> RouterState:
    logger.info("Router: running Support Agent", extra={"node_name": "run_support_agent"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = (
        create_span(
            trace_id=trace_id,
            name="support_agent",
            parent_observation_id=parent_id,
            input_data={"message": state["message"]},
        )
        if trace_id
        else None
    )

    result = support_agent.invoke(
        {
            "message": state["message"],
            "user_id": state["user_id"],
            "session_id": state["session_id"],
            "history": state["history"],
            "session_context": state.get("session_context") or {},
            "langfuse_trace_id": trace_id,
            "langfuse_parent_span_id": span.id if span else parent_id,
        }
    )

    state["response"] = result.get("response", "")
    state["agent_used"] = "support_agent"
    state["session_context"] = result.get("session_context") or state.get("session_context") or {}

    if span:
        end_span(span, {"response_length": len(state["response"])})

    return state


# ==========================================
# NODE 5 — ask_clarification (low confidence)
# ==========================================
def ask_clarification(state: RouterState) -> RouterState:
    logger.info("Router: low confidence — asking for clarification", extra={"node_name": "ask_clarification"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = (
        create_span(
            trace_id=trace_id,
            name="ask_clarification",
            parent_observation_id=parent_id,
            input_data={
                "intent": state.get("intent"),
                "confidence": state.get("confidence"),
                "reason": state.get("reason"),
            },
        )
        if trace_id
        else None
    )

    state["response"] = (
        "I want to make sure I help you correctly. "
        "Could you clarify what you need?\n\n"
        "1. Track or check an order — ask about delivery, "
        "status, tracking\n"
        "2. Find a product — search by category, price, or brand\n"
        "3. Raise a support request — report an issue, "
        "request a refund, or cancel an order"
    )
    state["agent_used"] = "clarification"

    if span:
        end_span(span, {"reason": "low_confidence_intent"})

    return state


# ==========================================
# NODE 6 — access_denied
# ==========================================


def access_denied(state: RouterState) -> RouterState:
    logger.info("Router: access denied for guest user", extra={"node_name": "access_denied"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = (
        create_span(
            trace_id=trace_id,
            name="access_denied",
            parent_observation_id=parent_id,
            input_data={"intent": state.get("intent")},
        )
        if trace_id
        else None
    )

    if state.get("intent") == "order_query":
        state["response"] = (
            "To track your orders please log in with your "
            "Google account. As a guest you can browse and "
            "search for products freely."
        )
    else:
        state["response"] = (
            "To raise a support request please log in with "
            "your Google account. As a guest you can browse "
            "and search for products freely."
        )

    state["agent_used"] = "access_control"

    if span:
        end_span(span, {"reason": "guest_blocked"})

    return state


# ==========================================
# NODE 7 — fallback_response
# ==========================================
def fallback_response(state: RouterState) -> RouterState:
    logger.info("Router: fallback response for unknown intent", extra={"node_name": "fallback_response"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = (
        create_span(
            trace_id=trace_id,
            name="fallback_response",
            parent_observation_id=parent_id,
            input_data={"intent": state.get("intent")},
        )
        if trace_id
        else None
    )

    state["response"] = (
        "I can help you with:\n\n"
        "1. Order tracking — ask about your order status or delivery\n"
        "2. Support — report issues, request refunds, cancel orders\n"
        "3. Product search — we have these categories:\n\n"
        "   Computers & Accessories\n"
        "     laptops, keyboards, mice, cables, chargers, USB hubs\n\n"
        "   Electronics\n"
        "     headphones, speakers, smartwatches, cameras\n\n"
        "   Home & Kitchen\n"
        "     fans, mixers, kettles, irons, geysers, vacuum cleaners\n\n"
        "   Office Products\n"
        "     pens, notebooks\n\n"
        "What are you looking for?"
    )
    state["agent_used"] = "fallback"

    if span:
        end_span(span, {"reason": "unknown_intent"})

    return state


# ==========================================
# BUILD INTENT ROUTER
# ==========================================
def build_intent_router():
    graph = StateGraph(RouterState)

    graph.add_node("intent_router", intent_router)
    graph.add_node("run_order_agent", run_order_agent)
    graph.add_node("run_product_agent", run_product_agent)
    graph.add_node("run_support_agent", run_support_agent)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("access_denied", access_denied)
    graph.add_node("fallback_response", fallback_response)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_to_agent,
        {
            "run_order_agent": "run_order_agent",
            "run_product_agent": "run_product_agent",
            "run_support_agent": "run_support_agent",
            "ask_clarification": "ask_clarification",
            "access_denied": "access_denied",
            "fallback_response": "fallback_response",
        },
    )

    graph.add_edge("run_order_agent", END)
    graph.add_edge("run_product_agent", END)
    graph.add_edge("run_support_agent", END)
    graph.add_edge("ask_clarification", END)
    graph.add_edge("access_denied", END)
    graph.add_edge("fallback_response", END)

    return graph.compile()


intent_router_graph = build_intent_router()
