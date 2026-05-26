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
    create_span, end_span,
    create_generation,
    get_prompt, compile_prompt,
    extract_token_usage
)

logger = logging.getLogger(__name__)


# ==========================================
# STATE
# ==========================================
class RouterState(TypedDict):
    message:                 str
    user_id:                 str
    session_id:              str
    history:                 list
    is_authenticated:        bool
    intent:                  Optional[str]
    confidence:              Optional[str]
    reason:                  Optional[str]
    response:                Optional[str]
    agent_used:              Optional[str]
    langfuse_trace_id:       Optional[str]
    langfuse_parent_span_id: Optional[str]


# ==========================================
# NODE 1 — intent_router (LLM)
# ==========================================
def intent_router(state: RouterState) -> RouterState:
    logger.info("Intent Router: classifying message")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    message   = state.get("message", "")
    history   = state.get("history", [])

    history_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-3:]
    ]) if history else "No previous conversation"

    fallback_prompt = f"""You are an intent classifier for an e-commerce customer support system.
Classify the user message into exactly one of these intents.

Intents:
- order_query: questions about orders, delivery, tracking, shipment status
- product_query: searching for products, recommendations, product comparisons
- support_query: complaints, refunds, damaged items, wrong items, cancellations

If the message does not match any of these intents, use: unknown

Previous conversation:
{history_text}

User message: "{message}"

Respond ONLY with a JSON object. No explanation.
{{
  "intent": "order_query or product_query or support_query or unknown",
  "confidence": "high or medium or low",
  "reason": "one sentence explaining why"
}}"""

    prompt_text, prompt_version = get_prompt(
        "intent_router",
        label    = settings.intent_router_prompt_label,
        fallback = fallback_prompt
    )

    if "{{message}}" in prompt_text:
        prompt_text = compile_prompt(
            prompt_text,
            message = message,
            history = history_text
        )

    llm      = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
    response = llm.invoke(prompt_text)
    usage    = extract_token_usage(response)

    if trace_id:
        create_generation(
            trace_id              = trace_id,
            name                  = "intent_router",
            model                 = settings.llm_model_name,
            prompt                = prompt_text,
            response              = response.content,
            usage                 = usage,
            parent_observation_id = parent_id,
            prompt_name           = "intent_router",
            prompt_version        = prompt_version
        )

    try:
        text  = response.content.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        result = json.loads(match.group()) if match else {
            "intent":     "unknown",
            "confidence": "low",
            "reason":     "Could not classify intent"
        }
    except Exception:
        result = {
            "intent":     "unknown",
            "confidence": "low",
            "reason":     "Could not classify intent"
        }

    state["intent"]     = result.get("intent", "unknown")
    state["confidence"] = result.get("confidence", "low")
    state["reason"]     = result.get("reason", "")

    logger.info(f"Intent: {state['intent']} | Confidence: {state['confidence']}")
    return state


# ==========================================
# EDGE — route to correct agent
# ==========================================
def route_to_agent(state: RouterState) -> str:
    intent           = state.get("intent", "unknown")
    confidence       = state.get("confidence", "low")
    is_authenticated = state.get("is_authenticated", False)

    valid_intents = ["order_query", "product_query", "support_query"]

    if intent not in valid_intents:
        return "fallback_response"

    
    # Low or medium confidence — ask user to clarify rather than guess
    if confidence in ("low", "medium"):
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
    logger.info("Router: running Order Agent")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "order_agent",
        parent_observation_id = parent_id,
        input_data            = {"message": state["message"]}
    ) if trace_id else None

    result = order_agent.invoke({
        "message":                 state["message"],
        "user_id":                 state["user_id"],
        "session_id":              state["session_id"],
        "history":                 state["history"],
        "langfuse_trace_id":       trace_id,
        "langfuse_parent_span_id": span.id if span else parent_id
    })

    state["response"]   = result.get("response", "")
    state["agent_used"] = "order_agent"

    if span:
        end_span(span, {"response_length": len(state["response"])})

    return state


# ==========================================
# NODE 3 — run_product_agent
# ==========================================
def run_product_agent(state: RouterState) -> RouterState:
    logger.info("Router: running Product Agent")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "product_agent",
        parent_observation_id = parent_id,
        input_data            = {"message": state["message"]}
    ) if trace_id else None

    result = product_agent.invoke({
        "message":                 state["message"],
        "user_id":                 state["user_id"],
        "session_id":              state["session_id"],
        "history":                 state["history"],
        "langfuse_trace_id":       trace_id,
        "langfuse_parent_span_id": span.id if span else parent_id
    })

    state["response"]   = result.get("response", "")
    state["agent_used"] = "product_agent"

    if span:
        end_span(span, {"response_length": len(state["response"])})

    return state


# ==========================================
# NODE 4 — run_support_agent
# ==========================================
def run_support_agent(state: RouterState) -> RouterState:
    logger.info("Router: running Support Agent")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "support_agent",
        parent_observation_id = parent_id,
        input_data            = {"message": state["message"]}
    ) if trace_id else None

    result = support_agent.invoke({
        "message":                 state["message"],
        "user_id":                 state["user_id"],
        "session_id":              state["session_id"],
        "history":                 state["history"],
        "langfuse_trace_id":       trace_id,
        "langfuse_parent_span_id": span.id if span else parent_id
    })

    state["response"]   = result.get("response", "")
    state["agent_used"] = "support_agent"

    if span:
        end_span(span, {"response_length": len(state["response"])})

    return state

# ==========================================
# NODE 5 — ask_clarification (low confidence)
# ==========================================
def ask_clarification(state: RouterState) -> RouterState:
    logger.info("Router: low confidence — asking for clarification")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "ask_clarification",
        parent_observation_id = parent_id,
        input_data            = {
            "intent":     state.get("intent"),
            "confidence": state.get("confidence"),
            "reason":     state.get("reason")
        }
    ) if trace_id else None

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
    logger.info("Router: access denied for guest user")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "access_denied",
        parent_observation_id = parent_id,
        input_data            = {"intent": state.get("intent")}
    ) if trace_id else None

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
    logger.info("Router: fallback response for unknown intent")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "fallback_response",
        parent_observation_id = parent_id,
        input_data            = {"intent": state.get("intent")}
    ) if trace_id else None

    state["response"] = (
        "I can help you with the following:\n\n"
        "1. Order tracking — ask about your order status or delivery\n"
        "2. Product search — find products by category, price, or brand\n"
        "3. Support — report issues like damaged items or request refunds\n\n"
        "Please try rephrasing your question."
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

    graph.add_node("intent_router",     intent_router)
    graph.add_node("run_order_agent",   run_order_agent)
    graph.add_node("run_product_agent", run_product_agent)
    graph.add_node("run_support_agent", run_support_agent)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("access_denied",     access_denied)
    graph.add_node("fallback_response", fallback_response)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_to_agent,
        {
            "run_order_agent":   "run_order_agent",
            "run_product_agent": "run_product_agent",
            "run_support_agent": "run_support_agent",
            "ask_clarification": "ask_clarification",
            "access_denied":     "access_denied",
            "fallback_response": "fallback_response"
        }
    )

    graph.add_edge("run_order_agent",   END)
    graph.add_edge("run_product_agent", END)
    graph.add_edge("run_support_agent", END)
    graph.add_edge("ask_clarification", END)
    graph.add_edge("access_denied",     END)
    graph.add_edge("fallback_response", END)

    return graph.compile()


intent_router_graph = build_intent_router()