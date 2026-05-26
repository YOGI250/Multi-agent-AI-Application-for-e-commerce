# agents/order_agent.py

import re
import json
import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from config.settings import settings
from subgraphs.shipment_tracking import shipment_tracking_subgraph
from tools.order_tools import fetch_order_data
from langfuse_helpers.tracing import (
    create_span, end_span,
    create_generation, get_prompt,
    extract_token_usage
)

logger = logging.getLogger(__name__)


# ==========================================
# STATE
# ==========================================
class OrderAgentState(TypedDict):
    message:                str
    user_id:                str
    session_id:             str
    history:                list
    order_id:               Optional[str]
    order_id_found:         Optional[bool]
    order_data:             Optional[dict]
    order_analysis:         Optional[dict]
    tracking_info:          Optional[dict]
    response:               Optional[str]
    langfuse_trace_id:      Optional[str]
    langfuse_parent_span_id: Optional[str]
    all_orders:       Optional[list]
    show_all_orders:  Optional[bool]



# ==========================================
# NODE 1 — validate_input
# ==========================================
def validate_input(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: validate_input")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "validate_input",
        parent_observation_id = parent_id,
        input_data            = {"message": state.get("message", "")}
    ) if trace_id else None

    message = state.get("message", "").upper()

    # check for specific order ID
    match = re.search(r'ORD-[A-Z0-9-]+', message)

    if match:
        state["order_id"]       = match.group()
        state["order_id_found"] = True
        state["show_all_orders"] = False
    else:
        # check if user wants to see all their orders
        show_all_keywords = [
    "MY ORDERS", "ALL ORDERS", "SHOW ORDERS",
    "LIST ORDERS", "MY ORDER LIST", "WHAT ORDERS",
    "DO I HAVE", "SHOW MY", "ALL MY",
    "WHERE IS MY ORDER", "WHERE IS THE ORDER",
    "CHECK MY ORDER", "MY ORDER STATUS",
    "ORDER STATUS", "SHOW ME MY ORDER",
    "WHAT IS MY ORDER", "TRACK MY ORDER"
]
        wants_all_orders = any(
            keyword in message for keyword in show_all_keywords
        )

        state["order_id"]        = None
        state["order_id_found"]  = wants_all_orders
        state["show_all_orders"] = wants_all_orders

    if span:
        end_span(span, {
            "order_id":        state["order_id"],
            "order_id_found":  state["order_id_found"],
            "show_all_orders": state["show_all_orders"]
        })

    return state


# ==========================================
# EDGE — order_id_found?
# ==========================================
def route_order_found(state: OrderAgentState) -> str:
    if state.get("order_id_found"):
        return "fetch_order_data_node"
    return "error_response"


# ==========================================
# NODE 2 — fetch_order_data_node (tool)
# ==========================================
def fetch_order_data_node(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: fetch_order_data_node")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    order_id  = state.get("order_id")
    user_id   = state.get("user_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "fetch_order_data",
        parent_observation_id = parent_id,
        input_data            = {"order_id": order_id, "user_id": user_id}
    ) if trace_id else None

    if order_id:
        # fetch specific order by ID
        result = fetch_order_data.invoke({"order_id": order_id})

        if not result:
            state["order_data"]     = None
            state["order_id_found"] = False
            if span:
                end_span(span, {"found": False, "reason": "order not in DB"})
            return state

        if result.get("user_id") != user_id:
            state["order_data"]     = None
            state["order_id_found"] = False
            if span:
                end_span(span, {"found": False, "reason": "ownership mismatch"})
            return state

        state["order_data"]      = result
        state["all_orders"]      = None
        state["order_id_found"]  = True

    elif state.get("show_all_orders"):
        # user asked to see all their orders
        from tools.order_tools import fetch_all_orders_for_user
        all_orders = fetch_all_orders_for_user.invoke({"user_id": user_id})

        if not all_orders:
            state["order_data"]     = None
            state["all_orders"]     = None
            state["order_id_found"] = False
        else:
            state["order_data"]     = all_orders[0]
            state["all_orders"]     = all_orders
            state["order_id_found"] = True

    else:
        # no order ID and not asking for all orders
        state["order_data"]     = None
        state["all_orders"]     = None
        state["order_id_found"] = False

    if span:
        end_span(span, {
            "found":        state["order_id_found"],
            "orders_count": len(state.get("all_orders") or [])
        })

    return state


# ==========================================
# EDGE — order_data_found?
# ==========================================
def route_order_data_found(state: OrderAgentState) -> str:
    if state.get("order_data"):
        return "analyze_order_status"
    return "error_response"


# ==========================================
# NODE 3 — analyze_order_status (LLM)
# ==========================================
def analyze_order_status(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: analyze_order_status")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    order     = state.get("order_data", {})

    fallback_prompt = f"""You are an e-commerce order analyst.
Analyze the following order and identify any issues.

Order details:
- Order ID: {order.get('order_id')}
- Status: {order.get('status')}
- Order date: {order.get('order_date')}
- Expected delivery: {order.get('expected_delivery')}
- Order value: ₹{order.get('order_value')}
- Carrier: {order.get('carrier')}


Respond in this exact JSON format:
{{
  "issue_type": "delayed/on_track/delivered/cancelled/processing",
  "summary": "one sentence summary of the order situation"
}}"""

    prompt_text, prompt_version = get_prompt(
        "analyze_order_status",
        label   = settings.order_analysis_prompt_label,
        fallback = fallback_prompt
    )

    # compile variables if using LangFuse template
    if "{{order_id}}" in prompt_text:
        from langfuse_helpers.tracing import compile_prompt
        prompt_text = compile_prompt(
            prompt_text,
            order_id          = order.get("order_id", ""),
            status            = order.get("status", ""),
            order_date        = order.get("order_date", ""),
            expected_delivery = order.get("expected_delivery", ""),
            order_value       = order.get("order_value", ""),
            carrier           = order.get("carrier", "")
        )

    llm      = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
    response = llm.invoke(prompt_text)
    usage    = extract_token_usage(response)

    if trace_id:
        create_generation(
            trace_id              = trace_id,
            name                  = "analyze_order_status",
            model                 = settings.llm_model_name,
            prompt                = prompt_text,
            response              = response.content,
            usage                 = usage,
            parent_observation_id = parent_id,
            prompt_name           = "analyze_order_status",
            prompt_version        = prompt_version
        )

    try:
        text       = response.content.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        analysis   = json.loads(json_match.group()) if json_match else {
            "issue_type": "unknown", "summary": text
        }
    except Exception:
        analysis = {
            "issue_type": "unknown",
            "summary": response.content
        }

    state["order_analysis"] = analysis
    return state


# ==========================================
# NODE 4 — shipment_tracking_node (subgraph)
# ==========================================
def shipment_tracking_node(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: shipment_tracking_node")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    order     = state.get("order_data", {})

    if not order.get("tracking_number"):
        state["tracking_info"] = {}
        return state

    # create a span for the subgraph container
    span = create_span(
        trace_id              = trace_id,
        name                  = "shipment_tracking_subgraph",
        parent_observation_id = parent_id,
        input_data            = {
            "tracking_number": order.get("tracking_number"),
            "carrier":         order.get("carrier")
        }
    ) if trace_id else None

    subgraph_result = shipment_tracking_subgraph.invoke({
        "order_data":              order,
        "langfuse_trace_id":       trace_id,
        "langfuse_parent_span_id": span.id if span else parent_id
    })

    state["tracking_info"] = {
        "carrier_name":     subgraph_result.get("carrier_name"),
        "eta":              subgraph_result.get("eta"),
        "current_location": subgraph_result.get("current_location"),
        "tracking_events":  subgraph_result.get("tracking_events", [])
    }

    if span:
        end_span(span, {"eta": state["tracking_info"].get("eta")})

    return state


# ==========================================
# NODE 5 — generate_response (LLM)
# ==========================================
def generate_response(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: generate_response")

    from langfuse_helpers.tracing import compile_prompt

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    order     = state.get("order_data", {})
    analysis  = state.get("order_analysis", {})
    tracking  = state.get("tracking_info", {})
    history   = state.get("history", [])
    all_orders = state.get("all_orders")

    history_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-4:]
    ]) if history else "No previous conversation"

    # ==========================================
    # CASE 1 — user asked to see all their orders
    # ==========================================
    if all_orders and len(all_orders) > 1 and not state.get("order_id"):
        orders_text = "\n".join([
            f"- Order ID: {o['order_id']} | "
            f"Items: {o['items']} | "
            f"Status: {o['status']} | "
            f"Value: Rs.{o['order_value']} | "
            f"Expected delivery: {o['expected_delivery']}"
            for o in all_orders
        ])

        all_orders_prompt = f"""You are a helpful e-commerce customer support assistant.
The customer asked to see their orders. Here are all their orders:

{orders_text}

Customer message: {state.get('message')}

List ALL their orders clearly in a friendly way. For each order mention
the order ID, items, status, value and expected delivery date.
Tell them they can ask about any specific order using its order ID
for more details like tracking information."""

        llm      = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
        response = llm.invoke(all_orders_prompt)
        usage    = extract_token_usage(response)

        if trace_id:
            create_generation(
                trace_id              = trace_id,
                name                  = "generate_response",
                model                 = settings.llm_model_name,
                prompt                = all_orders_prompt,
                response              = response.content,
                usage                 = usage,
                parent_observation_id = parent_id
            )

        state["response"] = response.content
        return state

    # ==========================================
    # CASE 2 — specific order query
    # ==========================================
    fallback_prompt = f"""You are a helpful e-commerce customer support assistant.
Answer the customer's question about their order.

Previous conversation:
{history_text}

Order details:
- Order ID: {order.get('order_id')}
- Status: {order.get('status')}
- Items: {order.get('items')}
- Order value: ₹{order.get('order_value')}
- Expected delivery: {order.get('expected_delivery')}
- Carrier: {order.get('carrier')}

Tracking information:
- Current location: {tracking.get('current_location', 'Not available')}
- ETA: {tracking.get('eta', 'Not available')}
- Situation summary: {analysis.get('summary', '')}

Customer message: {state.get('message')}

Write a helpful, friendly, and concise response. Include the order ID,
current status, and expected delivery date. Be empathetic if there is a delay."""

    prompt_text, prompt_version = get_prompt(
        "order_generate_response",
        label    = settings.order_response_prompt_label,    
        fallback = fallback_prompt
    )

    if "{{order_id}}" in prompt_text:
        prompt_text = compile_prompt(
            prompt_text,
            history           = history_text,
            order_id          = order.get("order_id", ""),
            status            = order.get("status", ""),
            items             = str(order.get("items", "")),
            order_value       = order.get("order_value", ""),
            expected_delivery = order.get("expected_delivery", ""),
            carrier           = order.get("carrier", ""),
            current_location  = tracking.get("current_location", "Not available"),
            eta               = tracking.get("eta", "Not available"),
            summary           = analysis.get("summary", ""),
            message           = state.get("message", "")
        )

    llm      = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
    response = llm.invoke(prompt_text)
    usage    = extract_token_usage(response)

    if trace_id:
        create_generation(
            trace_id              = trace_id,
            name                  = "generate_response",
            model                 = settings.llm_model_name,
            prompt                = prompt_text,
            response              = response.content,
            usage                 = usage,
            parent_observation_id = parent_id,
            prompt_name           = "order_generate_response",
            prompt_version        = prompt_version
        )

    state["response"] = response.content
    return state

# ==========================================
# NODE 6 — error_response
# ==========================================
def error_response(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: error_response")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    order_id  = state.get("order_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "error_response",
        parent_observation_id = parent_id,
        input_data            = {"order_id": order_id}
    ) if trace_id else None

    if order_id:
        state["response"] = (
            f"I could not find order {order_id} associated "
            f"with your account. Please check your order ID "
            f"and try again. Your order ID looks like ORD-XXXX "
            f"(e.g. ORD-1001) and can be found in your order "
            f"confirmation email."
        )
    else:
        state["response"] = (
            "I need your order ID to look up your order. "
            "You can type 'show my orders' to see all your orders "
            "and their order IDs. "
            "Then share the order ID and I will check the status "
            "for you."
        )

    if span:
        end_span(span, {"response_type": "error"})

    return state


# ==========================================
# BUILD ORDER AGENT
# ==========================================
def build_order_agent():
    graph = StateGraph(OrderAgentState)

    graph.add_node("validate_input",        validate_input)
    graph.add_node("fetch_order_data_node", fetch_order_data_node)
    graph.add_node("analyze_order_status",  analyze_order_status)
    graph.add_node("shipment_tracking_node",shipment_tracking_node)
    graph.add_node("generate_response",     generate_response)
    graph.add_node("error_response",        error_response)

    graph.set_entry_point("validate_input")

    graph.add_conditional_edges(
        "validate_input",
        route_order_found,
        {
            "fetch_order_data_node": "fetch_order_data_node",
            "error_response":        "error_response"
        }
    )

    graph.add_conditional_edges(
        "fetch_order_data_node",
        route_order_data_found,
        {
            "analyze_order_status": "analyze_order_status",
            "error_response":       "error_response"
        }
    )

    graph.add_edge("analyze_order_status",   "shipment_tracking_node")
    graph.add_edge("shipment_tracking_node", "generate_response")
    graph.add_edge("generate_response",      END)
    graph.add_edge("error_response",         END)

    return graph.compile()


order_agent = build_order_agent()