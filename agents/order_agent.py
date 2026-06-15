# agents/order_agent.py

import re
import json
import logging
from typing import TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
from langchain_groq import ChatGroq
from config.settings import settings
from subgraphs.shipment_tracking import shipment_tracking_subgraph
from tools.order_tools import fetch_order_data, fetch_all_orders_for_user
from langfuse_helpers.tracing import (
    create_span,
    end_span,
    create_generation,
    get_prompt,
    compile_prompt,
    extract_token_usage,
)
from utils.memory import format_context, format_recent_messages, merge_context
from utils.langfuse_context import set_trace_context

logger = logging.getLogger(__name__)


# ==========================================
# STATE
# ==========================================
class OrderAgentState(TypedDict):
    message: str
    user_id: str
    session_id: str
    history: list
    order_id: Optional[str]
    order_id_found: Optional[bool]
    order_data: Optional[dict]
    order_analysis: Optional[dict]
    tracking_info: Optional[dict]
    response: Optional[str]
    session_context: Optional[dict]
    langfuse_trace_id: Optional[str]
    langfuse_parent_span_id: Optional[str]
    langfuse_final_generation_id: Optional[str]
    total_input_tokens: Optional[int]
    total_output_tokens: Optional[int]
    all_orders: Optional[list]
    show_all_orders: Optional[bool]
    messages: Optional[List[Any]]


# ==========================================
# NODE 1 — validate_input
# ==========================================
def validate_input(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: validate_input", extra={"node_name": "validate_input"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = (
        create_span(
            trace_id=trace_id,
            name="validate_input",
            parent_observation_id=parent_id,
            input_data={"message": state.get("message", "")},
        )
        if trace_id
        else None
    )

    state["order_id"] = None
    state["order_id_found"] = False

    message = state.get("message", "").upper()
    msg_lower = state.get("message", "").lower()
    ctx = state.get("session_context") or {}
    order_statuses = ctx.get("order_statuses", {})

    # ── Step 1: explicit order ID in message ──
    match = re.search(r"ORD-[A-Z0-9-]+", message)

    if match:
        state["order_id"] = match.group()
        state["order_id_found"] = True
        state["show_all_orders"] = False

    else:
        # ── Step 2: status-based reference from session context ──
        # e.g. "tell me about the delayed one" → resolve via stored statuses
        resolved_id = None
        for status, oids in order_statuses.items():
            if status in msg_lower and len(oids) == 1:
                resolved_id = oids[0]
                break

        if resolved_id:
            state["order_id"] = resolved_id
            state["order_id_found"] = True
            state["show_all_orders"] = False

        else:
            # ── Step 3: pattern-based "show all orders" detection ──
            wants_all_orders = bool(
                # "show me orders", "list the orders", "track my order"
                re.search(r"\b(show|list|see|view|display|check|track|get)\b.*\borders?\b", msg_lower)
                # "my orders", "all orders", "the order", "any orders"
                or re.search(r"\b(my|all|the|any)\s+orders?\b", msg_lower)
                # "order status", "order history", "order list"
                or re.search(r"\border\s+(status|list|count|history)\b", msg_lower)
                # "how many orders", "total orders", "number of orders"
                or re.search(r"\b(how\s+many|total|number\s+of|count\s+of)\b.*\borders?\b", msg_lower)
                # "orders I placed", "orders I have"
                or re.search(r"\borders?\s+i\b", msg_lower)
            )

            # ── Step 4: session-context fallback ──
            # If no keyword matched but there is a specific order_id in context
            # (from any prior agent — order or support), use it directly.
            # e.g. "when will I get the refund" after a support cancellation flow.
            if not wants_all_orders and ctx.get("order_id"):
                state["order_id"] = ctx["order_id"]
                state["order_id_found"] = True
                state["show_all_orders"] = False
            elif not wants_all_orders and ctx.get("topic") in ("order_query", "support_query"):
                wants_all_orders = True

            if not state.get("order_id_found"):
                state["order_id"] = None
                state["order_id_found"] = wants_all_orders
                state["show_all_orders"] = wants_all_orders

    if span:
        end_span(
            span,
            {
                "order_id": state["order_id"],
                "order_id_found": state["order_id_found"],
                "show_all_orders": state["show_all_orders"],
            },
        )

    return state


# ==========================================
# EDGE — order_id_found?
# ==========================================
def route_order_found(state: OrderAgentState) -> str:
    if state.get("order_id_found"):
        return "fetch_order_data_node"
    return "error_response"


# ==========================================
# NODE 2a — prepare_order_fetch
# Builds an AIMessage with the appropriate tool_call so ToolNode can execute it.
# ==========================================
def prepare_order_fetch(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: prepare_order_fetch", extra={"node_name": "prepare_order_fetch"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    order_id = state.get("order_id")
    user_id = state.get("user_id")
    show_all = state.get("show_all_orders")

    span = (
        create_span(
            trace_id=trace_id,
            name="prepare_order_fetch",
            parent_observation_id=parent_id,
            input_data={"order_id": order_id, "show_all": show_all},
        )
        if trace_id
        else None
    )

    if order_id and not show_all:
        tool_call = {
            "name": "fetch_order_data",
            "args": {"order_id": order_id},
            "id": "order_fetch_1",
            "type": "tool_call",
        }
    else:
        tool_call = {
            "name": "fetch_all_orders_for_user",
            "args": {"user_id": user_id},
            "id": "order_fetch_1",
            "type": "tool_call",
        }

    state["messages"] = [AIMessage(content="", tool_calls=[tool_call])]

    if span:
        end_span(span, {"tool": tool_call["name"], "args": tool_call["args"]})

    return state


# ==========================================
# NODE 2b — process_order_result
# Reads the ToolMessage from ToolNode and updates state fields.
# ==========================================
def process_order_result(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: process_order_result", extra={"node_name": "process_order_result"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    user_id = state.get("user_id")
    messages = state.get("messages") or []

    span = (
        create_span(
            trace_id=trace_id,
            name="process_order_result",
            parent_observation_id=parent_id,
            input_data={"order_id": state.get("order_id"), "user_id": user_id},
        )
        if trace_id
        else None
    )

    tool_msg = next((m for m in messages if isinstance(m, ToolMessage)), None)
    result = None
    if tool_msg:
        try:
            result = json.loads(tool_msg.content)
        except (json.JSONDecodeError, TypeError):
            import ast

            try:
                result = ast.literal_eval(tool_msg.content)
            except Exception:
                result = None

    if isinstance(result, list):
        all_orders = result
        if all_orders:
            state["order_data"] = all_orders[0]
            state["all_orders"] = all_orders
            state["order_id_found"] = True
        else:
            state["order_data"] = None
            state["all_orders"] = None
            state["order_id_found"] = False
    elif isinstance(result, dict) and result:
        if result.get("user_id") != user_id:
            state["order_data"] = None
            state["order_id_found"] = False
        else:
            state["order_data"] = result
            state["all_orders"] = None
            state["order_id_found"] = True
    else:
        state["order_data"] = None
        state["all_orders"] = None
        state["order_id_found"] = False

    if span:
        end_span(span, {"found": state["order_id_found"], "orders_count": len(state.get("all_orders") or [])})

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
    logger.info("Order Agent node: analyze_order_status", extra={"node_name": "analyze_order_status"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    order = state.get("order_data", {})

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
        "analyze_order_status", label=settings.order_analysis_prompt_label, fallback=fallback_prompt
    )

    if "{{order_id}}" in prompt_text:
        prompt_text = compile_prompt(
            prompt_text,
            order_id=order.get("order_id", ""),
            status=order.get("status", ""),
            order_date=order.get("order_date", ""),
            expected_delivery=order.get("expected_delivery", ""),
            order_value=order.get("order_value", ""),
            carrier=order.get("carrier", ""),
        )

    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
    try:
        response = llm.invoke(prompt_text)
    except Exception as e:
        logger.error(f"LLM invoke failed in analyze_order_status: {e}")
        state["response"] = "I'm having trouble reaching the AI service right now. Please try again in a moment."
        return state
    usage = extract_token_usage(response)
    state["total_input_tokens"] = (state.get("total_input_tokens") or 0) + usage.get("input", 0)
    state["total_output_tokens"] = (state.get("total_output_tokens") or 0) + usage.get("output", 0)

    if trace_id:
        create_generation(
            trace_id=trace_id,
            name="analyze_order_status",
            model=settings.llm_model_name,
            prompt=prompt_text,
            response=response.content,
            usage=usage,
            parent_observation_id=parent_id,
            prompt_name="analyze_order_status",
            prompt_version=prompt_version,
            agent_used="order_agent",
        )

    try:
        text = response.content.strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        analysis = json.loads(json_match.group()) if json_match else {"issue_type": "unknown", "summary": text}
    except Exception:
        analysis = {"issue_type": "unknown", "summary": response.content}

    state["order_analysis"] = analysis
    return state


# ==========================================
# NODE 4 — shipment_tracking_node (subgraph)
# ==========================================
def shipment_tracking_node(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: shipment_tracking_node", extra={"node_name": "shipment_tracking_node"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    order = state.get("order_data", {})

    if not order.get("tracking_number"):
        state["tracking_info"] = {}
        return state

    # create a span for the subgraph container
    span = (
        create_span(
            trace_id=trace_id,
            name="shipment_tracking_subgraph",
            parent_observation_id=parent_id,
            input_data={"tracking_number": order.get("tracking_number"), "carrier": order.get("carrier")},
        )
        if trace_id
        else None
    )

    subgraph_result = shipment_tracking_subgraph.invoke(
        {"order_data": order, "langfuse_trace_id": trace_id, "langfuse_parent_span_id": span.id if span else parent_id}
    )

    state["tracking_info"] = {
        "carrier_name": subgraph_result.get("carrier_name"),
        "eta": subgraph_result.get("eta"),
        "current_location": subgraph_result.get("current_location"),
        "tracking_events": subgraph_result.get("tracking_events", []),
    }

    if span:
        end_span(span, {"eta": state["tracking_info"].get("eta")})

    return state


# ==========================================
# NODE 5 — generate_response (LLM)
# ==========================================
def generate_response(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: generate_response", extra={"node_name": "generate_response"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    order = state.get("order_data", {})
    analysis = state.get("order_analysis", {})
    tracking = state.get("tracking_info", {})
    history = state.get("history", [])
    all_orders = state.get("all_orders")
    ctx = state.get("session_context") or {}

    context_str = format_context(ctx)
    recent_msgs = format_recent_messages(history, n=2)

    # ==========================================
    # CASE 1 — user asked to see all their orders
    # ==========================================
    if all_orders and len(all_orders) > 1 and not state.get("order_id"):
        import ast as _ast

        msg_lower = state.get("message", "").lower()

        # ── Step 1: product name lookup (e.g. "status of laptop stand") ──
        product_match = None
        for o in all_orders:
            items_raw = o.get("items", "")
            try:
                items_list = _ast.literal_eval(str(items_raw)) if isinstance(items_raw, str) else items_raw
                items_str = " ".join(items_list).lower() if isinstance(items_list, list) else str(items_raw).lower()
            except Exception:
                items_str = str(items_raw).lower()
            words = [w for w in items_str.split() if len(w) > 3]
            if words and any(w in msg_lower for w in words):
                product_match = o
                break

        if product_match:
            state["response"] = (
                f"Here are the details for your order:\n\n"
                f"Order ID  : {product_match['order_id']}\n"
                f"Item      : {product_match['items']}\n"
                f"Status    : {product_match['status']}\n"
                f"Delivery  : {product_match['expected_delivery']}\n"
                f"Carrier   : {product_match['carrier']}\n\n"
                f"Ask me about any order using its ID for tracking and details."
            )
            state["session_context"] = merge_context(
                ctx, {"topic": "order_query", "order_id": product_match.get("order_id")}
            )
            return state

        # ── Step 2: status filter ──
        not_delivered_kw = ["not delivered", "undelivered", "not received", "not yet delivered"]
        delivered_kw = ["delivered", "received", "got it", "already received"]
        shipped_kw = ["shipped", "in transit", "on the way", "dispatched"]
        processing_kw = ["processing", "confirmed", "pending", "not shipped", "placed"]
        cancelled_kw = ["cancelled", "canceled"]
        delayed_kw = ["delayed", "late", "overdue"]

        if any(k in msg_lower for k in not_delivered_kw):
            display_orders = [o for o in all_orders if o.get("status", "").lower() != "delivered"]
            filter_label = "not yet delivered"
        elif any(k in msg_lower for k in delivered_kw):
            display_orders = [o for o in all_orders if o.get("status", "").lower() == "delivered"]
            filter_label = "delivered"
        elif any(k in msg_lower for k in shipped_kw):
            display_orders = [o for o in all_orders if o.get("status", "").lower() in ("shipped", "in transit")]
            filter_label = "shipped"
        elif any(k in msg_lower for k in processing_kw):
            display_orders = [
                o for o in all_orders if o.get("status", "").lower() in ("processing", "confirmed", "pending", "placed")
            ]
            filter_label = "in processing / pending"
        elif any(k in msg_lower for k in cancelled_kw):
            display_orders = [o for o in all_orders if o.get("status", "").lower() in ("cancelled", "canceled")]
            filter_label = "cancelled"
        elif any(k in msg_lower for k in delayed_kw):
            display_orders = [o for o in all_orders if o.get("status", "").lower() in ("delayed", "late")]
            filter_label = "delayed"
        else:
            display_orders = all_orders
            filter_label = None

        # ── Step 3: build response in Python — no LLM needed for structured data ──
        # If user asked for "detailed" / "full" / "more" info, guide them to
        # pick one order rather than dumping a flat list again.
        wants_detail = any(
            w in msg_lower for w in ("detail", "detailed", "full", "more info", "information", "elaborate")
        )

        count = len(display_orders)
        if filter_label:
            header = f"You have {count} order(s) with status '{filter_label}'.\n"
        else:
            header = f"You have {count} order(s).\n"

        lines = [header]
        for o in display_orders:
            lines.append(f"{o['order_id']} - {o['items']} ({o.get('status', 'unknown')})")

        if wants_detail and count > 1:
            lines.append(
                "\nI can show full tracking and delivery details for one order at a time. "
                "Which order would you like to check?"
            )
        else:
            lines.append("\nShare an order ID to get full tracking and delivery details.")

        status_map = {}
        for o in all_orders:
            s = str(o.get("status", "")).lower()
            status_map.setdefault(s, []).append(o["order_id"])

        state["response"] = "\n".join(lines)
        state["session_context"] = merge_context(
            ctx,
            {
                "topic": "order_query",
                "orders_listed": True,
                "order_count": len(all_orders),
                "order_statuses": status_map,
            },
        )
        return state

    # ==========================================
    # CASE 2 — specific order query
    # ==========================================
    fallback_prompt = f"""You are a helpful e-commerce customer support assistant.
Answer the customer's question about their order.

Session context: {context_str}
Recent messages:
{recent_msgs}

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
current status, and expected delivery date. Be empathetic if there is a delay.
Do not address the user by name — messages in the history may look like names but are product searches or typos."""

    prompt_text, prompt_version = get_prompt(
        "order_generate_response", label=settings.order_response_prompt_label, fallback=fallback_prompt
    )

    # Always enforce this regardless of whether prompt came from LangFuse or fallback
    if "do not address the user by name" not in prompt_text.lower():
        prompt_text += "\nDo not address the user by name — messages in the history may look like names but are product searches or typos."

    if "{{order_id}}" in prompt_text:
        prompt_text = compile_prompt(
            prompt_text,
            history=recent_msgs,
            order_id=order.get("order_id", ""),
            status=order.get("status", ""),
            items=str(order.get("items", "")),
            order_value=order.get("order_value", ""),
            expected_delivery=order.get("expected_delivery", ""),
            carrier=order.get("carrier", ""),
            current_location=tracking.get("current_location", "Not available"),
            eta=tracking.get("eta", "Not available"),
            summary=analysis.get("summary", ""),
            message=state.get("message", ""),
        )

    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
    try:
        response = llm.invoke(prompt_text)
    except Exception as e:
        logger.error(f"LLM invoke failed in generate_response: {e}")
        state["response"] = "I'm having trouble reaching the AI service right now. Please try again in a moment."
        return state
    usage = extract_token_usage(response)

    if trace_id:
        gen = create_generation(
            trace_id=trace_id,
            name="generate_response",
            model=settings.llm_model_name,
            prompt=prompt_text,
            response=response.content,
            usage=usage,
            parent_observation_id=parent_id,
            prompt_name="order_generate_response",
            prompt_version=prompt_version,
            agent_used="order_agent",
        )
        state["langfuse_final_generation_id"] = gen.id if gen else None

    state["response"] = response.content
    state["session_context"] = merge_context(ctx, {"topic": "order_query", "order_id": order.get("order_id")})
    return state


# ==========================================
# NODE 6 — error_response
# ==========================================
def error_response(state: OrderAgentState) -> OrderAgentState:
    logger.info("Order Agent node: error_response", extra={"node_name": "error_response"})

    trace_id = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    order_id = state.get("order_id")

    span = (
        create_span(
            trace_id=trace_id, name="error_response", parent_observation_id=parent_id, input_data={"order_id": order_id}
        )
        if trace_id
        else None
    )

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
    _tool_node = ToolNode([fetch_order_data, fetch_all_orders_for_user])

    def order_tool_node_fn(state: OrderAgentState) -> dict:
        logger.info("Order Agent node: order_tool_node", extra={"node_name": "order_tool_node"})
        trace_id = state.get("langfuse_trace_id")
        parent_id = state.get("langfuse_parent_span_id")
        msgs = state.get("messages", [])
        last_ai = next((m for m in reversed(msgs) if hasattr(m, "tool_calls") and m.tool_calls), None)
        tool_name = last_ai.tool_calls[0].get("name", "unknown") if last_ai else "unknown"
        tool_args = last_ai.tool_calls[0].get("args", {}) if last_ai else {}

        span = (
            create_span(
                trace_id=trace_id,
                name="order_tool_node",
                parent_observation_id=parent_id,
                input_data={"tool": tool_name, "args": tool_args},
            )
            if trace_id
            else None
        )

        set_trace_context(trace_id, span.id if span else parent_id)
        result = _tool_node.invoke(state)

        if span:
            tool_msgs = result.get("messages", [])
            output = tool_msgs[0].content[:200] if tool_msgs else None
            end_span(span, {"tool_output": output, "tool": tool_name})

        return result

    graph = StateGraph(OrderAgentState)

    graph.add_node("validate_input", validate_input)
    graph.add_node("fetch_order_data_node", prepare_order_fetch)
    graph.add_node("order_tool_node", order_tool_node_fn)
    graph.add_node("process_order_result", process_order_result)
    graph.add_node("analyze_order_status", analyze_order_status)
    graph.add_node("shipment_tracking_node", shipment_tracking_node)
    graph.add_node("generate_response", generate_response)
    graph.add_node("error_response", error_response)

    graph.set_entry_point("validate_input")

    graph.add_conditional_edges(
        "validate_input",
        route_order_found,
        {"fetch_order_data_node": "fetch_order_data_node", "error_response": "error_response"},
    )

    graph.add_edge("fetch_order_data_node", "order_tool_node")
    graph.add_edge("order_tool_node", "process_order_result")

    graph.add_conditional_edges(
        "process_order_result",
        route_order_data_found,
        {"analyze_order_status": "analyze_order_status", "error_response": "error_response"},
    )

    graph.add_edge("analyze_order_status", "shipment_tracking_node")
    graph.add_edge("shipment_tracking_node", "generate_response")
    graph.add_edge("generate_response", END)
    graph.add_edge("error_response", END)

    return graph.compile()


order_agent = build_order_agent()
