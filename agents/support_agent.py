# agents/support_agent.py

import logging
import json
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from config.settings import settings
from subgraphs.escalation_handler import escalation_handler_subgraph
from tools.support_tools import lookup_policy_tool
from langfuse_helpers.tracing import (
    create_span, end_span,
    create_generation,
    get_prompt, compile_prompt,
    extract_token_usage
)
from utils.memory import format_context, format_recent_messages, merge_context

logger = logging.getLogger(__name__)


# ==========================================
# STATE
# ==========================================
class SupportAgentState(TypedDict):
    message:                 str
    user_id:                 str
    session_id:              str
    history:                 list
    issue_type:              Optional[str]
    order_id:                Optional[str]
    issue_details:           Optional[str]
    severity:                Optional[str]
    order_status:            Optional[str]
    policy_text:             Optional[str]
    ticket_id:               Optional[str]
    ticket_created:          Optional[bool]
    is_duplicate:            Optional[bool]
    days_open:               Optional[int]
    resolution:              Optional[str]
    response:                Optional[str]
    session_context:         Optional[dict]
    langfuse_trace_id:       Optional[str]
    langfuse_parent_span_id: Optional[str]


# ==========================================
# NODE 1 — classify_issue (LLM)
# ==========================================
def classify_issue(state: SupportAgentState) -> SupportAgentState:
    logger.info("Support Agent node: classify_issue")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    message   = state.get("message", "")
    history   = state.get("history", [])

    ctx         = state.get("session_context") or {}
    # Exclude stale order_id — classify_issue must resolve it fresh from the current message
    context_str = format_context({k: v for k, v in ctx.items() if k != "order_id"})
    recent_msgs = format_recent_messages(history, n=2)

    order_statuses = ctx.get("order_statuses", {})
    order_hint = ""
    if order_statuses:
        lines = [f"  {status}: {', '.join(ids)}" for status, ids in order_statuses.items()]
        order_hint = "Known orders by status (use to resolve references like 'my delivered order'):\n" + "\n".join(lines) + "\n\n"

    fallback_prompt = f"""You are a customer support classifier.
Classify the customer complaint and extract key details.

FOLLOW-UP DETECTION:
If current message has no clear issue_type but history shows a recent support interaction:
- "how much time" / "when will it be resolved" → general_query about existing ticket
- "any update" / "what happened" → general_query about existing ticket
- "still not resolved" → use same issue_type as previous complaint

Examples:
- History: "my item is damaged, ticket created" → current: "how much time will it take"
  → issue_type="general_query", use existing ticket from history

Session context: {context_str}
{order_hint}Recent messages:
{recent_msgs}

Customer message: "{message}"

Issue types available:
- damaged_product: product arrived broken or damaged
- wrong_item: received a different product than ordered
- refund: customer wants money back
- cancellation: customer wants to cancel an order
- general_query: general question, not a complaint

For order_id: if the customer says "my delivered order", "my delayed order", etc. and there is
exactly ONE order with that status in the known orders above, use that order_id.
If there are multiple, set order_id to null (will ask user to clarify).
Only extract order_id from the current message or unambiguous context — never guess.

Respond ONLY with a JSON object. No explanation.
{{
  "issue_type": "one of the issue types above",
  "order_id": "resolved order ID or null",
  "details": "brief one sentence description of the issue"
}}"""

    prompt_text, prompt_version = get_prompt(
        "classify_issue",
        label    = settings.intent_router_prompt_label,
        fallback = fallback_prompt
    )
    

    if "{{message}}" in prompt_text:
        prompt_text = compile_prompt(
            prompt_text,
            message = message,
            history = recent_msgs
        )

    llm      = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
    response = llm.invoke(prompt_text)
    usage    = extract_token_usage(response)

    if trace_id:
        create_generation(
            trace_id              = trace_id,
            name                  = "classify_issue",
            model                 = settings.llm_model_name,
            prompt                = prompt_text,
            response              = response.content,
            usage                 = usage,
            parent_observation_id = parent_id,
            prompt_name           = "classify_issue",
            prompt_version        = prompt_version
        )

    try:
        text  = response.content.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        result = json.loads(match.group()) if match else {
            "issue_type": "general_query",
            "order_id":   None,
            "details":    message
        }
    except Exception:
        result = {
            "issue_type": "general_query",
            "order_id":   None,
            "details":    message
        }

    # Explicit order ID in the message always wins over LLM (history can confuse it)
    explicit_match = re.search(r'\bORD-[A-Z0-9]+-\d+\b', message, re.IGNORECASE)
    explicit_order_id = explicit_match.group().upper() if explicit_match else None

    state["issue_type"]    = result.get("issue_type", "general_query")
    state["order_id"]      = explicit_order_id or result.get("order_id")
    state["issue_details"] = result.get("details", message)
    state["ticket_id"]      = None
    state["ticket_created"] = False
    state["is_duplicate"]   = False
    state["days_open"]      = 0
    state["session_context"] = merge_context(ctx, {
        "topic":      "support_query",
        "issue_type": state["issue_type"],
        "order_id":   state["order_id"]
    })

    return state


# ==========================================
# EDGE — needs order id?
# ==========================================
ORDER_REQUIRED_ISSUES = {"damaged_product", "wrong_item", "refund", "cancellation"}

def route_after_classify(state: SupportAgentState) -> str:
    if state.get("issue_type") in ORDER_REQUIRED_ISSUES and not state.get("order_id"):
        return "request_order_id"
    return "assess_severity"


# ==========================================
# NODE 1b — request_order_id
# ==========================================
def request_order_id(state: SupportAgentState) -> SupportAgentState:
    logger.info("Support Agent node: request_order_id")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "request_order_id",
        parent_observation_id = parent_id,
        input_data            = {"issue_type": state.get("issue_type")}
    ) if trace_id else None

    state["response"] = (
        "I'm sorry to hear that. To look into this for you, I'll need your order ID.\n\n"
        "You can find it by typing 'show my orders' to see all your orders, "
        "or check your order confirmation email.\n\n"
        "Your order ID looks like ORD-XXXX (e.g. ORD-1001)."
    )
    state["agent_used"] = "support_agent"

    if span:
        end_span(span, {"reason": "order_id_missing"})

    return state


# ==========================================
# NODE 2 — assess_severity (pure code)
# ==========================================
def assess_severity(state: SupportAgentState) -> SupportAgentState:
    logger.info("Support Agent node: assess_severity")

    trace_id   = state.get("langfuse_trace_id")
    parent_id  = state.get("langfuse_parent_span_id")
    issue_type = state.get("issue_type", "general_query")
    order_id   = state.get("order_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "assess_severity",
        parent_observation_id = parent_id,
        input_data            = {"issue_type": issue_type, "order_id": order_id}
    ) if trace_id else None

    # Fetch order value, status, and validate ownership
    order_value = 0
    user_id = state.get("user_id")
    state["order_status"] = None
    if order_id:
        try:
            from database.connection import SessionLocal
            from database.models import Order
            db = SessionLocal()
            order = db.query(Order).filter(
                Order.order_id == order_id,
                Order.user_id  == user_id
            ).first()
            if order:
                order_value = float(order.order_value or 0)
                state["order_status"] = str(order.status or "unknown").lower()
            else:
                # order not found or doesn't belong to this user
                state["order_id"] = None
            db.close()
        except Exception:
            order_value = 0

    if issue_type in ["damaged_product", "wrong_item", "refund"]:
        if order_value >= settings.support_high_value_threshold:
            severity = "HIGH"
        else:
            severity = "MEDIUM"
    elif issue_type == "cancellation":
        severity = "MEDIUM"
    else:
        severity = "LOW"

    state["severity"] = severity

    if span:
        end_span(span, {
            "severity":    severity,
            "order_value": order_value,
            "issue_type":  issue_type
        })

    return state


# ==========================================
# NODE 3 — lookup_policy_node (tool)
# ==========================================
def lookup_policy_node(
    state: SupportAgentState
) -> SupportAgentState:
    logger.info("Support Agent node: lookup_policy_node")

    trace_id   = state.get("langfuse_trace_id")
    parent_id  = state.get("langfuse_parent_span_id")
    issue_type = state.get("issue_type", "general_query")

    span = create_span(
        trace_id              = trace_id,
        name                  = "lookup_policy",
        parent_observation_id = parent_id,
        input_data            = {"issue_type": issue_type}
    ) if trace_id else None

    policy = lookup_policy_tool.invoke({"issue_type": issue_type})
    state["policy_text"] = policy.get(
        "policy_text",
        "Please contact our support team for assistance."
    )

    if span:
        end_span(span, {
            "policy_found": bool(policy.get("policy_text")),
            "issue_type":   issue_type
        })

    return state


# ==========================================
# EDGE — severity_high?
# ==========================================
def route_severity(state: SupportAgentState) -> str:
    severity = state.get("severity", "LOW")
    # HIGH and MEDIUM are actionable — they need a ticket
    # Only LOW (general queries) skips ticket creation
    if severity in ("HIGH", "MEDIUM"):
        return "escalation_handler_node"
    return "draft_resolution"


# ==========================================
# NODE 4 — escalation_handler_node (subgraph)
# ==========================================
def escalation_handler_node(
    state: SupportAgentState
) -> SupportAgentState:
    logger.info("Support Agent node: escalation_handler_node")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "escalation_handler_subgraph",
        parent_observation_id = parent_id,
        input_data            = {
            "user_id":    state.get("user_id"),
            "issue_type": state.get("issue_type"),
            "severity":   state.get("severity")
        }
    ) if trace_id else None

    result = escalation_handler_subgraph.invoke({
        "user_id":                 state.get("user_id"),
        "issue_type":              state.get("issue_type"),
        "order_id":                state.get("order_id"),
        "severity":                state.get("severity"),
        "langfuse_trace_id":       trace_id,
        "langfuse_parent_span_id": span.id if span else parent_id
    })

    state["ticket_id"]      = result.get("ticket_id")
    state["ticket_created"] = result.get("ticket_created", False)
    state["is_duplicate"]   = result.get("is_duplicate", False)
    state["days_open"]      = result.get("days_open", 0)

    if span:
        end_span(span, {
            "ticket_id":      state["ticket_id"],
            "ticket_created": state["ticket_created"],
            "is_duplicate":   state["is_duplicate"],
            "days_open":      state["days_open"]
        })

    return state


# ==========================================
# NODE 5 — draft_resolution (LLM)
# ==========================================
def draft_resolution(
    state: SupportAgentState
) -> SupportAgentState:
    logger.info("Support Agent node: draft_resolution")

    trace_id       = state.get("langfuse_trace_id")
    parent_id      = state.get("langfuse_parent_span_id")
    policy_text    = state.get("policy_text", "")
    issue_type     = state.get("issue_type", "")
    issue_details  = state.get("issue_details", "")
    order_id       = state.get("order_id")
    order_status   = state.get("order_status") or "unknown"
    ticket_id      = state.get("ticket_id")
    ticket_created = state.get("ticket_created", False)
    is_duplicate   = state.get("is_duplicate", False)
    days_open      = state.get("days_open", 0)
    history        = state.get("history", [])

    ctx         = state.get("session_context") or {}
    context_str = format_context(ctx)
    recent_msgs = format_recent_messages(history, n=2)

    if ticket_created and ticket_id:
        ticket_info = f"A new support ticket has been created. Ticket ID: {ticket_id}"
    elif ticket_id and is_duplicate:
        if days_open == 0:
            ticket_info = (
                f"DUPLICATE COMPLAINT. Existing open ticket: {ticket_id}. "
                f"Ticket was created today. "
                f"INSTRUCTION: Tell customer their ticket was recently created "
                f"and to allow 48 hours. Do NOT say a new ticket was created."
            )
        elif 1 <= days_open <= 3:
            ticket_info = (
                f"DUPLICATE COMPLAINT. Existing open ticket: {ticket_id}. "
                f"Ticket has been open for {days_open} days. "
                f"INSTRUCTION: Tell customer ticket is being actively reviewed "
                f"and update within 24 hours. Do NOT say a new ticket was created."
            )
        else:
            ticket_info = (
                f"DUPLICATE COMPLAINT. Existing open ticket: {ticket_id}. "
                f"Ticket has been open for {days_open} days — OVERDUE. "
                f"INSTRUCTION: Sincerely apologise for the delay. Say ticket has been "
                f"open {days_open} days and is past resolution deadline. Say it has been "
                f"escalated to senior team. Do NOT give standard timelines. "
                f"Start response with a strong apology."
            )
    else:
        ticket_info = "No ticket was created for this query."

    # Pre-compute the exact duplicate response so the LLM cannot deviate from it
    if is_duplicate and ticket_id:
        if days_open == 0:
            duplicate_instruction = (
                f"This is a duplicate complaint — the user already has an open ticket.\n"
                f"Use EXACTLY this response (do not add or change anything):\n"
                f"\"Your ticket {ticket_id} was recently created. Our team will contact you "
                f"within 48 hours. Please allow us time to resolve this.\""
            )
        elif 1 <= days_open <= 3:
            duplicate_instruction = (
                f"This is a duplicate complaint — ticket has been open for {days_open} day(s).\n"
                f"Use EXACTLY this response (do not add or change anything):\n"
                f"\"Your ticket {ticket_id} is being actively reviewed. We understand your "
                f"concern and will update you within 24 hours.\""
            )
        else:
            duplicate_instruction = (
                f"This is a duplicate complaint — ticket has been open for {days_open} days (OVERDUE).\n"
                f"Use EXACTLY this response (do not add or change anything):\n"
                f"\"We sincerely apologise for the delay. Your ticket {ticket_id} has been open "
                f"for {days_open} days and is past our resolution deadline. This has been "
                f"escalated to our senior team.\""
            )
    else:
        duplicate_instruction = (
            "This is a new complaint. Provide a standard empathetic response referencing "
            "the policy and ticket ID."
        )

    fallback_prompt = f"""You are a helpful and empathetic customer support agent for an e-commerce store.
Write a resolution response for this customer complaint.

Session context: {context_str}
Recent messages:
{recent_msgs}

Issue type: {issue_type}
Issue details: {issue_details}
Order ID: {order_id if order_id else "Not provided"}
Actual order status in our system: {order_status}

Company policy for this issue:
{policy_text}

Ticket information: {ticket_info}

Duplicate complaint handling:
{duplicate_instruction}

Instructions:
- Be empathetic and apologetic where needed
- Reference the actual policy in your response
- IMPORTANT: Apply the policy strictly based on the actual order status above.
  If the order is "delivered" or "cancelled", do NOT say it can be cancelled or
  that you will proceed with cancellation — state clearly it is not eligible.
- If ticket_info says a ticket was created, mention the ticket ID
- If no ticket was created, do NOT mention any ticket or ticket ID
- Sign off as: Customer Support Team
- Keep response concise and helpful
- Do not use placeholder text like [Your Name]"""

    prompt_text, prompt_version = get_prompt(
        "draft_resolution",
        label    = settings.order_response_prompt_label,
        fallback = fallback_prompt
    )

    if "{{issue_type}}" in prompt_text:
        prompt_text = compile_prompt(
            prompt_text,
            history       = recent_msgs,
            issue_type    = issue_type,
            issue_details = issue_details,
            order_id      = order_id or "Not provided",
            policy_text   = policy_text,
            ticket_info   = ticket_info
        )

    llm      = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
    response = llm.invoke(prompt_text)
    usage    = extract_token_usage(response)

    if trace_id:
        create_generation(
            trace_id              = trace_id,
            name                  = "draft_resolution",
            model                 = settings.llm_model_name,
            prompt                = prompt_text,
            response              = response.content,
            usage                 = usage,
            parent_observation_id = parent_id,
            prompt_name           = "draft_resolution",
            prompt_version        = prompt_version
        )

    state["resolution"]    = response.content
    state["session_context"] = merge_context(ctx, {
        "topic":      "support_query",
        "issue_type": issue_type,
        "order_id":   order_id,
        "ticket_id":  ticket_id if ticket_created else None
    })
    return state


# ==========================================
# NODE 6 — format_response (pure code)
# ==========================================
def format_response(
    state: SupportAgentState
) -> SupportAgentState:
    logger.info("Support Agent node: format_response")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    resolution = state.get("resolution", "")
    ticket_id  = state.get("ticket_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "format_response",
        parent_observation_id = parent_id,
        input_data            = {"has_ticket": bool(ticket_id)}
    ) if trace_id else None

    if ticket_id and ticket_id not in resolution:
        response = (
            f"{resolution}\n\n"
            f"Your complaint reference: Ticket ID {ticket_id}"
        )
    else:
        response = resolution

    state["response"] = response

    if span:
        end_span(span, {"response_length": len(response)})

    return state


# ==========================================
# BUILD SUPPORT AGENT
# ==========================================
def build_support_agent():
    graph = StateGraph(SupportAgentState)

    graph.add_node("classify_issue",          classify_issue)
    graph.add_node("request_order_id",        request_order_id)
    graph.add_node("assess_severity",         assess_severity)
    graph.add_node("lookup_policy_node",      lookup_policy_node)
    graph.add_node("escalation_handler_node", escalation_handler_node)
    graph.add_node("draft_resolution",        draft_resolution)
    graph.add_node("format_response",         format_response)

    graph.set_entry_point("classify_issue")

    graph.add_conditional_edges(
        "classify_issue",
        route_after_classify,
        {
            "request_order_id": "request_order_id",
            "assess_severity":  "assess_severity"
        }
    )
    graph.add_edge("request_order_id", END)
    graph.add_edge("assess_severity", "lookup_policy_node")

    graph.add_conditional_edges(
        "lookup_policy_node",
        route_severity,
        {
            "escalation_handler_node": "escalation_handler_node",
            "draft_resolution":        "draft_resolution"
        }
    )

    graph.add_edge("escalation_handler_node", "draft_resolution")
    graph.add_edge("draft_resolution",        "format_response")
    graph.add_edge("format_response",         END)

    return graph.compile()


support_agent = build_support_agent()