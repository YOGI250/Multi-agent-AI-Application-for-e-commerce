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
    policy_text:             Optional[str]
    ticket_id:               Optional[str]
    ticket_created:          Optional[bool]
    priority:                Optional[str]
    resolution:              Optional[str]
    response:                Optional[str]
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

    history_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-4:]
    ]) if history else "No previous conversation"

    fallback_prompt = f"""You are a customer support classifier.
Classify the customer complaint and extract key details.

Previous conversation:
{history_text}

Customer message: "{message}"

Issue types available:
- damaged_product: product arrived broken or damaged
- wrong_item: received a different product than ordered
- refund: customer wants money back
- cancellation: customer wants to cancel an order
- general_query: general question, not a complaint

Respond ONLY with a JSON object. No explanation.
{{
  "issue_type": "one of the issue types above",
  "order_id": "order ID only if explicitly mentioned in the CURRENT customer message, not from history. If the history shows an order was already confirmed as not found, set to null. Otherwise null.",
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
            history = history_text
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

    state["issue_type"]    = result.get("issue_type", "general_query")
    state["order_id"]      = result.get("order_id")
    state["issue_details"] = result.get("details", message)
    state["ticket_id"]     = None
    state["ticket_created"] = False
    state["priority"]      = None

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

    # Fetch order value if order_id is available
    order_value = 0
    if order_id:
        try:
            from database.connection import SessionLocal
            from database.models import Order
            db = SessionLocal()
            order = db.query(Order).filter(Order.order_id == order_id).first()
            if order:
                order_value = float(order.order_value or 0)
            db.close()
        except Exception:
            order_value = 0

    # Orders above ₹10,000 get HIGH severity
    # Orders below ₹10,000 get MEDIUM for the same issue
    HIGH_VALUE_THRESHOLD = 10000

    if issue_type in ["damaged_product", "wrong_item", "refund"]:
        if order_value >= HIGH_VALUE_THRESHOLD:
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
    state["priority"]       = result.get("ticket_priority")

    if span:
        end_span(span, {
            "ticket_id":      state["ticket_id"],
            "ticket_created": state["ticket_created"],
            "priority":       state["priority"]
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
    ticket_id      = state.get("ticket_id")
    ticket_created = state.get("ticket_created", False)
    priority       = state.get("priority")
    history        = state.get("history", [])

    history_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-4:]
    ]) if history else "No previous conversation"

    ticket_info = (
        f"A support ticket has been created. "
        f"Ticket ID: {ticket_id} | Priority: {priority}"
    ) if ticket_created and ticket_id else "No ticket was created for this query."

    fallback_prompt = f"""You are a helpful and empathetic customer support agent for an e-commerce store.
Write a resolution response for this customer complaint.

Previous conversation:
{history_text}

Issue type: {issue_type}
Issue details: {issue_details}
Order ID: {order_id if order_id else "Not provided"}

Company policy for this issue:
{policy_text}

Ticket information: {ticket_info}

Instructions:
- Be empathetic and apologetic where needed
- Reference the actual policy in your response
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
            history       = history_text,
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

    state["resolution"] = response.content
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
    graph.add_node("assess_severity",         assess_severity)
    graph.add_node("lookup_policy_node",      lookup_policy_node)
    graph.add_node("escalation_handler_node", escalation_handler_node)
    graph.add_node("draft_resolution",        draft_resolution)
    graph.add_node("format_response",         format_response)

    graph.set_entry_point("classify_issue")

    graph.add_edge("classify_issue",  "assess_severity")
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