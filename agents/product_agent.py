# agents/product_agent.py

import logging
import json
import re
from typing import TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
from langchain_groq import ChatGroq
from config.settings import settings
from subgraphs.product_enrichment import product_enrichment_subgraph
from tools.product_tools import search_products_tool
from langfuse_helpers.tracing import (
    create_span, end_span,
    create_generation,
    get_prompt, compile_prompt,
    extract_token_usage
)
from utils.memory import merge_context

logger = logging.getLogger(__name__)


# ==========================================
# STATE
# ==========================================
class ProductAgentState(TypedDict):
    message:                 str
    user_id:                 str
    session_id:              str
    history:                 list
    filters:                 Optional[dict]
    original_filters:        Optional[dict]
    broaden_attempts:        Optional[int]
    search_results:          Optional[list]
    ranked_products:         Optional[list]
    final_recommendations:   Optional[list]
    products:                Optional[list]   # all enriched products for frontend pagination
    response:                Optional[str]
    session_context:         Optional[dict]
    langfuse_trace_id:       Optional[str]
    langfuse_parent_span_id: Optional[str]
    messages:                Optional[List[Any]]


# ==========================================
# NODE 1 — extract_preferences (LLM)
# ==========================================
def extract_preferences(state: ProductAgentState) -> ProductAgentState:
    logger.info("Product Agent node: extract_preferences", extra={"node_name": "extract_preferences"})

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    message   = state.get("message", "")
    filters   = state.get("filters")
    attempts  = state.get("broaden_attempts", 0)

    if filters and attempts > 0:
        logger.info(f"Using broadened filters: {filters}")
        return state

    history     = state.get("history", [])
    # For extract_preferences we only need user messages from history
    # User messages contain the product search terms we need to carry forward
    user_msgs = [m for m in history if m.get("role") == "user"]
    if user_msgs:
        last_msgs = user_msgs[-4:]
        lines = []
        for i, m in enumerate(last_msgs):
            if i == len(last_msgs) - 1:
                lines.append(f"[MOST RECENT] {m['content'][:200]}")
            else:
                lines.append(f"[EARLIER] {m['content'][:200]}")
        recent_user_msgs = "\n".join(lines)
    else:
        recent_user_msgs = "No previous searches"
    

    fallback_prompt = f"""You are a product search filter extractor.

Recent user searches:
{recent_user_msgs}

Current message: "{message}"

RULE 1 — EXPLICIT PRODUCT (ignore history):
If current message names a specific product → extract it directly.
Examples: "mixer grinder under 2000" → mixer | "laptop bag under 1000" → laptop_bag
          "bluetooth speaker under 2000" → speaker | "steam iron under 1500" → iron

RULE 2 — VAGUE FOLLOW-UP (use most recent history):
If current message is vague with no product named → carry forward product_type
from the MOST RECENT search in history that has a product type.
Examples: "what about boAt" → use last product_type from history, change brand
          "under 2000" → use last product_type from history, change max_price
          "ones with calling feature" → use last product_type from history

PRODUCT TYPES (use exact values):
mouse, keyboard, headphones, speaker, laptop, tablet, smartwatch, monitor,
webcam, router, cable, charger, usb_hub, pendrive, ssd, hard_disk, ram,
memory_card, printer, stand, mousepad, laptop_bag, phone_case, extension,
fan, mixer, iron, kettle, water_heater, room_heater, vacuum, air_purifier,
water_purifier, camera, pen, notebook, light, trimmer, microwave, other

NORMALISE: smartwatches→smartwatch | mice→mouse | earphones/earbuds→headphones
air purifier→air_purifier | laptops→laptop | fans→fan | irons→iron

CATEGORIES: Electronics | Home and Kitchen | Computers and Accessories |
OfficeProducts | HomeImprovement | MusicalInstruments | Car and Motorbike |
Health and PersonalCare | Toys and Games

Respond ONLY with JSON:
{{
  "category":     "exact category or null",
  "product_type": "exact value from list or null",
  "max_price":    number or null,
  "min_price":    number or null,
  "brand":        "brand name or null",
  "min_rating":   number or null
}}"""

    prompt_text, prompt_version = get_prompt(
        "extract_preferences",
        label    = settings.order_response_prompt_label,
        fallback = fallback_prompt
    )

    if "{{message}}" in prompt_text:
        prompt_text = compile_prompt(
            prompt_text,
            message = message,
            history = recent_user_msgs
        )

    llm      = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
    response = llm.invoke(prompt_text)
    usage    = extract_token_usage(response)

    if trace_id:
        create_generation(
            trace_id              = trace_id,
            name                  = "extract_preferences",
            model                 = settings.llm_model_name,
            prompt                = prompt_text,
            response              = response.content,
            usage                 = usage,
            parent_observation_id = parent_id,
            prompt_name           = "extract_preferences",
            prompt_version        = prompt_version,
            agent_used            = "product_agent"
        )

    try:
        text       = response.content.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        filters    = json.loads(json_match.group()) if json_match else {}
    except Exception:
        filters = {}

    # Normalize product_type — LLM occasionally returns plural/variant forms
    if filters and filters.get("product_type"):
        ptype = filters["product_type"].lower().strip()
        normalization = {
            "smartwatches": "smartwatch",
            "smartwatche":  "smartwatch",
            "mice":         "mouse",
            "earphones":    "headphones",
            "earphone":     "headphones",
            "earbuds":      "headphones",
            "earbud":       "headphones",
            "headphone":    "headphones",
            "laptops":      "laptop",
            "keyboards":    "keyboard",
            "speakers":     "speaker",
            "tablets":      "tablet",
            "fans":         "fan",
            "mixers":       "mixer",
            "irons":        "iron",
        }
        filters["product_type"] = normalization.get(ptype, ptype)

    # If LLM returned no product_type, apply fallbacks in priority order:
    # 1. keyword match — for explicit product messages the LLM missed
    # 2. session context — for genuinely vague follow-ups ("under 2000", "what about X")
    if not filters.get("product_type"):
        KEYWORD_MAP = [
            ("smart watch", "smartwatch"), ("smartwatch", "smartwatch"),
            ("laptop bag", "laptop_bag"), ("air purifier", "air_purifier"),
            ("water purifier", "water_purifier"), ("room heater", "room_heater"),
            ("water heater", "water_heater"), ("phone case", "phone_case"),
            ("hard disk", "hard_disk"), ("memory card", "memory_card"),
            ("usb hub", "usb_hub"), ("pen drive", "pendrive"),
            ("headphone", "headphones"), ("earphone", "headphones"), ("earbud", "headphones"),
            ("speaker", "speaker"), ("keyboard", "keyboard"), ("monitor", "monitor"),
            ("tablet", "tablet"), ("webcam", "webcam"), ("router", "router"),
            ("charger", "charger"), ("pendrive", "pendrive"),
            ("grinder", "mixer"), ("mixer", "mixer"), ("vacuum", "vacuum"),
            ("kettle", "kettle"), ("microwave", "microwave"), ("trimmer", "trimmer"),
            ("camera", "camera"), ("printer", "printer"), ("laptop", "laptop"),
            ("mouse", "mouse"), ("iron", "iron"), ("fan", "fan"), ("ssd", "ssd"),
        ]
        msg_lower = message.lower()
        keyword_ptype = next(
            (ptype for kw, ptype in KEYWORD_MAP if kw in msg_lower),
            None
        )
        if keyword_ptype:
            filters["product_type"] = keyword_ptype
        else:
            last_ptype = (state.get("session_context") or {}).get("last_product_type")
            if last_ptype:
                filters["product_type"] = last_ptype

    state["filters"]          = filters
    state["original_filters"] = filters.copy() if filters else {}
    state["broaden_attempts"] = state.get("broaden_attempts", 0)

    logger.info(f"Extracted filters: {filters}")
    return state


# ==========================================
# NODE 2a — search_products_node (prepare tool call for ToolNode)
# ==========================================
def search_products_node(state: ProductAgentState) -> ProductAgentState:
    logger.info("Product Agent node: search_products_node", extra={"node_name": "search_products_node"})

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    filters   = state.get("filters") or {}

    span = create_span(
        trace_id              = trace_id,
        name                  = "search_products_node",
        parent_observation_id = parent_id,
        input_data            = {"filters": filters}
    ) if trace_id else None

    tool_call = {
        "name": "search_products_tool",
        "args": {"filters": filters},
        "id":   "product_search_1",
        "type": "tool_call",
    }
    state["messages"] = [AIMessage(content="", tool_calls=[tool_call])]

    if span:
        end_span(span, {"tool": "search_products_tool", "filters": filters})

    return state


# ==========================================
# NODE 2b — process_search_result (reads ToolNode output)
# ==========================================
def process_search_result(state: ProductAgentState) -> ProductAgentState:
    logger.info("Product Agent node: process_search_result", extra={"node_name": "process_search_result"})

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    messages  = state.get("messages") or []

    tool_msg = next((m for m in messages if isinstance(m, ToolMessage)), None)
    results  = []
    if tool_msg:
        try:
            results = json.loads(tool_msg.content) or []
        except (json.JSONDecodeError, TypeError):
            results = []

    state["search_results"] = results if isinstance(results, list) else []

    span = create_span(
        trace_id              = trace_id,
        name                  = "search_products",
        parent_observation_id = parent_id,
        input_data            = {"filters": state.get("filters", {})}
    ) if trace_id else None
    if span:
        end_span(span, {"results_count": len(state["search_results"])})

    logger.info(f"Search returned {len(state['search_results'])} products")
    return state


# ==========================================
# EDGE — results_found?
# ==========================================
def route_results_found(state: ProductAgentState) -> str:
    results  = state.get("search_results", [])
    attempts = state.get("broaden_attempts", 0)

    if results:
        return "rank_and_filter"
    if attempts >= 1:
        return "no_results_response"
    return "broaden_search"


# ==========================================
# NODE 3 — broaden_search (pure code)
# ==========================================
def broaden_search(state: ProductAgentState) -> ProductAgentState:
    logger.info("Product Agent node: broaden_search", extra={"node_name": "broaden_search"})

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    filters   = state.get("filters", {}).copy()
    attempts  = state.get("broaden_attempts", 0)

    span = create_span(
        trace_id              = trace_id,
        name                  = "broaden_search",
        parent_observation_id = parent_id,
        input_data            = {"filters": filters, "attempt": attempts}
    ) if trace_id else None

    if filters.get("brand"):
        filters["brand"] = None
    elif filters.get("max_price"):
        filters["max_price"] = filters["max_price"] * settings.product_price_broaden_factor
    elif filters.get("min_rating"):
        filters["min_rating"] = None
    # product_type is never removed — it defines what the user is looking for

    state["filters"]          = filters
    state["broaden_attempts"] = attempts + 1

    if span:
        end_span(span, {"broadened_filters": filters})

    return state


# ==========================================
# NODE 4 — rank_and_filter (LLM)
# ==========================================
def rank_and_filter(state: ProductAgentState) -> ProductAgentState:
    logger.info("Product Agent node: rank_and_filter", extra={"node_name": "rank_and_filter"})

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    message   = state.get("message", "")
    products  = state.get("search_results", [])

    # Build a richer request string so rank_and_filter LLM understands context
    # for vague follow-ups like "under 2000" that carry no product name themselves
    filters      = state.get("filters") or {}
    product_type = filters.get("product_type", "")
    if product_type and product_type.lower() not in message.lower():
        rich_request = f"{product_type.replace('_', ' ')} — {message}"
    else:
        rich_request = message

    product_list = "\n".join([
        f"{i+1}. {p['name'][:60]} | "
        f"Price: ₹{p['price']} | "
        f"Rating: {p['rating']} | "
        f"Brand: {p['brand']}"
        for i, p in enumerate(products[:settings.product_search_candidates])
    ])

    fallback_prompt = f"""You are a product recommendation expert.
Rank these products by relevance to the user request.

User request: "{rich_request}"

Products:
{product_list}

Rules:
- Include any product that is a direct or close match.
- Include partial matches (e.g. a regular laptop when asked for a gaming laptop).
- Return [] ONLY if the products are entirely unrelated to the request
  (e.g. user wants furniture but the list contains only electronics).

Return ONLY a JSON array of product indices, most relevant first.
Indices are 1-based. Example: [3, 1, 7, 2, 5]
Maximum {settings.product_recommendation_count} products. No explanation."""

    prompt_text, prompt_version = get_prompt(
        "rank_and_filter",
        label    = settings.order_response_prompt_label,
        fallback = fallback_prompt
    )

    if "{{message}}" in prompt_text:
        prompt_text = compile_prompt(
            prompt_text,
            message      = rich_request,
            product_list = product_list
        )

    llm      = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model_name)
    response = llm.invoke(prompt_text)
    usage    = extract_token_usage(response)

    if trace_id:
        create_generation(
            trace_id              = trace_id,
            name                  = "rank_and_filter",
            model                 = settings.llm_model_name,
            prompt                = prompt_text,
            response              = response.content,
            usage                 = usage,
            parent_observation_id = parent_id,
            prompt_name           = "rank_and_filter",
            prompt_version        = prompt_version,
            agent_used            = "product_agent"
        )

    try:
        text  = response.content.strip()
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            indices = json.loads(match.group())
            if not indices:
                # LLM judged all products as entirely unrelated — honour that
                state["ranked_products"] = []
            else:
                ranked = []
                for idx in indices:
                    if 1 <= idx <= len(products):
                        ranked.append(products[idx - 1])
                ranked_ids = {p['product_id'] for p in ranked}
                for p in products:
                    if p['product_id'] not in ranked_ids:
                        ranked.append(p)
                state["ranked_products"] = ranked
        else:
            state["ranked_products"] = products
    except Exception:
        state["ranked_products"] = products

    logger.info(f"Ranked {len(state['ranked_products'])} products")
    return state


# ==========================================
# NODE 5 — product_enrichment_node (subgraph)
# ==========================================
def product_enrichment_node(
    state: ProductAgentState
) -> ProductAgentState:
    logger.info("Product Agent node: product_enrichment_node", extra={"node_name": "product_enrichment_node"})

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    filters   = state.get("filters", {})
    max_price = filters.get("max_price") or settings.product_default_max_price

    span = create_span(
        trace_id              = trace_id,
        name                  = "product_enrichment_subgraph",
        parent_observation_id = parent_id,
        input_data            = {
            "products_count": len(state.get("ranked_products", [])),
            "max_price":      max_price
        }
    ) if trace_id else None

    result = product_enrichment_subgraph.invoke({
        "ranked_products":         state.get("ranked_products", []),
        "max_price":               max_price,
        "langfuse_trace_id":       trace_id,
        "langfuse_parent_span_id": span.id if span else parent_id
    })

    state["final_recommendations"] = result.get("final_recommendations", [])
    state["products"]              = state["final_recommendations"]

    if span:
        end_span(span, {
            "recommendations_count": len(state["final_recommendations"])
        })

    return state


# ==========================================
# NODE 6 — format_recommendations (pure code)
# ==========================================
def format_recommendations(
    state: ProductAgentState
) -> ProductAgentState:
    logger.info("Product Agent node: format_recommendations", extra={"node_name": "format_recommendations"})

    trace_id         = state.get("langfuse_trace_id")
    parent_id        = state.get("langfuse_parent_span_id")
    products         = state.get("final_recommendations", [])
    message          = state.get("message", "")
    original_filters = state.get("original_filters", {}) or {}
    original_max     = original_filters.get("max_price")
    ptype_name       = (original_filters.get("product_type") or "").replace("_", " ")

    span = create_span(
        trace_id              = trace_id,
        name                  = "format_recommendations",
        parent_observation_id = parent_id,
        input_data            = {"products_count": len(products)}
    ) if trace_id else None

    state["products"] = products

    if not products:
        state["response"] = (
            f"I couldn't find any products matching \"{message}\".\n\n"
            f"Here's what we currently have available:\n\n"
            f"  -Computers & Accessories\n"
            f"    laptops, keyboards, mice, cables, chargers, USB hubs, webcams\n\n"
            f"  -Electronics\n"
            f"    headphones, speakers, smartwatches, cameras\n\n"
            f"  -Home & Kitchen\n"
            f"    fans, mixers, kettles, irons, geysers, vacuum cleaners, air purifiers\n\n"
            f"  -Office Products\n"
            f"    pens, notebooks, desk organizers\n\n"
            f"  -Car & Motorbike\n"
            f"    car accessories\n\n"
            f"Try searching within one of these categories."
        )
        if span:
            end_span(span, {"response_type": "no_products"})
        return state

    ctx = state.get("session_context") or {}

    over_budget = (
        original_max
        and all(float(p.get("price", 0)) > original_max for p in products)
    )

    if over_budget and ptype_name:
        lines = [
            f"No {ptype_name} found under ₹{int(original_max):,}. "
            f"Here's the closest option available:\n"
        ]
    else:
        lines = [
            f"Here are my top recommendations for \"{message}\":\n"
        ]

    # Format first 3 in text; the rest surface via cards in the frontend
    for i, p in enumerate(products[:3], 1):
        lines.append(f"{i}. {p['name'][:70]}")
        lines.append(f"   Price  : ₹{p['price']}")
        lines.append(
            f"   Rating : {p['rating']} ⭐ "
            f"({p['rating_count']:,} reviews)"
        )
        lines.append(f"   Brand  : {p['brand']}")

        features = p.get("features", [])
        if features:
            lines.append("   Features:")
            for feat in features[:3]:
                lines.append(f"   • {str(feat)[:80]}")

        lines.append("")

    if len(products) > 3:
        extra = len(products) - 3
        lines.append(
            f"...and {extra} more product{'s' if extra > 1 else ''} available."
        )

    state["response"] = "\n".join(lines)
    ctx_update = {
        "topic":             "product_query",
        "products_shown":    [p["name"][:50] for p in products[:3]],
    }
    if original_filters.get("product_type"):
        ctx_update["last_product_type"] = original_filters["product_type"]
    if original_filters.get("max_price"):
        ctx_update["last_max_price"] = original_filters["max_price"]
    state["session_context"] = merge_context(ctx, ctx_update)

    if span:
        end_span(span, {"response_type": "recommendations"})

    return state


# ==========================================
# NODE 7 — no_results_response (pure code)
# ==========================================
def no_results_response(
    state: ProductAgentState
) -> ProductAgentState:
    logger.info("Product Agent node: no_results_response", extra={"node_name": "no_results_response"})

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "no_results_response",
        parent_observation_id = parent_id,
        input_data            = {"filters": state.get("filters", {})}
    ) if trace_id else None

    original_query = state.get("message", "your request")
    state["response"] = (
        f"I couldn't find any products matching \"{original_query}\".\n\n"
        f"Here's what we currently have available:\n\n"
        f"  -Computers & Accessories\n"
        f"    laptops, keyboards, mice, cables, chargers, USB hubs, webcams\n\n"
        f"  -Electronics\n"
        f"    headphones, speakers, smartwatches, cameras\n\n"
        f"  -Home & Kitchen\n"
        f"    fans, mixers, kettles, irons, geysers, vacuum cleaners, air purifiers\n\n"
        f"  -Office Products\n"
        f"    pens, notebooks, desk organizers\n\n"
        f"  -Car & Motorbike\n"
        f"    car accessories\n\n"
        f"Try searching within one of these categories."
    )

    if span:
        end_span(span, {"response_type": "no_results"})

    return state


# ==========================================
# BUILD PRODUCT AGENT
# ==========================================
def build_product_agent():
    _tool_node = ToolNode([search_products_tool])

    def product_tool_node_fn(state: ProductAgentState) -> dict:
        logger.info("Product Agent node: product_tool_node", extra={"node_name": "product_tool_node"})
        trace_id  = state.get("langfuse_trace_id")
        parent_id = state.get("langfuse_parent_span_id")
        msgs      = state.get("messages", [])
        last_ai   = next((m for m in reversed(msgs) if hasattr(m, "tool_calls") and m.tool_calls), None)
        tool_name = last_ai.tool_calls[0].get("name", "unknown") if last_ai else "unknown"
        tool_args = last_ai.tool_calls[0].get("args", {}) if last_ai else {}

        span = create_span(
            trace_id              = trace_id,
            name                  = "product_tool_node",
            parent_observation_id = parent_id,
            input_data            = {"tool": tool_name, "args": tool_args}
        ) if trace_id else None

        result = _tool_node.invoke(state)

        if span:
            tool_msgs = result.get("messages", [])
            output    = tool_msgs[0].content[:200] if tool_msgs else None
            end_span(span, {"tool_output": output, "tool": tool_name})

        return result

    graph = StateGraph(ProductAgentState)

    graph.add_node("extract_preferences",     extract_preferences)
    graph.add_node("search_products_node",    search_products_node)
    graph.add_node("product_tool_node",       product_tool_node_fn)
    graph.add_node("process_search_result",   process_search_result)
    graph.add_node("broaden_search",          broaden_search)
    graph.add_node("rank_and_filter",         rank_and_filter)
    graph.add_node("product_enrichment_node", product_enrichment_node)
    graph.add_node("format_recommendations",  format_recommendations)
    graph.add_node("no_results_response",     no_results_response)

    graph.set_entry_point("extract_preferences")

    graph.add_edge("extract_preferences",   "search_products_node")
    graph.add_edge("search_products_node",  "product_tool_node")
    graph.add_edge("product_tool_node",     "process_search_result")

    graph.add_conditional_edges(
        "process_search_result",
        route_results_found,
        {
            "rank_and_filter":     "rank_and_filter",
            "broaden_search":      "broaden_search",
            "no_results_response": "no_results_response"
        }
    )

    graph.add_edge("broaden_search",          "search_products_node")
    graph.add_edge("rank_and_filter",         "product_enrichment_node")
    graph.add_edge("product_enrichment_node", "format_recommendations")
    graph.add_edge("format_recommendations",  END)
    graph.add_edge("no_results_response",     END)

    return graph.compile()


product_agent = build_product_agent()
