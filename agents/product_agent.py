# agents/product_agent.py

import logging
import json
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
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
from utils.memory import merge_context, format_context, format_recent_messages

logger = logging.getLogger(__name__)

AVAILABLE_CATEGORIES = [
    "Electronics",
    "Home and Kitchen",
    "Computers and Accessories",
    "OfficeProducts",
    "HomeImprovement",
    "MusicalInstruments",
    "Car and Motorbike",
    "Health and PersonalCare",
    "Toys and Games"
]


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
    partial_match:           Optional[bool]   # True when results exist but don't fully match request
    response:                Optional[str]
    session_context:         Optional[dict]
    langfuse_trace_id:       Optional[str]
    langfuse_parent_span_id: Optional[str]


# ==========================================
# NODE 1 — extract_preferences (LLM)
# ==========================================
def extract_preferences(state: ProductAgentState) -> ProductAgentState:
    logger.info("Product Agent node: extract_preferences")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    message   = state.get("message", "")
    filters   = state.get("filters")
    attempts  = state.get("broaden_attempts", 0)

    if filters and attempts > 0:
        logger.info(f"Using broadened filters: {filters}")
        return state

    ctx         = state.get("session_context") or {}
    history     = state.get("history", [])
    context_str = format_context(ctx)
    recent_msgs = format_recent_messages(history, n=2)

    fallback_prompt = f"""Extract product search filters from this message.

Session context: {context_str}
Recent messages:
{recent_msgs}

Message: "{message}"

IMPORTANT — If the message is a follow-up (e.g. "any cheaper options?", "any with better rating?",
"show me more", "different brand?"), look at recent messages to identify the product type that
was previously searched and carry it forward into the filters.


IMPORTANT — Only use these exact category names:
- Electronics
- Home and Kitchen
- Computers and Accessories
- OfficeProducts
- HomeImprovement
- MusicalInstruments
- Car and Motorbike
- Health and PersonalCare
- Toys and Games

Mapping guide:
- "USB cable", "laptop", "mobile", "keyboard", "charger", "tablet" → "Computers and Accessories"
- "headphones", "speaker", "TV", "camera", "smartwatch" → "Electronics"
- "fan", "mixer", "cookware", "vacuum", "robot vacuum", "AC", "purifier" → "Home and Kitchen"
- "pen", "notebook", "desk organizer" → "OfficeProducts"

For product_type — extract the specific product type using these exact values only:
mouse, keyboard, headphones, speaker, laptop, tablet, smartwatch, monitor, webcam,
router, cable, charger, usb_hub, pendrive, ssd, hard_disk, ram, memory_card,
printer, stand, mousepad, laptop_bag, phone_case, extension, fan, mixer, iron,
kettle, water_heater, room_heater, vacuum, air_purifier, water_purifier, camera,
pen, notebook, light, trimmer, microwave, other

Examples:
- "mouse", "mice", "pointing device", "wireless mouse" → product_type="mouse"
- "earphones", "earbuds", "headset", "TWS" → product_type="headphones"
- "pendrive", "flash drive", "USB drive" → product_type="pendrive"
- "geyser", "water heater" → product_type="water_heater"
- anything not in the list → product_type="other"

Respond ONLY with a JSON object. No explanation. No markdown.
{{
  "category": "exact category name or null",
  "product_type": "exact product type from the list above or null",
  "max_price": number or null,
  "min_price": number or null,
  "brand": "brand name or null",
  "min_rating": number or null
}}"""

    prompt_text, prompt_version = get_prompt(
        "extract_preferences",
        label    = settings.order_response_prompt_label,
        fallback = fallback_prompt
    )

    if "{{message}}" in prompt_text:
        prompt_text = compile_prompt(prompt_text, message=message)

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
            prompt_version        = prompt_version
        )

    try:
        text       = response.content.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        filters    = json.loads(json_match.group()) if json_match else {}
    except Exception:
        filters = {}

    state["filters"]          = filters
    state["original_filters"] = filters.copy() if filters else {}
    state["broaden_attempts"] = state.get("broaden_attempts", 0)

    logger.info(f"Extracted filters: {filters}")
    return state


# ==========================================
# NODE 2 — search_products_node (tool)
# ==========================================
def search_products_node(state: ProductAgentState) -> ProductAgentState:
    logger.info("Product Agent node: search_products_node")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    filters   = state.get("filters", {})

    span = create_span(
        trace_id              = trace_id,
        name                  = "search_products",
        parent_observation_id = parent_id,
        input_data            = {"filters": filters}
    ) if trace_id else None

    results = search_products_tool.invoke({"filters": filters})
    state["search_results"] = results

    if span:
        end_span(span, {"results_count": len(results)})

    logger.info(f"Search returned {len(results)} products")
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
    logger.info("Product Agent node: broaden_search")

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
    logger.info("Product Agent node: rank_and_filter")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    message   = state.get("message", "")
    products  = state.get("search_results", [])

    product_list = "\n".join([
        f"{i+1}. {p['name'][:60]} | "
        f"Price: ₹{p['price']} | "
        f"Rating: {p['rating']} | "
        f"Brand: {p['brand']}"
        for i, p in enumerate(products[:settings.product_search_candidates])
    ])

    fallback_prompt = f"""You are a product recommendation expert.
Rank these products by relevance to the user request.

User request: "{message}"

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
            message      = message,
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
            prompt_version        = prompt_version
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
    logger.info("Product Agent node: product_enrichment_node")

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
    logger.info("Product Agent node: format_recommendations")

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

    partial = state.get("partial_match", False)

    # Detect when every result is above the user's original budget
    # (takes priority over partial_match — a found product that's over budget
    # is more informative than a generic "no exact match" message)
    over_budget = (
        original_max
        and all(float(p.get("price", 0)) > original_max for p in products)
    )

    if over_budget and ptype_name:
        lines = [
            f"No {ptype_name} found under ₹{int(original_max):,}. "
            f"Here's the closest option available:\n"
        ]
    elif partial:
        lines = [
            f"I couldn't find an exact match for \"{message}\". "
            f"Here are the closest products we have:\n"
        ]
    else:
        shown = min(len(products), 3)
        lines = [
            f"Here are my top {shown} recommendation{'s' if shown > 1 else ''} "
            f"for \"{message}\":\n"
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
    state["session_context"] = merge_context(ctx, {
        "topic":          "product_query",
        "products_shown": [p["name"][:50] for p in products[:3]]
    })

    if span:
        end_span(span, {"response_type": "recommendations"})

    return state


# ==========================================
# NODE 7 — no_results_response (pure code)
# ==========================================
def no_results_response(
    state: ProductAgentState
) -> ProductAgentState:
    logger.info("Product Agent node: no_results_response")

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
    graph = StateGraph(ProductAgentState)

    graph.add_node("extract_preferences",     extract_preferences)
    graph.add_node("search_products_node",    search_products_node)
    graph.add_node("broaden_search",          broaden_search)
    graph.add_node("rank_and_filter",         rank_and_filter)
    graph.add_node("product_enrichment_node", product_enrichment_node)
    graph.add_node("format_recommendations",  format_recommendations)
    graph.add_node("no_results_response",     no_results_response)

    graph.set_entry_point("extract_preferences")

    graph.add_edge("extract_preferences", "search_products_node")

    graph.add_conditional_edges(
        "search_products_node",
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