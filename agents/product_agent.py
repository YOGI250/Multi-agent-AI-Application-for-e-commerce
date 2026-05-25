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
    response:                Optional[str]
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

    fallback_prompt = f"""Extract product search filters from this message.

Message: "{message}"

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

For keyword — extract the specific product type the user is looking for.
Examples: "robot vacuum" → keyword="vacuum", "USB cable" → keyword="cable", "TV" → keyword="TV"

Respond ONLY with a JSON object. No explanation. No markdown.
{{
  "category": "exact category name or null",
  "keyword": "specific product type for name search or null",
  "max_price": number or null,
  "min_price": number or null,
  "brand": "brand name or null",
  "min_rating": number or null
}}"""

    prompt_text, prompt_version = get_prompt(
        "extract_preferences",
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
    elif filters.get("keyword"):
        filters["keyword"] = None
    elif filters.get("max_price"):
        filters["max_price"] = filters["max_price"] * 1.5
    elif filters.get("min_rating"):
        filters["min_rating"] = None

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
        for i, p in enumerate(products[:20])
    ])

    fallback_prompt = f"""You are a product recommendation expert.
Rank these products by relevance to the user request.

User request: "{message}"

Products:
{product_list}

Return ONLY a JSON array of product indices, most relevant first.
Indices are 1-based. Example: [3, 1, 7, 2, 5]
Maximum 8 products. No explanation."""

    prompt_text, prompt_version = get_prompt(
        "rank_and_filter",
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
            ranked  = []
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
    max_price = filters.get("max_price") or 100000

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

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    products  = state.get("final_recommendations", [])
    message   = state.get("message", "")

    span = create_span(
        trace_id              = trace_id,
        name                  = "format_recommendations",
        parent_observation_id = parent_id,
        input_data            = {"products_count": len(products)}
    ) if trace_id else None

    if not products:
        state["response"] = (
            "I could not find any products matching your request. "
            "Try adjusting your budget or category."
        )
        if span:
            end_span(span, {"response_type": "no_products"})
        return state

    lines = [
        f"Here are my top {len(products)} recommendations "
        f"for \"{message}\":\n"
    ]

    for i, p in enumerate(products, 1):
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

        lines.append(f"   Score  : {p['score']}")
        lines.append("")

    state["response"] = "\n".join(lines)

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

    state["response"] = (
        "I could not find any products matching your request "
        "even after broadening the search. "
        "Try using a different category or removing specific "
        "brand or price requirements."
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