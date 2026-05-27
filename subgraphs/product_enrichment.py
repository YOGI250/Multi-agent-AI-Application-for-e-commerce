# subgraphs/product_enrichment.py

import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from tools.product_tools import fetch_ratings_tool, fetch_specs_tool
from langfuse_helpers.tracing import create_span, end_span

logger = logging.getLogger(__name__)


class ProductEnrichmentState(TypedDict):
    ranked_products:         list
    max_price:               Optional[float]
    reviews_data:            Optional[dict]
    specs_data:              Optional[dict]
    final_recommendations:   Optional[list]
    langfuse_trace_id:       Optional[str]
    langfuse_parent_span_id: Optional[str]


def fetch_reviews_node(
    state: ProductEnrichmentState
) -> ProductEnrichmentState:
    logger.info("Subgraph node: fetch_reviews")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    products  = state.get("ranked_products", [])

    if not products:
        state["reviews_data"] = {}
        return state

    span = create_span(
        trace_id              = trace_id,
        name                  = "fetch_reviews",
        parent_observation_id = parent_id,
        input_data            = {"products_count": len(products)}
    ) if trace_id else None

    # rating and rating_count are already in search_results
    # from the earlier search_products query
    # no need to hit the database again for the same data
    reviews = {
        p["product_id"]: {
            "rating":       p.get("rating", 0),
            "rating_count": p.get("rating_count", 0)
        }
        for p in products
    }
    state["reviews_data"] = reviews

    if span:
        end_span(span, {"reviews_fetched": len(reviews)})

    return state


def fetch_specs_node(
    state: ProductEnrichmentState
) -> ProductEnrichmentState:
    logger.info("Subgraph node: fetch_specs")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    products  = state.get("ranked_products", [])

    if not products:
        state["specs_data"] = {}
        return state

    product_ids = [p["product_id"] for p in products]

    span = create_span(
        trace_id              = trace_id,
        name                  = "fetch_specs",
        parent_observation_id = parent_id,
        input_data            = {"product_ids_count": len(product_ids)}
    ) if trace_id else None

    specs = fetch_specs_tool.invoke({"product_ids": product_ids})
    state["specs_data"] = specs or {}

    if span:
        end_span(span, {"specs_fetched": len(specs or {})})

    return state


def compute_score(
    state: ProductEnrichmentState
) -> ProductEnrichmentState:
    logger.info("Subgraph node: compute_score")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    products  = state.get("ranked_products", [])
    reviews   = state.get("reviews_data", {})
    specs     = state.get("specs_data", {})
    max_price = state.get("max_price")

    span = create_span(
        trace_id              = trace_id,
        name                  = "compute_score",
        parent_observation_id = parent_id,
        input_data            = {"products_count": len(products)}
    ) if trace_id else None

    total    = len(products)
    enriched = []

    for position, product in enumerate(products):
        pid          = product["product_id"]
        review       = reviews.get(pid, {})
        spec         = specs.get(pid, {})
        rating       = review.get("rating", 0)
        rating_count = review.get("rating_count", 0)
        price        = float(product.get("price", 0))

        # Rating quality (40%)
        rating_score = (rating / 5) * 0.40

        # Review trust (30%)
        trust_score = min(rating_count / 1000, 1.0) * 0.30

        # Price fit (30%)
        # Only applies when user gave a budget
        # No budget = neutral score
        if max_price and max_price > 0 and price > 0:
            price_ratio = price / max_price
            if price_ratio <= 0.7:
                price_fit = 1.0      # well under budget
            elif price_ratio <= 0.9:
                price_fit = 0.75     # comfortable fit
            elif price_ratio <= 1.0:
                price_fit = 0.50     # right at the limit
            else:
                price_fit = 0.0      # over budget
        else:
            price_fit = 1.0          # no budget = neutral

        price_score = price_fit * 0.30

        # LLM rank bonus (10%)
        # Products ranked higher by LLM get a positional bonus
        # This preserves LLM relevance judgment
        if total > 1:
            position_bonus = ((total - position) / total) * 0.10
        else:
            position_bonus = 0.10

        final_score = rating_score + trust_score + price_score + position_bonus

        enriched.append({
            **product,
            "features":    spec.get("features", []),
            "description": spec.get("description", ""),
            "score":       round(final_score, 4)
        })

    # Sort by final score — LLM ranking preserved via position_bonus
    # Return however many in-stock products exist (up to 3), not a fixed 3
    in_stock = [p for p in enriched if p.get("in_stock", True)]
    top_n    = sorted(in_stock, key=lambda x: x["score"], reverse=True)[:9]
    state["final_recommendations"] = top_n

    if span:
        end_span(span, {"top_recommendations": len(top_n)})

    return state


def build_product_enrichment_subgraph():
    graph = StateGraph(ProductEnrichmentState)

    graph.add_node("fetch_reviews", fetch_reviews_node)
    graph.add_node("fetch_specs",   fetch_specs_node)
    graph.add_node("compute_score", compute_score)

    graph.set_entry_point("fetch_reviews")
    graph.add_edge("fetch_reviews", "fetch_specs")
    graph.add_edge("fetch_specs",   "compute_score")
    graph.add_edge("compute_score", END)

    return graph.compile()


product_enrichment_subgraph = build_product_enrichment_subgraph()