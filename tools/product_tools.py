# tools/product_tools.py

import logging
from langchain_core.tools import tool
from services.mock_product_api import search_products, get_specs
from utils.langfuse_context import get_trace_context
from langfuse_helpers.tracing import create_span, end_span

logger = logging.getLogger(__name__)


@tool
def search_products_tool(filters: dict) -> list:
    """
    Searches the product catalogue using structured filters.
    Filters can include category, max_price, min_price,
    brand, min_rating, and in_stock.
    Returns a list of matching products.
    Returns empty list if no products match.
    """
    logger.info(f"Tool called: search_products_tool with filters={filters}")
    trace_id, parent_span_id = get_trace_context()
    span = (
        create_span(
            trace_id, "tool:search_products", parent_observation_id=parent_span_id, input_data={"filters": filters}
        )
        if trace_id
        else None
    )

    results = search_products(filters)

    logger.info(f"search_products_tool found {len(results)} products")
    if span:
        end_span(span, {"results_count": len(results)})
    return results


@tool
def fetch_specs_tool(product_ids: list) -> dict:
    """
    Fetches description and features for a list of product IDs.
    Used by fetch_specs node inside product_enrichment subgraph.
    Returns dict of product_id to specs data.
    Used by format_recommendations to show feature lists.
    """
    logger.info(f"Tool called: fetch_specs_tool for {len(product_ids)} products")
    trace_id, parent_span_id = get_trace_context()
    span = (
        create_span(
            trace_id, "tool:fetch_specs", parent_observation_id=parent_span_id, input_data={"product_ids": product_ids}
        )
        if trace_id
        else None
    )

    results = get_specs(product_ids)

    logger.info(f"fetch_specs_tool returned specs for {len(results)} products")
    if span:
        end_span(span, {"specs_count": len(results)})
    return results
