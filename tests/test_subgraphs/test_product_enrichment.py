# tests/test_subgraphs/test_product_enrichment.py

import pytest
from unittest.mock import patch


def _state(**kwargs):
    base = {
        "ranked_products": [
            {
                "product_id":   "P001",
                "name":         "Logitech Wireless Mouse",
                "price":        999.0,
                "brand":        "Logitech",
                "rating":       4.5,
                "rating_count": 500,
                "in_stock":     True
            }
        ],
        "max_price":              2000.0,
        "reviews_data":           None,
        "specs_data":             None,
        "final_recommendations":  None,
        "langfuse_trace_id":      None,
        "langfuse_parent_span_id": None,
    }
    base.update(kwargs)
    return base


class TestFetchReviewsNode:

    def test_builds_reviews_from_products(self):
        from subgraphs.product_enrichment import fetch_reviews_node
        result = fetch_reviews_node(_state())
        assert "P001" in result["reviews_data"]
        assert result["reviews_data"]["P001"]["rating"] == 4.5
        assert result["reviews_data"]["P001"]["rating_count"] == 500

    def test_empty_products_returns_empty_dict(self):
        from subgraphs.product_enrichment import fetch_reviews_node
        result = fetch_reviews_node(_state(ranked_products=[]))
        assert result["reviews_data"] == {}

    def test_missing_rating_defaults_to_zero(self):
        from subgraphs.product_enrichment import fetch_reviews_node
        products = [{"product_id": "P002", "name": "No Rating Product", "in_stock": True}]
        result = fetch_reviews_node(_state(ranked_products=products))
        assert result["reviews_data"]["P002"]["rating"] == 0


class TestFetchSpecsNode:

    def test_calls_fetch_specs_tool(self):
        from subgraphs.product_enrichment import fetch_specs_node
        mock_specs = {"P001": {"features": ["wireless", "compact"], "description": "A mouse"}}
        with patch("subgraphs.product_enrichment.fetch_specs_tool") as mock_tool:
            mock_tool.invoke.return_value = mock_specs
            result = fetch_specs_node(_state())
        assert "P001" in result["specs_data"]
        assert "wireless" in result["specs_data"]["P001"]["features"]

    def test_empty_products_skips_tool_call(self):
        from subgraphs.product_enrichment import fetch_specs_node
        result = fetch_specs_node(_state(ranked_products=[]))
        assert result["specs_data"] == {}

    def test_handles_none_response_from_tool(self):
        from subgraphs.product_enrichment import fetch_specs_node
        with patch("subgraphs.product_enrichment.fetch_specs_tool") as mock_tool:
            mock_tool.invoke.return_value = None
            result = fetch_specs_node(_state())
        assert result["specs_data"] == {}


class TestComputeScore:

    def test_scores_single_product(self):
        from subgraphs.product_enrichment import compute_score
        state = _state(
            reviews_data={"P001": {"rating": 4.5, "rating_count": 500}},
            specs_data={"P001": {"features": ["wireless"], "description": "A mouse"}}
        )
        result = compute_score(state)
        assert len(result["final_recommendations"]) == 1
        assert "score" in result["final_recommendations"][0]
        assert result["final_recommendations"][0]["score"] > 0

    def test_no_budget_treats_price_as_neutral(self):
        from subgraphs.product_enrichment import compute_score
        state = _state(
            max_price=None,
            reviews_data={"P001": {"rating": 4.0, "rating_count": 100}},
            specs_data={}
        )
        result = compute_score(state)
        assert len(result["final_recommendations"]) == 1

    def test_over_budget_product_scores_lower(self):
        from subgraphs.product_enrichment import compute_score
        # price 1800 > max_price 1000 → price_fit=0 → lower score
        products = [
            {"product_id": "P001", "name": "Cheap", "price": 500.0, "in_stock": True},
            {"product_id": "P002", "name": "Expensive", "price": 1800.0, "in_stock": True},
        ]
        reviews = {
            "P001": {"rating": 4.0, "rating_count": 100},
            "P002": {"rating": 4.0, "rating_count": 100},
        }
        state = _state(ranked_products=products, max_price=1000.0,
                       reviews_data=reviews, specs_data={})
        result = compute_score(state)
        scores = {r["product_id"]: r["score"] for r in result["final_recommendations"]}
        assert scores["P001"] > scores["P002"]

    def test_out_of_stock_products_excluded(self):
        from subgraphs.product_enrichment import compute_score
        products = [
            {"product_id": "P001", "name": "In Stock",  "price": 999.0, "in_stock": True},
            {"product_id": "P002", "name": "Out Stock", "price": 999.0, "in_stock": False},
        ]
        reviews = {
            "P001": {"rating": 4.5, "rating_count": 100},
            "P002": {"rating": 4.5, "rating_count": 100},
        }
        state = _state(ranked_products=products, reviews_data=reviews, specs_data={})
        result = compute_score(state)
        ids = [r["product_id"] for r in result["final_recommendations"]]
        assert "P001" in ids
        assert "P002" not in ids


class TestProductEnrichmentSubgraph:

    def test_subgraph_compiles(self):
        from subgraphs.product_enrichment import product_enrichment_subgraph
        assert product_enrichment_subgraph is not None

    def test_full_flow(self):
        from subgraphs.product_enrichment import product_enrichment_subgraph
        mock_specs = {"P001": {"features": ["wireless"], "description": "Mouse"}}
        with patch("subgraphs.product_enrichment.fetch_specs_tool") as mock_tool:
            mock_tool.invoke.return_value = mock_specs
            result = product_enrichment_subgraph.invoke(_state())
        assert result["final_recommendations"] is not None
        assert len(result["final_recommendations"]) >= 1
