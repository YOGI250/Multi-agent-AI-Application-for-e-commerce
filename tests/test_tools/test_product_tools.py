# tests/test_tools/test_product_tools.py

import pytest
from unittest.mock import patch, MagicMock
from utils.langfuse_context import set_trace_context


@pytest.fixture
def mock_products():
    return [
        {
            "product_id": "P001",
            "name": "Logitech Wireless Mouse",
            "category": "Electronics",
            "price": 999.0,
            "brand": "Logitech",
            "rating": 4.5,
            "in_stock": True,
        },
        {
            "product_id": "P002",
            "name": "Dell Laptop",
            "category": "Computers",
            "price": 45000.0,
            "brand": "Dell",
            "rating": 4.2,
            "in_stock": True,
        },
    ]


class TestSearchProductsTool:

    def test_search_returns_list(self, mock_products):
        with patch("tools.product_tools.search_products", return_value=mock_products):
            from tools.product_tools import search_products_tool

            result = search_products_tool.invoke(
                {"filters": {"category": "Electronics", "max_price": 50000, "keyword": "mouse"}}
            )
        assert isinstance(result, list)

    def test_search_returns_empty_for_no_match(self):
        with patch("tools.product_tools.search_products", return_value=[]):
            from tools.product_tools import search_products_tool

            result = search_products_tool.invoke({"filters": {"category": "Unknown"}})
        assert result == []


class TestFetchSpecsTool:

    def test_fetch_specs_returns_dict(self):
        mock_specs = {"P001": {"features": ["wireless", "compact"]}}
        with patch("tools.product_tools.get_specs", return_value=mock_specs):
            from tools.product_tools import fetch_specs_tool

            result = fetch_specs_tool.invoke({"product_ids": ["P001"]})
        assert isinstance(result, dict)


class TestProductToolsWithSpan:

    def setup_method(self):
        set_trace_context("test-trace-id", "test-span-id")

    def teardown_method(self):
        set_trace_context(None, None)

    def test_search_products_with_span(self, mock_products):
        mock_span = MagicMock()
        with patch("tools.product_tools.search_products", return_value=mock_products), patch(
            "tools.product_tools.create_span", return_value=mock_span
        ), patch("tools.product_tools.end_span") as mock_end:
            from tools.product_tools import search_products_tool

            result = search_products_tool.invoke({"filters": {"category": "Electronics"}})

        assert isinstance(result, list)
        mock_end.assert_called_once()

    def test_fetch_specs_with_span(self):
        mock_specs = {"P001": {"features": ["wireless"]}}
        mock_span = MagicMock()
        with patch("tools.product_tools.get_specs", return_value=mock_specs), patch(
            "tools.product_tools.create_span", return_value=mock_span
        ), patch("tools.product_tools.end_span") as mock_end:
            from tools.product_tools import fetch_specs_tool

            result = fetch_specs_tool.invoke({"product_ids": ["P001"]})

        assert isinstance(result, dict)
        mock_end.assert_called_once()
