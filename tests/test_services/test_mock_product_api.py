# tests/test_services/test_mock_product_api.py

import pytest
from unittest.mock import patch, MagicMock


class TestSearchProducts:

    def test_search_returns_list(self):
        mock_product        = MagicMock()
        mock_product.product_id = "P001"
        mock_product.name       = "Laptop"
        mock_product.category   = "Computers"
        mock_product.price      = 45000.0
        mock_product.brand      = "Dell"
        mock_product.rating     = 4.2
        mock_product.in_stock   = True
        mock_product.description = "A laptop"
        mock_product.image_url   = ""
        mock_product.specifications = {}

        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value    = mock_query
        mock_query.all.return_value      = [mock_product]

        with patch("services.mock_product_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"keyword": "laptop"})

        assert isinstance(result, list)

    def test_search_returns_empty_list(self):
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value           = mock_query
        mock_query.filter.return_value       = mock_query
        mock_query.order_by.return_value     = mock_query
        mock_query.limit.return_value        = mock_query
        mock_query.all.return_value          = []

        with patch("services.mock_product_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"keyword": "nonexistent"})

        assert result == []

    def test_search_with_empty_filters(self):
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value       = mock_query
        mock_query.filter.return_value   = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value    = mock_query
        mock_query.all.return_value      = []

        with patch("services.mock_product_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({})

        assert isinstance(result, list)




    def _make_mock_db(self, products=None):
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value       = mock_query
        mock_query.filter.return_value   = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value    = mock_query
        mock_query.all.return_value      = products or []
        return mock_db, mock_query

    def _make_product(self, product_type="laptop"):
        p = MagicMock()
        p.product_id      = "P001"
        p.name            = "Test Laptop"
        p.category        = "Computers"
        p.price           = 45000.0
        p.actual_price    = 50000.0
        p.discount_percent = 10.0
        p.brand           = "Dell"
        p.rating          = 4.2
        p.rating_count    = 100
        p.in_stock        = True
        p.product_type    = product_type
        return p

    def test_search_with_category_filter(self):
        mock_db, _ = self._make_mock_db()
        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"category": "Computers"})
        assert isinstance(result, list)

    def test_search_with_max_price_filter(self):
        mock_db, _ = self._make_mock_db()
        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"max_price": 50000})
        assert isinstance(result, list)

    def test_search_with_min_price_filter(self):
        mock_db, _ = self._make_mock_db()
        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"min_price": 10000})
        assert isinstance(result, list)

    def test_search_with_brand_as_string(self):
        mock_db, _ = self._make_mock_db()
        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"brand": "Dell"})
        assert isinstance(result, list)

    def test_search_with_brand_as_list(self):
        mock_db, _ = self._make_mock_db()
        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"brand": ["Dell", "HP", "Lenovo"]})
        assert isinstance(result, list)

    def test_search_with_min_rating_filter(self):
        mock_db, _ = self._make_mock_db()
        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"min_rating": 4.0})
        assert isinstance(result, list)

    def test_search_with_in_stock_filter(self):
        mock_db, _ = self._make_mock_db()
        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"in_stock": True})
        assert isinstance(result, list)

    def test_search_with_product_type_returns_results(self):
        p = self._make_product("laptop")
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value       = mock_query
        mock_query.filter.return_value   = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value    = mock_query
        mock_query.all.return_value      = [p]

        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"product_type": "laptop"})

        assert isinstance(result, list)
        assert len(result) == 1

    def test_search_with_product_type_returns_empty_when_no_match(self):
        mock_db    = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value       = mock_query
        mock_query.filter.return_value   = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value    = mock_query
        mock_query.all.return_value      = []

        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"product_type": "gaming_chair"})

        assert result == []

    def test_search_with_product_type_other_uses_keyword_fallback(self):
        mock_db, _ = self._make_mock_db()
        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"product_type": "other", "keyword": "chair"})
        assert isinstance(result, list)

    def test_search_exception_returns_empty_list(self):
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("DB error")
        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import search_products
            result = search_products({"keyword": "anything"})
        assert result == []


class TestGetProductSpecs:

    def test_get_specs_returns_dict(self):
        mock_product              = MagicMock()
        mock_product.product_id   = "P001"
        mock_product.specifications = {"RAM": "8GB"}
        mock_product.description  = "A laptop"

        mock_db = MagicMock()
        mock_db.query.return_value \
            .filter.return_value.all.return_value = [mock_product]

        with patch("services.mock_product_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_product_api import get_specs
            result = get_specs(["P001"])

        assert isinstance(result, dict)

    def test_get_specs_exception_returns_empty_dict(self):
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("DB error")
        with patch("services.mock_product_api.SessionLocal", return_value=mock_db):
            from services.mock_product_api import get_specs
            result = get_specs(["P999"])
        assert result == {}