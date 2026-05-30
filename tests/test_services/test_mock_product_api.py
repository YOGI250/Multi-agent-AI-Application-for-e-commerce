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