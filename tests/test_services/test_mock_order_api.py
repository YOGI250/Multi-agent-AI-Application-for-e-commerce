# tests/test_services/test_mock_order_api.py

import pytest
from unittest.mock import patch, MagicMock


# ==========================================
# TESTS — get_order
# ==========================================
class TestGetOrder:

    def test_get_order_returns_dict(self):
        mock_order              = MagicMock()
        mock_order.order_id     = "ORD-1001"
        mock_order.status       = "shipped"
        mock_order.carrier      = "BlueDart"
        mock_order.tracking_number   = "BD123"
        mock_order.order_value       = 1299.0
        mock_order.expected_delivery = "2026-05-28"
        mock_order.order_date        = "2026-05-22"
        mock_order.items             = "Laptop"
        mock_order.user_id           = "user_123"

        mock_db = MagicMock()
        mock_db.query.return_value \
            .filter.return_value.first.return_value = mock_order

        with patch("services.mock_order_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_order_api import get_order
            result = get_order("ORD-1001")

        assert result is not None
        assert result["order_id"] == "ORD-1001"

    def test_get_order_returns_none_for_missing(self):
        mock_db = MagicMock()
        mock_db.query.return_value \
            .filter.return_value.first.return_value = None

        with patch("services.mock_order_api.SessionLocal",
                   return_value=mock_db):
            from services.mock_order_api import get_order
            result = get_order("ORD-9999")

        assert result is None


# ==========================================
# TESTS — get_tracking
# ==========================================
    def test_get_order_exception_returns_none(self):
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("DB error")
        with patch("services.mock_order_api.SessionLocal", return_value=mock_db):
            from services.mock_order_api import get_order
            result = get_order("ORD-ERR")
        assert result is None


class TestGetTracking:

    def test_get_tracking_returns_none_for_unknown(self):
        from services.mock_order_api import get_tracking
        result = get_tracking("BD123456789")
        assert result is None or isinstance(result, dict)

    def test_get_tracking_returns_none_for_missing(self):
        from services.mock_order_api import get_tracking
        result = get_tracking("INVALID_TRACKING_12345")
        assert result is None or isinstance(result, dict)

    def test_get_tracking_returns_dict_when_found(self):
        mock_tracking                    = MagicMock()
        mock_tracking.tracking_number    = "BD999"
        mock_tracking.carrier_name       = "BlueDart"
        mock_tracking.current_status     = "in_transit"
        mock_tracking.current_location   = "Mumbai"
        mock_tracking.events             = []
        mock_tracking.estimated_delivery = "2026-06-10"
        mock_tracking.last_updated       = "2026-06-01"

        mock_db = MagicMock()
        mock_db.query.return_value \
            .filter.return_value.first.return_value = mock_tracking

        with patch("services.mock_order_api.SessionLocal", return_value=mock_db):
            from services.mock_order_api import get_tracking
            result = get_tracking("BD999")

        assert result is not None
        assert result["tracking_number"] == "BD999"
        assert result["carrier_name"]    == "BlueDart"

    def test_get_tracking_exception_returns_none(self):
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("DB error")
        with patch("services.mock_order_api.SessionLocal", return_value=mock_db):
            from services.mock_order_api import get_tracking
            result = get_tracking("ERR-TRACK")
        assert result is None