# tests/test_tools/test_order_tools.py

import pytest
from unittest.mock import patch, MagicMock
from utils.langfuse_context import set_trace_context


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture
def mock_order():
    return {
        "order_id": "ORD-1001",
        "user_id": "test_user_123",
        "status": "shipped",
        "carrier": "BlueDart",
        "tracking_number": "BD123456789",
        "order_value": 1299.0,
        "expected_delivery": "2026-05-28",
        "order_date": "2026-05-22",
        "items": "Laptop Stand",
    }


@pytest.fixture
def mock_tracking():
    return {
        "tracking_number": "BD123456789",
        "carrier": "BlueDart",
        "current_status": "In Transit",
        "current_location": "Mumbai Hub",
        "eta": "2026-05-28",
        "events": [],
    }


# ==========================================
# TESTS — fetch_order_data
# ==========================================
class TestFetchOrderData:

    def test_fetch_existing_order(self, mock_order):
        with patch("tools.order_tools.get_order") as mock_get:
            mock_get.return_value = mock_order
            from tools.order_tools import fetch_order_data

            result = fetch_order_data.invoke({"order_id": "ORD-1001"})

        assert result["order_id"] == "ORD-1001"
        assert result["status"] == "shipped"

    def test_fetch_nonexistent_order(self):
        with patch("tools.order_tools.get_order") as mock_get:
            mock_get.return_value = None
            from tools.order_tools import fetch_order_data

            result = fetch_order_data.invoke({"order_id": "ORD-9999"})

        assert result == {}

    def test_fetch_order_returns_correct_fields(self, mock_order):
        with patch("tools.order_tools.get_order") as mock_get:
            mock_get.return_value = mock_order
            from tools.order_tools import fetch_order_data

            result = fetch_order_data.invoke({"order_id": "ORD-1001"})

        assert "order_id" in result
        assert "status" in result
        assert "carrier" in result
        assert "order_value" in result


# ==========================================
# TESTS — fetch_tracking_data
# ==========================================
class TestFetchTrackingData:

    def test_fetch_existing_tracking(self, mock_tracking):
        with patch("tools.order_tools.get_tracking") as mock_get:
            mock_get.return_value = mock_tracking
            from tools.order_tools import fetch_tracking_data

            result = fetch_tracking_data.invoke({"tracking_number": "BD123456789"})

        assert result["tracking_number"] == "BD123456789"
        assert result["current_status"] == "In Transit"

    def test_fetch_nonexistent_tracking(self):
        with patch("tools.order_tools.get_tracking") as mock_get:
            mock_get.return_value = None
            from tools.order_tools import fetch_tracking_data

            result = fetch_tracking_data.invoke({"tracking_number": "INVALID123"})

        assert result == {}


# ==========================================
# TESTS — fetch_all_orders_for_user
# ==========================================
class TestFetchAllOrdersForUser:

    def test_fetch_returns_list(self):
        mock_order_obj = MagicMock()
        mock_order_obj.order_id = "ORD-1001"
        mock_order_obj.user_id = "test_user_123"
        mock_order_obj.status = "shipped"
        mock_order_obj.carrier = "BlueDart"
        mock_order_obj.tracking_number = "BD123"
        mock_order_obj.order_value = 1299.0
        mock_order_obj.expected_delivery = "2026-05-28"
        mock_order_obj.order_date = "2026-05-22"
        mock_order_obj.items = "Laptop Stand"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            mock_order_obj
        ]

        with patch("database.connection.SessionLocal", return_value=mock_db):
            from tools.order_tools import fetch_all_orders_for_user

            result = fetch_all_orders_for_user.invoke({"user_id": "test_user_123"})

        assert isinstance(result, list)

    def test_fetch_returns_empty_on_db_error(self):
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("DB error")

        with patch("database.connection.SessionLocal", return_value=mock_db):
            from tools.order_tools import fetch_all_orders_for_user

            result = fetch_all_orders_for_user.invoke({"user_id": "test_user_123"})

        assert result == []


# ==========================================
# TESTS — LangFuse span branches (with trace context)
# ==========================================
class TestOrderToolsWithSpan:

    def setup_method(self):
        set_trace_context("test-trace-id", "test-span-id")

    def teardown_method(self):
        set_trace_context(None, None)

    def test_fetch_order_found_with_span(self, mock_order):
        mock_span = MagicMock()
        with patch("tools.order_tools.get_order", return_value=mock_order), patch(
            "tools.order_tools.create_span", return_value=mock_span
        ), patch("tools.order_tools.end_span") as mock_end:
            from tools.order_tools import fetch_order_data

            result = fetch_order_data.invoke({"order_id": "ORD-1001"})

        assert result["order_id"] == "ORD-1001"
        mock_end.assert_called_once()

    def test_fetch_order_not_found_with_span(self):
        mock_span = MagicMock()
        with patch("tools.order_tools.get_order", return_value=None), patch(
            "tools.order_tools.create_span", return_value=mock_span
        ), patch("tools.order_tools.end_span") as mock_end:
            from tools.order_tools import fetch_order_data

            result = fetch_order_data.invoke({"order_id": "ORD-9999"})

        assert result == {}
        mock_end.assert_called_once()

    def test_fetch_tracking_found_with_span(self, mock_tracking):
        mock_span = MagicMock()
        with patch("tools.order_tools.get_tracking", return_value=mock_tracking), patch(
            "tools.order_tools.create_span", return_value=mock_span
        ), patch("tools.order_tools.end_span") as mock_end:
            from tools.order_tools import fetch_tracking_data

            result = fetch_tracking_data.invoke({"tracking_number": "BD123456789"})

        assert result["tracking_number"] == "BD123456789"
        mock_end.assert_called_once()

    def test_fetch_tracking_not_found_with_span(self):
        mock_span = MagicMock()
        with patch("tools.order_tools.get_tracking", return_value=None), patch(
            "tools.order_tools.create_span", return_value=mock_span
        ), patch("tools.order_tools.end_span") as mock_end:
            from tools.order_tools import fetch_tracking_data

            result = fetch_tracking_data.invoke({"tracking_number": "INVALID"})

        assert result == {}
        mock_end.assert_called_once()

    def test_fetch_all_orders_with_span(self):
        mock_order_obj = MagicMock()
        mock_order_obj.order_id = "ORD-1001"
        mock_order_obj.user_id = "test_user_123"
        mock_order_obj.status = "shipped"
        mock_order_obj.carrier = "BlueDart"
        mock_order_obj.tracking_number = "BD123"
        mock_order_obj.order_value = 1299.0
        mock_order_obj.expected_delivery = "2026-05-28"
        mock_order_obj.order_date = "2026-05-22"
        mock_order_obj.items = "Laptop Stand"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            mock_order_obj
        ]

        mock_span = MagicMock()
        with patch("database.connection.SessionLocal", return_value=mock_db), patch(
            "tools.order_tools.create_span", return_value=mock_span
        ), patch("tools.order_tools.end_span") as mock_end:
            from tools.order_tools import fetch_all_orders_for_user

            result = fetch_all_orders_for_user.invoke({"user_id": "test_user_123"})

        assert isinstance(result, list)
        mock_end.assert_called_once()

    def test_fetch_all_orders_error_with_span(self):
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("DB error")

        mock_span = MagicMock()
        with patch("database.connection.SessionLocal", return_value=mock_db), patch(
            "tools.order_tools.create_span", return_value=mock_span
        ), patch("tools.order_tools.end_span") as mock_end:
            from tools.order_tools import fetch_all_orders_for_user

            result = fetch_all_orders_for_user.invoke({"user_id": "test_user_123"})

        assert result == []
        mock_end.assert_called_once()
