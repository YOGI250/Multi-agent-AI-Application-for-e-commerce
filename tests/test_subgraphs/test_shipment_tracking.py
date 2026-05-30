# tests/test_subgraphs/test_shipment_tracking.py

import pytest
from unittest.mock import patch


def _state(**kwargs):
    base = {
        "order_data": {
            "carrier":         "DTDC",
            "tracking_number": "DT123456"
        },
        "carrier_name":            None,
        "tracking_number":         None,
        "tracking_data":           None,
        "eta":                     None,
        "tracking_events":         None,
        "current_location":        None,
        "langfuse_trace_id":       None,
        "langfuse_parent_span_id": None,
    }
    base.update(kwargs)
    return base


class TestGetCarrierInfo:

    def test_extracts_carrier_and_tracking_number(self):
        from subgraphs.shipment_tracking import get_carrier_info
        result = get_carrier_info(_state())
        assert result["carrier_name"] == "DTDC"
        assert result["tracking_number"] == "DT123456"

    def test_handles_missing_carrier_fields(self):
        from subgraphs.shipment_tracking import get_carrier_info
        result = get_carrier_info(_state(order_data={}))
        assert result["carrier_name"] is None
        assert result["tracking_number"] is None


class TestFetchTrackingDataNode:

    def test_fetches_tracking_data_when_number_present(self):
        from subgraphs.shipment_tracking import fetch_tracking_data_node
        mock_tracking = {
            "current_status":   "Delivered",
            "current_location": "Mumbai",
            "events":           [{"status": "dispatched"}]
        }
        with patch("subgraphs.shipment_tracking.fetch_tracking_data") as mock_tool:
            mock_tool.invoke.return_value = mock_tracking
            result = fetch_tracking_data_node(_state(tracking_number="DT123456"))
        assert result["tracking_data"]["current_status"] == "Delivered"
        assert result["tracking_data"]["current_location"] == "Mumbai"

    def test_skips_fetch_when_no_tracking_number(self):
        from subgraphs.shipment_tracking import fetch_tracking_data_node
        result = fetch_tracking_data_node(_state(tracking_number=None))
        assert result["tracking_data"] == {}

    def test_handles_none_tracking_response(self):
        from subgraphs.shipment_tracking import fetch_tracking_data_node
        with patch("subgraphs.shipment_tracking.fetch_tracking_data") as mock_tool:
            mock_tool.invoke.return_value = None
            result = fetch_tracking_data_node(_state(tracking_number="DT123456"))
        assert result["tracking_data"] == {}


class TestParseEta:

    def test_parses_all_tracking_fields(self):
        from subgraphs.shipment_tracking import parse_eta
        state = _state(tracking_data={
            "estimated_delivery": "2026-06-01",
            "current_location":   "Delhi Hub",
            "events":             [{"status": "dispatched"}, {"status": "in-transit"}]
        })
        result = parse_eta(state)
        assert result["eta"] == "2026-06-01"
        assert result["current_location"] == "Delhi Hub"
        assert len(result["tracking_events"]) == 2

    def test_defaults_when_tracking_data_empty(self):
        from subgraphs.shipment_tracking import parse_eta
        result = parse_eta(_state(tracking_data={}))
        assert result["eta"] == "Not available"
        assert result["current_location"] == "Unknown"
        assert result["tracking_events"] == []


class TestShipmentTrackingSubgraph:

    def test_subgraph_compiles(self):
        from subgraphs.shipment_tracking import shipment_tracking_subgraph
        assert shipment_tracking_subgraph is not None

    def test_full_flow(self):
        from subgraphs.shipment_tracking import shipment_tracking_subgraph
        mock_tracking = {
            "current_status":     "Shipped",
            "current_location":   "Chennai",
            "estimated_delivery": "2026-06-05",
            "events":             []
        }
        with patch("subgraphs.shipment_tracking.fetch_tracking_data") as mock_tool:
            mock_tool.invoke.return_value = mock_tracking
            result = shipment_tracking_subgraph.invoke(_state())
        assert result["carrier_name"] == "DTDC"
        assert result["eta"] == "2026-06-05"
        assert result["current_location"] == "Chennai"
