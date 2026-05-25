# subgraphs/shipment_tracking.py

import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from tools.order_tools import fetch_tracking_data
from langfuse_helpers.tracing import create_span, end_span

logger = logging.getLogger(__name__)


class ShipmentTrackingState(TypedDict):
    order_data:              dict
    carrier_name:            Optional[str]
    tracking_number:         Optional[str]
    tracking_data:           Optional[dict]
    eta:                     Optional[str]
    tracking_events:         Optional[list]
    current_location:        Optional[str]
    langfuse_trace_id:       Optional[str]
    langfuse_parent_span_id: Optional[str]


def get_carrier_info(state: ShipmentTrackingState) -> ShipmentTrackingState:
    logger.info("Subgraph node: get_carrier_info")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")

    span = create_span(
        trace_id              = trace_id,
        name                  = "get_carrier_info",
        parent_observation_id = parent_id,
        input_data            = {"order_data": state.get("order_data", {})}
    ) if trace_id else None

    order = state.get("order_data", {})
    state["carrier_name"]    = order.get("carrier")
    state["tracking_number"] = order.get("tracking_number")

    if span:
        end_span(span, {
            "carrier_name":    state["carrier_name"],
            "tracking_number": state["tracking_number"]
        })

    return state


def fetch_tracking_data_node(
    state: ShipmentTrackingState
) -> ShipmentTrackingState:
    logger.info("Subgraph node: fetch_tracking_data_node")

    trace_id        = state.get("langfuse_trace_id")
    parent_id       = state.get("langfuse_parent_span_id")
    tracking_number = state.get("tracking_number")

    if not tracking_number:
        state["tracking_data"] = {}
        return state

    span = create_span(
        trace_id              = trace_id,
        name                  = "fetch_tracking_data",
        parent_observation_id = parent_id,
        input_data            = {"tracking_number": tracking_number}
    ) if trace_id else None

    tracking = fetch_tracking_data.invoke(
        {"tracking_number": tracking_number}
    )
    state["tracking_data"] = tracking or {}

    if span:
        end_span(span, {
            "status":   (tracking or {}).get("current_status"),
            "location": (tracking or {}).get("current_location")
        })

    return state


def parse_eta(state: ShipmentTrackingState) -> ShipmentTrackingState:
    logger.info("Subgraph node: parse_eta")

    trace_id  = state.get("langfuse_trace_id")
    parent_id = state.get("langfuse_parent_span_id")
    tracking  = state.get("tracking_data", {})

    span = create_span(
        trace_id              = trace_id,
        name                  = "parse_eta",
        parent_observation_id = parent_id,
        input_data            = {"tracking_data": tracking}
    ) if trace_id else None

    state["eta"]              = str(tracking.get("estimated_delivery", "Not available"))
    state["current_location"] = tracking.get("current_location", "Unknown")
    state["tracking_events"]  = tracking.get("events", [])

    if span:
        end_span(span, {
            "eta":              state["eta"],
            "current_location": state["current_location"]
        })

    return state


def build_shipment_tracking_subgraph():
    graph = StateGraph(ShipmentTrackingState)

    graph.add_node("get_carrier_info",         get_carrier_info)
    graph.add_node("fetch_tracking_data_node", fetch_tracking_data_node)
    graph.add_node("parse_eta",                parse_eta)

    graph.set_entry_point("get_carrier_info")
    graph.add_edge("get_carrier_info",         "fetch_tracking_data_node")
    graph.add_edge("fetch_tracking_data_node", "parse_eta")
    graph.add_edge("parse_eta",                END)

    return graph.compile()


shipment_tracking_subgraph = build_shipment_tracking_subgraph()