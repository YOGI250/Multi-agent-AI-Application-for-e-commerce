# services/mock_order_api.py

import logging
from typing import Optional
from database.connection import SessionLocal
from database.models import Order, CarrierTracking

logger = logging.getLogger(__name__)


def get_order(order_id: str) -> Optional[dict]:
    """
    Fetches a single order by order_id.
    Called by fetch_order_data tool node.
    Returns order as dictionary or None if not found.
    """
    db = SessionLocal()
    try:
        order = db.query(Order).filter(
            Order.order_id == order_id
        ).first()

        if not order:
            logger.info(f"Order not found: {order_id}")
            return None

        return {
            "order_id":          order.order_id,
            "user_id":           order.user_id,
            "status":            order.status,
            "items":             order.items,
            "carrier":           order.carrier,
            "tracking_number":   order.tracking_number,
            "order_date":        str(order.order_date),
            "expected_delivery": str(order.expected_delivery),
            "order_value":       float(order.order_value)
        }

    except Exception as e:
        logger.error(f"Error fetching order {order_id}: {e}")
        return None
    finally:
        db.close()


def get_tracking(tracking_number: str) -> Optional[dict]:
    """
    Fetches carrier tracking data by tracking number.
    Called by fetch_tracking_data tool node inside
    shipment_tracking subgraph.
    Simulates what a real carrier API would return.
    """
    db = SessionLocal()
    try:
        tracking = db.query(CarrierTracking).filter(
            CarrierTracking.tracking_number == tracking_number
        ).first()

        if not tracking:
            logger.info(f"Tracking not found: {tracking_number}")
            return None

        return {
            "tracking_number":   tracking.tracking_number,
            "carrier_name":      tracking.carrier_name,
            "current_status":    tracking.current_status,
            "current_location":  tracking.current_location,
            "events":            tracking.events,
            "estimated_delivery": str(tracking.estimated_delivery),
            "last_updated":      str(tracking.last_updated)
        }

    except Exception as e:
        logger.error(f"Error fetching tracking {tracking_number}: {e}")
        return None
    finally:
        db.close()
