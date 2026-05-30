# tools/order_tools.py

import logging
from langchain_core.tools import tool
from services.mock_order_api import get_order, get_tracking

logger = logging.getLogger(__name__)


@tool
def fetch_order_data(order_id: str) -> dict:
    """
    Fetches order details from the database using the order ID.
    Returns order information including status, items,
    carrier, tracking number, and expected delivery date.
    Returns empty dict if order not found.
    """
    logger.info(f"Tool called: fetch_order_data with order_id={order_id}")

    result = get_order(order_id)

    if not result:
        logger.warning(f"No order found for order_id: {order_id}")
        return {}

    logger.info(f"Order fetched successfully: {order_id} status={result['status']}")
    return result


@tool
def fetch_tracking_data(tracking_number: str) -> dict:
    """
    Fetches shipment tracking data from the carrier database
    using the tracking number.
    Returns current location, status, tracking events,
    and estimated delivery date.
    Returns empty dict if tracking not found.
    """
    logger.info(
        f"Tool called: fetch_tracking_data with "
        f"tracking_number={tracking_number}"
    )

    result = get_tracking(tracking_number)

    if not result:
        logger.warning(f"No tracking found for: {tracking_number}")
        return {}

    logger.info(
        f"Tracking fetched: {tracking_number} "
        f"status={result['current_status']}"
    )
    return result


@tool
def fetch_all_orders_for_user(user_id: str) -> list:
    """
    Fetches all orders placed by a specific user.
    Returns a list of orders with their details.
    Used when user asks to see all their orders
    without specifying a specific order ID.
    """
    logger.info(
        f"Tool called: fetch_all_orders_for_user "
        f"with user_id={user_id}"
    )
    try:
        from database.connection import SessionLocal
        from database.models import Order

        db      = SessionLocal()
        orders  = db.query(Order).filter(
            Order.user_id == user_id
        ).order_by(Order.order_date.desc()).all()

        result = []
        for o in orders:
            result.append({
                "order_id":          o.order_id,
                "status":            o.status,
                "carrier":           o.carrier,
                "tracking_number":   o.tracking_number,
                "order_value":       float(o.order_value),
                "expected_delivery": str(o.expected_delivery),
                "order_date":        str(o.order_date),
                "items":             o.items,
                "user_id":           o.user_id
            })
        db.close()
        logger.info(
            f"fetch_all_orders_for_user found "
            f"{len(result)} orders for {user_id}"
        )
        return result
    except Exception as e:
        logger.error(f"Error fetching all orders for user: {e}")
        return []