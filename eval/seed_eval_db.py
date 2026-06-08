#!/usr/bin/env python3
"""
eval/seed_eval_db.py — Idempotently seeds PostgreSQL with eval-specific data.

Creates:
  - User: eval_guest_001 (guest, is_authenticated=False)
  - Orders: ORD-EVAL-001..005 (order agent test cases)
  - Orders: ORD-EVAL-011..014 (support agent test cases, order_value >= 1500)
  - CarrierTracking rows for the orders that have tracking numbers

Safe to run multiple times — skips rows that already exist.
"""

import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def seed():
    from database.connection import SessionLocal
    from database.models import User, Order, CarrierTracking, SupportTicket

    db = SessionLocal()
    try:
        # ── clear stale support tickets so eval always starts from a clean state ──
        deleted = db.query(SupportTicket).filter(SupportTicket.user_id == "eval_guest_001").delete()
        db.commit()
        if deleted:
            logger.info(f"Cleared {deleted} stale support ticket(s) for eval_guest_001")

        # ── eval_guest_001 user ───────────────────────────────────────────────
        if not db.query(User).filter(User.user_id == "eval_guest_001").first():
            db.add(User(
                user_id="eval_guest_001",
                is_authenticated=False,
                auth_provider="anonymous",
                created_at=datetime.now(timezone.utc),
                last_login_at=datetime.now(timezone.utc),
            ))
            db.commit()
            logger.info("Created user: eval_guest_001")
        else:
            logger.info("User eval_guest_001 already exists — skipping")

        now = datetime.now(timezone.utc)
        today = now.date()

        # ── eval orders ───────────────────────────────────────────────────────
        eval_orders = [
            # Order agent cases
            {
                "order_id": "ORD-EVAL-001",
                "status": "shipped",
                "items": ["Sony WH-1000XM5 Headphones"],
                "carrier": "DTDC",
                "tracking_number": "DT251092001",
                "order_value": 24990.0,
                "order_date": datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=4),
                "expected_delivery": today + timedelta(days=3),
            },
            {
                "order_id": "ORD-EVAL-002",
                "status": "shipped",
                "items": ["Logitech MX Master 3S Mouse"],
                "carrier": "BlueDart",
                "tracking_number": "BD123456001",
                "order_value": 8995.0,
                "order_date": datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=5),
                "expected_delivery": today,
            },
            {
                "order_id": "ORD-EVAL-003",
                "status": "delivered",
                "items": ["Apple AirPods Pro 2nd Gen"],
                "carrier": "FedEx",
                "tracking_number": "FX987654001",
                "order_value": 24900.0,
                "order_date": datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=8),
                "expected_delivery": today - timedelta(days=1),
            },
            {
                "order_id": "ORD-EVAL-004",
                "status": "processing",
                "items": ["Samsung Galaxy Watch 6"],
                "carrier": None,
                "tracking_number": None,
                "order_value": 29999.0,
                "order_date": datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=1),
                "expected_delivery": today + timedelta(days=5),
            },
            {
                "order_id": "ORD-EVAL-005",
                "status": "delayed",
                "items": ["Philips Air Fryer HD9200"],
                "carrier": "DHL",
                "tracking_number": "DL941982001",
                "order_value": 6995.0,
                "order_date": datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=8),
                "expected_delivery": today - timedelta(days=2),
            },
            # Support agent cases — order_value >= 1500 required for HIGH severity ticket
            {
                "order_id": "ORD-EVAL-011",
                "status": "delivered",
                "items": ["LG 43-inch 4K Smart TV"],
                "carrier": "Ekart",
                "tracking_number": "EK112233011",
                "order_value": 32999.0,
                "order_date": datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=12),
                "expected_delivery": today - timedelta(days=5),
            },
            {
                "order_id": "ORD-EVAL-012",
                "status": "delivered",
                "items": ["Dyson V12 Vacuum Cleaner"],
                "carrier": "BlueDart",
                "tracking_number": "BD123456012",
                "order_value": 45999.0,
                "order_date": datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=10),
                "expected_delivery": today - timedelta(days=4),
            },
            {
                "order_id": "ORD-EVAL-013",
                "status": "delivered",
                "items": ["OnePlus 12 5G Smartphone"],
                "carrier": "FedEx",
                "tracking_number": "FX987654013",
                "order_value": 64999.0,
                "order_date": datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=9),
                "expected_delivery": today - timedelta(days=3),
            },
            {
                "order_id": "ORD-EVAL-014",
                "status": "delivered",
                "items": ["Bose QuietComfort 45 Headphones"],
                "carrier": "DTDC",
                "tracking_number": "DT251092014",
                "order_value": 29900.0,
                "order_date": datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=8),
                "expected_delivery": today - timedelta(days=2),
            },
        ]

        for o in eval_orders:
            if db.query(Order).filter(Order.order_id == o["order_id"]).first():
                logger.info(f"Order {o['order_id']} already exists — skipping")
                continue
            db.add(Order(
                order_id=o["order_id"],
                user_id="eval_guest_001",
                status=o["status"],
                items=o["items"],
                carrier=o["carrier"],
                tracking_number=o["tracking_number"],
                order_value=o["order_value"],
                order_date=o["order_date"],
                expected_delivery=o["expected_delivery"],
            ))
            logger.info(f"Created order: {o['order_id']}")

        db.commit()

        # ── carrier_tracking rows ─────────────────────────────────────────────
        tracking_rows = [
            {
                "tracking_number": "DT251092001",
                "carrier_name": "DTDC",
                "current_status": "In Transit",
                "current_location": "Mumbai Hub",
                "events": [{"timestamp": str(now - timedelta(hours=6)), "status": "In Transit", "location": "Mumbai Hub"}],
            },
            {
                "tracking_number": "BD123456001",
                "carrier_name": "BlueDart",
                "current_status": "Out for Delivery",
                "current_location": "Local Delivery Center",
                "events": [{"timestamp": str(now - timedelta(hours=2)), "status": "Out for Delivery", "location": "Local Delivery Center"}],
            },
            {
                "tracking_number": "FX987654001",
                "carrier_name": "FedEx",
                "current_status": "Delivered",
                "current_location": "Delivered to Customer",
                "events": [{"timestamp": str(now - timedelta(days=1)), "status": "Delivered", "location": "Delivered to Customer"}],
            },
            {
                "tracking_number": "DL941982001",
                "carrier_name": "DHL",
                "current_status": "In Transit — Delayed",
                "current_location": "Chennai Hub",
                "events": [{"timestamp": str(now - timedelta(hours=12)), "status": "Delayed in Transit", "location": "Chennai Hub"}],
            },
            {
                "tracking_number": "EK112233011",
                "carrier_name": "Ekart",
                "current_status": "Delivered",
                "current_location": "Delivered to Customer",
                "events": [{"timestamp": str(now - timedelta(days=5)), "status": "Delivered", "location": "Delivered to Customer"}],
            },
        ]

        for row in tracking_rows:
            if db.query(CarrierTracking).filter(CarrierTracking.tracking_number == row["tracking_number"]).first():
                logger.info(f"Tracking {row['tracking_number']} already exists — skipping")
                continue
            db.add(CarrierTracking(
                tracking_number=row["tracking_number"],
                carrier_name=row["carrier_name"],
                current_status=row["current_status"],
                current_location=row["current_location"],
                events=row["events"],
                estimated_delivery=today + timedelta(days=1),
                last_updated=now,
            ))
            logger.info(f"Created tracking: {row['tracking_number']}")

        db.commit()
        logger.info("seed_eval_db complete — eval_guest_001 + 9 orders + 5 tracking rows ready")

    except Exception as e:
        db.rollback()
        logger.error(f"seed_eval_db failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
