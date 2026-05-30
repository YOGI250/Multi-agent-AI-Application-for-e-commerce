#!/usr/bin/env python3
"""
tests/e2e_40cases.py — 40-case end-to-end test

Calls intent_router_graph directly (no HTTP / no JWT needed).
Creates a disposable test user + 5 sample orders on first run.

Usage:
    python tests/e2e_40cases.py
    python tests/e2e_40cases.py --stop-on-fail
"""

import argparse
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import SessionLocal
from database.models import User, Order
from agents.intent_router import intent_router_graph
from monitoring.logging_config import setup_logging

setup_logging(log_level="ERROR")   # silence agent logs during test run

# ── test user ───────────────────────────────────────────────────────────────
TEST_USER_ID = "google_e2e_test_999001"
UID_SUFFIX   = TEST_USER_ID[-6:].upper()   # "999001"

ORDER_IDS = {
    1: f"ORD-{UID_SUFFIX}-001",   # Wireless Headphones  | delivered  | ₹1299
    2: f"ORD-{UID_SUFFIX}-002",   # Laptop Stand         | shipped    | ₹2500
    3: f"ORD-{UID_SUFFIX}-003",   # USB Hub              | processing | ₹499
    4: f"ORD-{UID_SUFFIX}-004",   # Mechanical Keyboard  | delayed    | ₹899
    5: f"ORD-{UID_SUFFIX}-005",   # Phone Case           | delivered  | ₹350
}


def seed_test_user(db):
    """Create test user + 5 orders if they don't exist yet."""
    user = db.query(User).filter(User.user_id == TEST_USER_ID).first()
    if not user:
        user = User(
            user_id          = TEST_USER_ID,
            is_authenticated = True,
            email            = "e2e_test@example.com",
            name             = "E2E Test User",
            auth_provider    = "google",
            created_at       = datetime.utcnow(),
            last_login_at    = datetime.utcnow(),
        )
        db.add(user)
        db.commit()

        today = datetime.today().date()
        sample_orders = [
            dict(order_id=ORDER_IDS[1], status="delivered",  carrier="DTDC",     tracking_number="DT251092059",
                 order_value=1299.0, expected_delivery=str(today - timedelta(days=18)),
                 order_date=str(today - timedelta(days=27)), items="Wireless Headphones"),
            dict(order_id=ORDER_IDS[2], status="shipped",    carrier="BlueDart", tracking_number="BD123456789",
                 order_value=2500.0, expected_delivery=str(today + timedelta(days=3)),
                 order_date=str(today - timedelta(days=6)),  items="Laptop Stand"),
            dict(order_id=ORDER_IDS[3], status="processing", carrier="FedEx",    tracking_number="FX987654321",
                 order_value=499.0,  expected_delivery=str(today + timedelta(days=5)),
                 order_date=str(today - timedelta(days=2)),  items="USB Hub"),
            dict(order_id=ORDER_IDS[4], status="delayed",    carrier="Ekart",    tracking_number="EK112233445",
                 order_value=899.0,  expected_delivery=str(today - timedelta(days=3)),
                 order_date=str(today - timedelta(days=13)), items="Mechanical Keyboard"),
            dict(order_id=ORDER_IDS[5], status="delivered",  carrier="DHL",      tracking_number="DL941982162",
                 order_value=350.0,  expected_delivery=str(today - timedelta(days=23)),
                 order_date=str(today - timedelta(days=27)), items="Phone Case"),
        ]
        for o in sample_orders:
            db.add(Order(user_id=TEST_USER_ID, **o))
        db.commit()
        print(f"[setup] Created test user {TEST_USER_ID} with 5 orders\n")
    else:
        print(f"[setup] Using existing test user {TEST_USER_ID}\n")


# ── test cases ──────────────────────────────────────────────────────────────
# Each entry: (label, message)
# Cases within the same numbered group share session context.

CASES = [
    # ── GROUP A: Smartwatch thread (5 turns, context flows) ──────────────
    ("smartwatch search",         "show me smartwatches under 3000"),
    ("calling feature filter",    "which ones have bluetooth calling feature"),
    ("brand switch Fire-Boltt",   "show only Fire-Boltt"),
    ("price refinement 2k",       "under 2000"),
    ("top rated smartwatch",      "show me the highest rated one"),

    # ── GROUP B: Order queries (5 turns) ─────────────────────────────────
    ("list all orders",           "show me all my orders"),
    ("track shipped order",       f"track my order {ORDER_IDS[2]}"),
    ("delayed order status",      f"what is the status of {ORDER_IDS[4]}"),
    ("delayed order ETA",         "when will the delayed one arrive"),
    ("order count",               "how many orders have I placed"),

    # ── GROUP C: Support queries (5 turns) ────────────────────────────────
    ("damaged product",           f"I received a damaged product from {ORDER_IDS[1]}"),
    ("wrong item",                f"wrong item was delivered in {ORDER_IDS[2]}"),
    ("cancel eligible order",     f"I want to cancel {ORDER_IDS[3]}"),
    ("cancel delivered order",    f"I want to cancel {ORDER_IDS[1]}"),
    ("refund request",            f"I want a refund for {ORDER_IDS[2]}"),

    # ── GROUP D: Cross-agent context switch ───────────────────────────────
    ("greeting after support",    "hi"),
    ("help request",              "what can you help me with"),
    ("product after support",     "show me boAt headphones under 1500"),
    ("price refinement hdphones", "under 1000"),
    ("brand switch Sony",         "what about Sony"),

    # ── GROUP E: Home appliances thread ───────────────────────────────────
    ("mixer grinder",             "mixer grinder under 2000"),
    ("brand switch Bajaj mixer",  "what about Bajaj"),
    ("steam iron",                "steam iron under 1500"),
    ("brand switch Philips iron", "what about Philips"),
    ("coffee maker",              "show me coffee makers under 3000"),

    # ── GROUP F: More order scenarios ─────────────────────────────────────
    ("re-list orders",            "list all my orders again"),
    ("specific delivered order",  f"tell me about {ORDER_IDS[5]}"),
    ("processing order status",   f"is {ORDER_IDS[3]} shipped yet"),
    ("vague order question",      "I placed an order, where is it"),
    ("compare two orders",        f"which arrived first {ORDER_IDS[1]} or {ORDER_IDS[5]}"),

    # ── GROUP G: Edge cases ───────────────────────────────────────────────
    ("misspelling speaker",       "blutooth speker under 2000"),
    ("out of domain query",       "masala dosa near me"),
    ("laptop bag",                "laptop bag under 1000"),
    ("screen protector",          "tempered glass screen protector under 300"),
    ("gaming chair",              "gaming chair under 10000"),
    ("USB hub search",            "USB hub under 500"),

    # ── GROUP H: Final support + context carry ────────────────────────────
    ("vague complaint",           "I have an issue with my recent order"),
    ("provide order ID",          f"it is {ORDER_IDS[2]}"),
    ("high value refund",         f"I need a refund for {ORDER_IDS[2]}"),
    ("webcam search",             "webcam for work from home under 2000"),
]

assert len(CASES) == 40, f"Expected 40 cases, got {len(CASES)}"


# ── runner ──────────────────────────────────────────────────────────────────

def run_case(n: int, label: str, message: str,
             session_id: str, history: list,
             session_context: dict) -> dict:
    """Invoke the intent_router_graph and return the full result state."""
    state = intent_router_graph.invoke({
        "message":                 message,
        "user_id":                 TEST_USER_ID,
        "session_id":              session_id,
        "history":                 history,
        "is_authenticated":        True,
        "intent":                  None,
        "confidence":              None,
        "reason":                  None,
        "response":                None,
        "agent_used":              None,
        "products":                None,
        "session_context":         session_context,
        "langfuse_trace_id":       None,
        "langfuse_parent_span_id": None,
    })
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Stop immediately if a case crashes")
    args = parser.parse_args()

    # seed test user
    db = SessionLocal()
    try:
        seed_test_user(db)
    finally:
        db.close()

    session_id      = str(uuid.uuid4())
    history         = []
    session_context = {}
    passed          = 0
    failed          = 0

    print(f"Session: {session_id}\n{'─' * 60}")

    for i, (label, message) in enumerate(CASES, start=1):
        print(f"\n=== {i}: {label} ===")
        print(f"Message: {message}")
        try:
            result = run_case(
                i, label, message,
                session_id, history, session_context
            )

            agent    = result.get("agent_used", "unknown")
            response = result.get("response", "") or ""
            new_ctx  = result.get("session_context") or {}

            # update session context for next turn
            session_context.update(new_ctx)

            # append to history so follow-ups have context
            history.append({"role": "user",      "content": message})
            history.append({"role": "assistant",  "content": response})
            history = history[-20:]   # keep last 20 messages

            print(f"Agent   : {agent}")
            print(f"Response: {response[:200]}")
            passed += 1

        except Exception as e:
            print(f"ERROR   : {e}")
            failed += 1
            if args.stop_on_fail:
                break

    print(f"\n{'─' * 60}")
    print(f"Results : {passed} passed  |  {failed} failed  |  {passed+failed} total")


if __name__ == "__main__":
    main()
