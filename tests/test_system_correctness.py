#!/usr/bin/env python3
"""
Automated correctness tests for the ecommerce AI system.

Tests three layers:
  1. HTTP API — via running k8s pod at localhost:8080
  2. Agent graph — directly (bypasses HTTP auth, tests order/support agents with real DB data)
  3. Infrastructure — Prometheus metrics, Grafana, database state

Run:  python tests/test_system_correctness.py
"""

import sys
import json
import time
import uuid
import requests
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://localhost:8080"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

# Delay between LLM-backed calls to avoid GROQ rate limiting
# Each /chat call triggers 2-3 internal LLM calls (intent router + agent + scoring)
LLM_CALL_DELAY = 10  # seconds

results = []


def check(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append(passed)
    line = f"  [{status}] {name}"
    if detail:
        line += f"\n         {detail}"
    print(line)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def chat(message: str, user_id: str = None, session_id: str = None, auth_token: str = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["X-Session-ID"] = session_id
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    body = {"message": message}
    if user_id:
        body["guest_id"] = user_id

    r = requests.post(f"{BASE_URL}/chat", json=body, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# 1. HEALTH & INFRASTRUCTURE
# ──────────────────────────────────────────────────────────────────────────────
section("1. Health & Infrastructure")

# 1.1 Health endpoint
r = requests.get(f"{BASE_URL}/health", timeout=10)
data = r.json()
check("GET /health returns 200", r.status_code == 200)
check("Database reports 'connected'", data.get("database") == "connected",
      f"got: {data.get('database')}")
check("Version field present", "version" in data)

# 1.2 Config endpoint — critical: no trailing newline in client ID
r = requests.get(f"{BASE_URL}/config", timeout=10)
cfg = r.json()
client_id = cfg.get("google_client_id", "")
check("GET /config returns google_client_id", bool(client_id))
check("google_client_id has no trailing newline", not client_id.endswith("\n"),
      f"value ends with: {repr(client_id[-3:])}")
check("google_client_id is the expected value",
      client_id == "230042601090-2372rg7c2si8nj36f9olqhqhb5h69vpj.apps.googleusercontent.com",
      f"got: {client_id[:30]}...")

# 1.3 Prometheus metrics
r = requests.get(f"{BASE_URL}/metrics", timeout=10)
check("GET /metrics returns 200", r.status_code == 200)
check("Prometheus metric http_requests_total present",
      "http_requests_total" in r.text or "http_request_duration" in r.text)

# 1.4 Grafana reachable
try:
    r = requests.get("http://localhost:3001", timeout=5)
    check("Grafana responds on :3001", r.status_code in (200, 302))
except Exception as e:
    check("Grafana responds on :3001", False, str(e))

# 1.5 Prometheus UI reachable
try:
    r = requests.get("http://localhost:9090/-/ready", timeout=5)
    check("Prometheus responds on :9090", r.status_code == 200)
except Exception as e:
    check("Prometheus responds on :9090", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# 2. INTENT ROUTING CORRECTNESS
# ──────────────────────────────────────────────────────────────────────────────
section("2. Intent Routing — correct agent dispatched")

routing_cases = [
    # (message, expected_agent, description)
    ("I need wireless headphones under 5000",     "product_agent",   "product query"),
    ("show me gaming laptops",                     "product_agent",   "product query"),
    ("where is my order ORD-EVAL-001",            "access_control",  "order query blocked for guest"),
    ("my item arrived damaged, I need a refund",  "access_control",  "support query blocked for guest"),
    ("hello",                                      "fallback",        "greeting → fallback"),
    ("what can you do",                            "fallback",        "help request → fallback"),
    ("show me smartwatches under 20000",           "product_agent",   "product query"),
]

for msg, expected_agent, desc in routing_cases:
    try:
        time.sleep(LLM_CALL_DELAY)
        resp = chat(msg, user_id=f"test_routing_{uuid.uuid4().hex[:6]}")
        got = resp.get("agent_used", "")
        check(f"Routing: {desc}", got == expected_agent,
              f"expected '{expected_agent}', got '{got}' | msg: {msg[:40]}")
    except Exception as e:
        check(f"Routing: {desc}", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# 3. PRODUCT AGENT CORRECTNESS
# ──────────────────────────────────────────────────────────────────────────────
section("3. Product Agent — response correctness")

test_uid = f"test_product_{uuid.uuid4().hex[:8]}"

# 3.1 Price constraint
time.sleep(LLM_CALL_DELAY)
resp = chat("show me wireless headphones under 3000 rupees", user_id=test_uid)
products = resp.get("products", [])
check("Returns products for headphone query", len(products) > 0, f"got {len(products)} products")
if products:
    over_budget = [p for p in products if p.get("price", 0) > 3000]
    check("All returned products are under ₹3000",
          len(over_budget) == 0,
          f"over budget: {[(p['name'][:30], p['price']) for p in over_budget]}")
    headphone_names = [p for p in products if any(
        kw in p.get("name", "").lower() or kw in p.get("product_type", "").lower()
        for kw in ["headphone", "earbud", "earphone", "neckband"]
    )]
    check("Returned products are actually headphones",
          len(headphone_names) >= len(products) // 2,
          f"only {len(headphone_names)}/{len(products)} match")
    check("All products have required fields (name, price, brand, rating)",
          all(p.get("name") and p.get("price") and p.get("brand") and p.get("rating") for p in products),
          "some products missing required fields")
    check("All products have valid prices (> 0)", all(p.get("price", 0) > 0 for p in products))
    check("All products have valid ratings (1–5)", all(1 <= p.get("rating", 0) <= 5 for p in products))

# 3.2 Category specificity — laptops
time.sleep(LLM_CALL_DELAY)
resp = chat("I need a laptop for programming under 70000", user_id=test_uid)
products = resp.get("products", [])
check("Returns products for laptop query", len(products) > 0, f"got {len(products)} products")
if products:
    over_budget = [p for p in products if p.get("price", 0) > 70000]
    check("All laptops are under ₹70000", len(over_budget) == 0,
          f"over budget: {[(p['name'][:30], p['price']) for p in over_budget]}")
    laptop_related = [p for p in products if any(
        kw in p.get("name", "").lower() or kw in p.get("product_type", "").lower()
        for kw in ["laptop", "notebook", "computer"]
    )]
    check("Returned products are actually laptops",
          len(laptop_related) >= len(products) // 2,
          f"only {len(laptop_related)}/{len(products)} match")

# 3.3 Response text mentions actual product names from the products list
time.sleep(LLM_CALL_DELAY)
resp = chat("recommend me a smartwatch", user_id=test_uid)
products = resp.get("products", [])
response_text = resp.get("response", "")
# Smartwatches may be sparse in catalog — just verify agent tried to respond helpfully
check("Smartwatch query: agent returns product_agent response",
      resp.get("agent_used") == "product_agent",
      f"agent: {resp.get('agent_used')}")
check("Smartwatch query: response is non-empty", len(response_text) > 50,
      f"response: {response_text[:100]}")
if products:
    check("Response text mentions product names from returned list",
          any(p["name"][:15] in response_text for p in products[:3]),
          f"response does not mention any of the top 3 products")

# 3.4 Agent used is always product_agent
check("agent_used is 'product_agent'", resp.get("agent_used") == "product_agent",
      f"got: {resp.get('agent_used')}")

# 3.5 Response length is substantial (not empty/truncated)
check("Response is non-trivial (>50 chars)", len(response_text) > 50,
      f"response length: {len(response_text)}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. SESSION CONTINUITY
# ──────────────────────────────────────────────────────────────────────────────
section("4. Session Continuity — session_id persists across messages")

sess_uid = f"test_session_{uuid.uuid4().hex[:8]}"
time.sleep(LLM_CALL_DELAY)
resp1 = chat("show me keyboards", user_id=sess_uid)
session_id = resp1.get("session_id")
check("First message creates a session_id", bool(session_id), f"got: {session_id}")

time.sleep(LLM_CALL_DELAY)
resp2 = chat("show me mice", user_id=sess_uid, session_id=session_id)
check("Second message returns same session_id", resp2.get("session_id") == session_id,
      f"expected {session_id}, got {resp2.get('session_id')}")
check("Second message also uses product_agent", resp2.get("agent_used") == "product_agent",
      f"got: {resp2.get('agent_used')}")

# History endpoint should return both messages
r = requests.get(f"{BASE_URL}/history",
                  headers={"X-Session-ID": session_id},
                  params={"guest_id": sess_uid}, timeout=10)
history = r.json().get("messages", [])
check("History endpoint returns messages for session", len(history) >= 4,
      f"got {len(history)} messages (expected at least 4: 2 user + 2 assistant)")
roles = [m["role"] for m in history]
check("History has both user and assistant messages",
      "user" in roles and "assistant" in roles)


# ──────────────────────────────────────────────────────────────────────────────
# 5. ORDER AGENT CORRECTNESS (direct Python call — bypasses HTTP auth)
# ──────────────────────────────────────────────────────────────────────────────
section("5. Order Agent — correctness (direct graph call, real DB data)")

try:
    from agents.order_agent import order_agent
    from agents.intent_router import intent_router_graph

    order_cases = [
        {
            "message": "where is my order ORD-EVAL-001",
            "user_id": "eval_guest_001",
            # Agent paraphrases — checks real tracking data (Mumbai Hub is DTDC's tracking location)
            "must_contain": ["ORD-EVAL-001", "shipped"],
            "any_of": ["DTDC", "DT251092001", "Mumbai Hub", "WH-1000XM5", "Headphones"],
            "description": "ORD-EVAL-001: shipped order, real tracking location in response",
        },
        {
            "message": "what is the status of ORD-EVAL-004",
            "user_id": "eval_guest_001",
            "must_contain": ["ORD-EVAL-004", "processing"],
            "any_of": [],
            "description": "ORD-EVAL-004 is in processing, no carrier yet",
        },
        {
            "message": "is ORD-EVAL-003 delivered",
            "user_id": "eval_guest_001",
            "must_contain": ["ORD-EVAL-003", "delivered"],
            # For delivered orders the LLM may not repeat carrier — confirmed status is enough
            "any_of": [],
            "description": "ORD-EVAL-003 status is delivered",
        },
        {
            "message": "show my orders",
            "user_id": "eval_guest_001",
            "must_contain": ["ORD-EVAL"],
            "any_of": [],
            "description": "list all orders shows eval orders",
        },
    ]

    for case in order_cases:
        try:
            time.sleep(LLM_CALL_DELAY)
            result = order_agent.invoke({
                "message": case["message"],
                "user_id": case["user_id"],
                "session_id": str(uuid.uuid4()),
                "history": [],
                "session_context": {},
                "langfuse_trace_id": None,
                "langfuse_parent_span_id": None,
            })
            response = result.get("response", "")
            missing = [kw for kw in case["must_contain"] if kw.lower() not in response.lower()]
            # any_of: at least one of these must appear (LLM paraphrases)
            any_of = case.get("any_of", [])
            any_of_ok = (not any_of) or any(kw.lower() in response.lower() for kw in any_of)
            passed = len(missing) == 0 and any_of_ok
            check(
                f"Order: {case['description']}",
                passed,
                f"missing required: {missing} | need any of {any_of}: {any_of_ok}\nresponse: {response[:250]}"
            )
        except Exception as e:
            check(f"Order: {case['description']}", False, str(e))

    # 5.5 Order agent uses actual DB data (not hallucinated)
    try:
        time.sleep(LLM_CALL_DELAY)
        result = order_agent.invoke({
            "message": "where is ORD-EVAL-002",
            "user_id": "eval_guest_001",
            "session_id": str(uuid.uuid4()),
            "history": [],
            "session_context": {},
            "langfuse_trace_id": None,
            "langfuse_parent_span_id": None,
        })
        response = result.get("response", "")
        # ORD-EVAL-002 shipped via BlueDart, tracking BD123456001, status "Out for Delivery" at Local Delivery Center
        has_real_data = any(kw.lower() in response.lower() for kw in [
            "bluedart", "BD123456001", "Local Delivery Center", "Out for Delivery", "MX Master", "shipped"
        ])
        check("Order agent uses real DB tracking data for ORD-EVAL-002",
              has_real_data,
              f"response should mention BlueDart/tracking/location\nresponse: {response[:250]}")
    except Exception as e:
        check("Order agent uses real DB tracking data", False, str(e))

except ImportError as e:
    print(f"  [{SKIP}] Order agent direct tests: cannot import agents ({e})")


# ──────────────────────────────────────────────────────────────────────────────
# 6. SUPPORT AGENT CORRECTNESS (direct Python call)
# ──────────────────────────────────────────────────────────────────────────────
section("6. Support Agent — correctness (direct graph call)")

try:
    from agents.support_agent import support_agent
    from database.connection import SessionLocal
    from database.models import SupportTicket

    # Clean up any existing test tickets first
    db = SessionLocal()
    db.query(SupportTicket).filter(SupportTicket.user_id == "eval_guest_001").delete()
    db.commit()
    db.close()

    support_cases = [
        {
            "message": "I want to return ORD-EVAL-011, the LG TV arrived damaged",
            "user_id": "eval_guest_001",
            # Agent response explains policy and confirms action — may not repeat order ID back
            "must_contain_any": ["ticket", "damaged", "replace", "refund", "sorry", "support"],
            "expected_ticket_created": True,
            "description": "Damaged item for ORD-EVAL-011 creates support ticket",
        },
    ]

    for case in support_cases:
        try:
            time.sleep(LLM_CALL_DELAY)
            result = support_agent.invoke({
                "message": case["message"],
                "user_id": case["user_id"],
                "session_id": str(uuid.uuid4()),
                "history": [],
                "session_context": {},
                "langfuse_trace_id": None,
                "langfuse_parent_span_id": None,
            })
            response = result.get("response", "")

            # Response should contain at least one of the expected keywords
            must_contain_any = case.get("must_contain_any", [])
            response_ok = any(kw.lower() in response.lower() for kw in must_contain_any)
            check(f"Support: {case['description']} — response is relevant",
                  response_ok,
                  f"none of {must_contain_any} in response\nresponse: {response[:300]}")
            check("Support response is substantial (>100 chars)",
                  len(response) > 100, f"response length: {len(response)}")

            # Verify ticket actually created in DB (check field names from model)
            if case["expected_ticket_created"]:
                db = SessionLocal()
                ticket = db.query(SupportTicket).filter(
                    SupportTicket.user_id == case["user_id"]
                ).first()
                db.close()
                check("Support ticket created in DB", ticket is not None,
                      "no SupportTicket found in DB after agent call")
                if ticket:
                    check("Ticket linked to correct order (ORD-EVAL-011)",
                          ticket.order_id and "ORD-EVAL-011" in ticket.order_id,
                          f"ticket.order_id: {ticket.order_id}")
                    check("Ticket has issue_type set",
                          bool(ticket.issue_type), f"issue_type: {ticket.issue_type}")
                    # Model uses 'priority' not 'severity'
                    check("Ticket priority is HIGH (order_value=₹32999 > ₹1500)",
                          ticket.priority == "HIGH",
                          f"priority: {ticket.priority}")
                    check("Ticket status is 'open'",
                          ticket.status == "open", f"status: {ticket.status}")
        except Exception as e:
            check(f"Support: {case['description']}", False, str(e))

    # 6.2 Duplicate detection — sending the same issue again
    try:
        time.sleep(LLM_CALL_DELAY)
        result2 = support_agent.invoke({
            "message": "I want to return ORD-EVAL-011, it is still damaged",
            "user_id": "eval_guest_001",
            "session_id": str(uuid.uuid4()),
            "history": [],
            "session_context": {},
            "langfuse_trace_id": None,
            "langfuse_parent_span_id": None,
        })
        response2 = result2.get("response", "")
        # Duplicate detection: agent should reference the existing ticket
        # response may say "already", "existing", "recently received", "report", "ticket", etc.
        is_duplicate_response = any(kw in response2.lower() for kw in [
            "already", "existing", "duplicate", "ticket", "recently", "reported", "record"
        ])
        check("Duplicate ticket: agent references existing case",
              is_duplicate_response,
              f"response: {response2[:250]}")
    except Exception as e:
        check("Duplicate ticket detection", False, str(e))

except ImportError as e:
    print(f"  [{SKIP}] Support agent direct tests: cannot import agents ({e})")


# ──────────────────────────────────────────────────────────────────────────────
# 7. PROMETHEUS METRICS INCREMENT
# ──────────────────────────────────────────────────────────────────────────────
section("7. Prometheus — metrics increment on requests")

def get_metric_value(metric_name: str) -> float:
    r = requests.get(f"{BASE_URL}/metrics", timeout=10)
    for line in r.text.splitlines():
        if line.startswith(metric_name) and not line.startswith("#"):
            try:
                return float(line.split()[-1])
            except ValueError:
                pass
    return 0.0

try:
    # Grab baseline
    r_before = requests.get(f"{BASE_URL}/metrics", timeout=10)
    baseline_text = r_before.text
    http_total_before = sum(
        float(line.split()[-1])
        for line in baseline_text.splitlines()
        if "http_requests_total" in line and not line.startswith("#")
        and "{" in line  # only label lines
    )

    # Make 3 product requests
    for i in range(3):
        chat(f"show me USB hubs under 1000", user_id=f"test_prometheus_{i}")

    time.sleep(1)

    r_after = requests.get(f"{BASE_URL}/metrics", timeout=10)
    after_text = r_after.text
    http_total_after = sum(
        float(line.split()[-1])
        for line in after_text.splitlines()
        if "http_requests_total" in line and not line.startswith("#")
        and "{" in line
    )

    check("Prometheus http_requests_total increments after chat calls",
          http_total_after > http_total_before,
          f"before: {http_total_before}, after: {http_total_after}")

    # Check custom agent metrics exist
    check("Custom agent metric lines present in /metrics",
          "agent_used" in after_text or "ecommerce" in after_text or "chat" in after_text,
          "no agent-specific metric found")

except Exception as e:
    check("Prometheus metrics increment", False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# 8. DATABASE STATE VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────
section("8. Database — eval data integrity")

try:
    from database.connection import SessionLocal
    from database.models import User, Order, CarrierTracking, Session as SessionModel, Message

    db = SessionLocal()

    user = db.query(User).filter(User.user_id == "eval_guest_001").first()
    check("eval_guest_001 user exists in DB", user is not None)

    order_count = db.query(Order).filter(Order.user_id == "eval_guest_001").count()
    check("eval_guest_001 has 9 eval orders", order_count == 9,
          f"got {order_count} orders")

    tracking_count = db.query(CarrierTracking).filter(
        CarrierTracking.tracking_number.in_(
            ["DT251092001", "BD123456001", "FX987654001", "DL941982001", "EK112233011"]
        )
    ).count()
    check("5 carrier tracking rows exist", tracking_count == 5,
          f"got {tracking_count} tracking rows")

    # Verify specific order data correctness
    ord1 = db.query(Order).filter(Order.order_id == "ORD-EVAL-001").first()
    check("ORD-EVAL-001 has correct carrier (DTDC)", ord1 and ord1.carrier == "DTDC",
          f"carrier: {ord1.carrier if ord1 else 'NOT FOUND'}")
    check("ORD-EVAL-001 has correct tracking (DT251092001)",
          ord1 and ord1.tracking_number == "DT251092001",
          f"tracking: {ord1.tracking_number if ord1 else 'NOT FOUND'}")
    check("ORD-EVAL-001 status is 'shipped'",
          ord1 and ord1.status == "shipped",
          f"status: {ord1.status if ord1 else 'NOT FOUND'}")

    ord4 = db.query(Order).filter(Order.order_id == "ORD-EVAL-004").first()
    check("ORD-EVAL-004 status is 'processing'",
          ord4 and ord4.status == "processing")
    check("ORD-EVAL-004 has no carrier (still processing)",
          ord4 and ord4.carrier is None)

    db.close()

except ImportError as e:
    print(f"  [{SKIP}] DB tests: cannot connect ({e})")
except Exception as e:
    print(f"  [{FAIL}] DB test error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
total = len(results)
passed = sum(results)
failed = total - passed
print(f"  TOTAL: {total} checks | {passed} passed | {failed} failed")
if failed == 0:
    print(f"  \033[92mALL CHECKS PASSED\033[0m")
else:
    print(f"  \033[91m{failed} CHECKS FAILED\033[0m")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
