# E-Commerce AI Support System — Full Project Context

This document covers every aspect of the project: what it does, how it is built,
why each design decision was made, and how all the pieces connect. Read this top to
bottom before asking questions about any part of the system.

---

## 1. What This Project Is

A **multi-agent AI customer support chatbot** for an e-commerce store.
Users talk to a single chat interface. Depending on what they ask, one of three
specialist AI agents handles the request:

| Agent | What it handles |
|---|---|
| Order Agent | Tracking, status, delivery dates, "how many orders do I have?" |
| Product Agent | Product search, recommendations, comparisons, filters |
| Support Agent | Complaints, refunds, damaged items, wrong items, cancellations |

A top-level **Intent Router** receives every message and decides which agent to call.

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI (Python) |
| AI Orchestration | LangGraph (StateGraph-based agent graphs) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| LLM Calls | LangChain `ChatGroq` |
| Database | PostgreSQL (via SQLAlchemy ORM) |
| Auth | Google OAuth 2.0 (JWT token verification) |
| Observability | LangFuse (tracing, prompt management, scoring) |
| Metrics | Prometheus + Grafana |
| Frontend | Plain HTML/JS served as static files by FastAPI |
| Deployment | Docker Compose |

---

## 3. Project Directory Structure

```
ecommerce-agent/
├── main.py                    ← FastAPI app entry point
├── api/
│   ├── routes.py              ← /chat, /health, /config endpoints
│   └── schemas.py             ← Pydantic request/response models
├── agents/
│   ├── intent_router.py       ← Top-level router (classifies intent, calls agents)
│   ├── order_agent.py         ← Handles order tracking queries
│   ├── product_agent.py       ← Handles product search queries
│   └── support_agent.py       ← Handles complaints and support issues
├── subgraphs/
│   ├── escalation_handler.py  ← Sub-workflow inside support_agent (ticket creation)
│   ├── shipment_tracking.py   ← Sub-workflow inside order_agent (tracking lookup)
│   └── product_enrichment.py  ← Sub-workflow inside product_agent (scoring)
├── tools/
│   ├── order_tools.py         ← LangChain tools for fetching order/tracking data
│   ├── product_tools.py       ← LangChain tools for searching products
│   └── support_tools.py       ← LangChain tools for policy lookup and ticket creation
├── services/
│   ├── mock_order_api.py      ← DB queries for orders and tracking
│   ├── mock_product_api.py    ← DB queries for products
│   └── mock_support_api.py    ← DB queries for policies and ticket creation
├── database/
│   ├── models.py              ← SQLAlchemy ORM models (all tables)
│   └── connection.py          ← DB engine, SessionLocal, create_tables()
├── utils/
│   └── memory.py              ← Shared session context helpers
├── langfuse_helpers/
│   ├── tracing.py             ← LangFuse trace/span/generation/prompt helpers
│   └── scoring.py             ← Auto-scoring heuristics per agent
├── monitoring/
│   ├── metrics.py             ← Prometheus counters and histograms
│   └── logging_config.py      ← Structured JSON logging setup
├── config/
│   └── settings.py            ← All config from .env (Pydantic BaseSettings)
├── frontend/
│   └── index.html             ← Chat UI (served at /app)
└── docker-compose.yml         ← Runs app + postgres + langfuse + grafana
```

---

## 4. How A Request Flows (End to End)

```
User types message in browser
        ↓
POST /chat  (api/routes.py)
        ↓
resolve_user()       — identify user (Google OAuth or guest UUID)
resolve_session()    — load or create session, fetch last 10 messages as history
                       also loads session_context JSON from DB
        ↓
create_trace()       — open a LangFuse trace for this request
        ↓
intent_router_graph.invoke(...)   ← main entry point
        ↓
┌─────────────────────────────────────┐
│  INTENT ROUTER  (agents/intent_router.py)
│
│  Node 1: intent_router
│    ─ Pre-check: is message a greeting/help phrase? → unknown (no LLM)
│    ─ Pre-check: did support agent ask for order ID and user now provided one?
│        → force intent=support_query (no LLM)
│    ─ Otherwise: call LLM → classify into order_query / product_query /
│                             support_query / unknown
│
│  Edge: route_to_agent()
│    ─ unknown or low confidence → ask_clarification or fallback_response
│    ─ not authenticated + not product_query → access_denied
│    ─ order_query  → run_order_agent node
│    ─ product_query → run_product_agent node
│    ─ support_query → run_support_agent node
│
│  Node 2/3/4: run_order_agent / run_product_agent / run_support_agent
│    ─ creates a LangFuse span for the subgraph call
│    ─ invokes the relevant compiled agent graph
│    ─ collects response + updated session_context
└─────────────────────────────────────┘
        ↓
update_session_context()  — persist updated session_context to DB
score_response()          — auto-score on 5 metrics in LangFuse
record_request_metrics()  — update Prometheus counters
save_messages()           — save user + assistant messages to DB
        ↓
return ChatResponse(response, session_id, user_id, agent_used, products, ...)
```

---

## 5. Session and Memory System

### Problem it solves
Without memory, the LLM re-reads the full conversation history on every message.
This wastes tokens and causes biased routing (e.g. "hi" being classified as support
because recent messages were about a support issue).

### How it works

**`session_context`** is a JSON object stored in the `sessions` table (PostgreSQL JSONB column).
It is a compact structured summary of what has happened in the conversation so far.

Example value:
```json
{
  "topic": "order_query",
  "orders_listed": true,
  "order_count": 5,
  "order_statuses": {
    "delivered": ["ORD-001", "ORD-005"],
    "delayed": ["ORD-004"],
    "shipped": ["ORD-002"],
    "processing": ["ORD-003"]
  }
}
```

**Three helper functions** in `utils/memory.py`:

```python
format_context(ctx)
# Converts session_context dict into a compact one-line string for LLM prompts.
# Example output: "[topic=order_query | orders_listed=true(5 orders)]"
# ~15 tokens instead of injecting raw history

format_recent_messages(history, n=2, max_chars=150)
# Returns last n conversation turns, each truncated to 150 chars.
# Used instead of dumping history[-8:] into prompts.
# Keeps prompts small while giving LLM enough context for follow-up questions.

merge_context(old, new_facts)
# Merges new facts into existing context. Only overwrites non-None values.
# Preserves older facts (e.g. order_statuses) when a new turn adds topic only.
```

**Flow per request:**
1. `resolve_session()` loads `session_context` from DB and passes it into the LangGraph state
2. Every agent reads `state["session_context"]` and passes `format_context()` + `format_recent_messages()` into its LLM prompts
3. Each agent updates `session_context` with new facts via `merge_context()`
4. After the agent returns, `update_session_context()` in routes.py persists the new context back to DB

---

## 6. Intent Router — Deep Dive

**File:** `agents/intent_router.py`

### State (RouterState TypedDict)
```
message, user_id, session_id, history, is_authenticated,
intent, confidence, reason, response, agent_used,
products, session_context, langfuse_trace_id, langfuse_parent_span_id
```

### Pre-LLM Checks (run before the LLM to avoid context bias)

**Check 1 — Greeting/help detection (hard-coded, no LLM)**
```python
_GREETINGS = {"hi", "hello", "hey", "hiya", "greetings", ...}
_HELP_PREFIXES = ("help", "what can you do", "what do you do", ...)
```
If the message matches → `intent = "unknown"` immediately.
Why: The LLM was being biased by session context (e.g. `topic=support_query`) and
routing "hi" to the Support Agent. Hard-coded bypass is 100% reliable.

**Check 2 — Pending support context detection**
```python
_SUPPORT_NEEDS_ORDER_PHRASES = [
    "i'll need your order id",
    "order id looks like ord-",
    "typing 'show my orders'",
]
```
If the last few assistant messages contain these phrases AND the current message
contains an ORD- pattern → `intent = "support_query"` immediately.
Why: When the support agent asks for an order ID and the user responds with one,
the LLM was sometimes routing it to the order agent instead.

### LLM Classification
The LLM classifies into: `order_query`, `product_query`, `support_query`, `unknown`.

### Routing Logic (`route_to_agent`)
```
unknown                          → fallback_response
confidence < 0.7 (threshold)    → ask_clarification
not authenticated + not product  → access_denied
order_query                      → run_order_agent
support_query                    → run_support_agent
product_query                    → run_product_agent
```

Guest users (not authenticated) can only use the product agent.
Order and support require login.

---

## 7. Order Agent — Deep Dive

**File:** `agents/order_agent.py`

### Flow (nodes in sequence)

```
validate_input
    ↓ (order_id_found=True → fetch, False → error_response)
fetch_order_data_node
    ↓ (order_data found → analyze, not found → error_response)
analyze_order_status    ← LLM: classifies order situation
    ↓
shipment_tracking_node  ← calls shipment_tracking subgraph
    ↓
generate_response       ← LLM: writes the final answer
```

### validate_input — 4-Step Logic (most important node)

**Step 1 — Explicit ORD- pattern**
```
regex: ORD-[A-Z0-9-]+
```
If found → set `order_id`, `order_id_found=True`, `show_all_orders=False`

**Step 2 — Status-based resolution from session_context**
```python
order_statuses = ctx.get("order_statuses", {})
# e.g. {"delayed": ["ORD-004"], "delivered": ["ORD-001", "ORD-005"]}
for status, oids in order_statuses.items():
    if status in msg_lower and len(oids) == 1:
        resolved_id = oids[0]  # unambiguous — only one order with this status
```
Why: "tell me about the delayed one" should resolve to ORD-004 without user typing the ID.
Only resolves if exactly one order has that status; if multiple → falls through to ask user.

**Step 3 — Keyword-based "show all orders" detection**
```python
show_all_keywords = ["MY ORDERS", "ALL ORDERS", "HOW MANY", "I PLACED", ...]
wants_all_orders = any(k in message for k in show_all_keywords)
```

**Step 4 — Session-context topic fallback**
```python
if not wants_all_orders and ctx.get("topic") == "order_query":
    wants_all_orders = True
```
Why: "out of these, which are delivered?" doesn't contain keywords but is a follow-up
in an order conversation. This ensures the agent fetches all orders for follow-up questions.

### generate_response — Two Cases

**Case 1 — All orders (show_all_orders=True)**
- Fetches all orders for the user
- Builds `order_statuses` map: `{"delivered": ["ORD-001"], "delayed": ["ORD-004"], ...}`
- Stores in `session_context` for future status-based resolution
- LLM formats response as plain text: `ORD-XXX - Product Name` (no markdown, no price/status)

**Case 2 — Specific order**
- Provides order details + tracking info to LLM
- LLM writes a friendly natural language response

### Shipment Tracking Subgraph
`subgraphs/shipment_tracking.py`

```
get_carrier_info → fetch_tracking_data_node → parse_eta
```
Fetches tracking record from `carrier_tracking` table using `tracking_number`.
Returns: `carrier_name`, `eta`, `current_location`, `tracking_events` (list of events).

---

## 8. Product Agent — Deep Dive

**File:** `agents/product_agent.py`

### Flow

```
extract_preferences   ← LLM: extracts category, product_type, price, brand, rating
    ↓
search_products_node  ← DB query using extracted filters
    ↓
   found?
    ├─ yes → rank_and_filter     ← LLM: re-ranks by relevance to user message
    │             ↓
    │         product_enrichment_node  ← subgraph: ratings + specs + scoring
    │             ↓
    │         format_recommendations  ← pure code: builds text response
    │
    └─ no (0 results) → broaden_search (removes brand/price filter, retries once)
                              ↓ (still no results)
                          no_results_response
```

### extract_preferences
LLM extracts structured filters:
```json
{
  "category": "Electronics",
  "product_type": "headphones",
  "max_price": 2000,
  "min_price": null,
  "brand": null,
  "min_rating": 4.0
}
```
`product_type` must be from a fixed list (mouse, keyboard, headphones, speaker, etc.)
If not in list → `"other"`.

The prompt includes `format_context()` + `format_recent_messages()` so follow-up
queries like "any cheaper options?" can inherit `product_type` from the previous search.

### search_products (services/mock_product_api.py)
Queries the `products` table with SQLAlchemy:
- First tries `product_type == exact_match` (fastest, most accurate)
- Falls back to keyword search on name if no exact matches
- Returns up to 20 candidates sorted by rating descending
- If `product_type = "other"` (e.g. "masala dosa") → empty results → leads to `no_results_response`

### rank_and_filter
LLM receives the top 20 search results and ranks by relevance.
Returns a list of indices like `[3, 1, 7]` (most relevant first).
If LLM returns `[]` (completely unrelated products) → `ranked_products = []`
→ `format_recommendations` returns "couldn't find any products".

**Important:** When the LLM returns `[]` it means the search results are entirely
unrelated to the request. This is the signal to show "not available" instead of
random products. This is different from "no search results" which triggers `broaden_search`.

### Product Enrichment Subgraph
`subgraphs/product_enrichment.py`

```
fetch_reviews_node → fetch_specs_node → compute_score
```

`compute_score` calculates a final score for each product:
```
Rating quality   (40%) = product.rating / 5 × 0.40
Review trust     (30%) = min(rating_count / 1000, 1.0) × 0.30
Price fit        (30%) = based on price vs user's max_price budget
LLM rank bonus   (10%) = positional bonus (earlier in LLM ranking = higher bonus)
```
Returns top 9 in-stock products sorted by score.

### format_recommendations
Builds the final text response. Three possible openings:
- Over budget: "No headphones found under ₹2,000. Here's the closest option:"
- Partial match (LLM returned partial relevance): "Couldn't find an exact match. Here are the closest products:"
- Normal: "Here are my top 3 recommendations for 'headphones under 2000':"

Only the first 3 products are shown in text. The rest are available as cards
in the frontend (returned as `products` list in the API response).

---

## 9. Support Agent — Deep Dive

**File:** `agents/support_agent.py`

### Flow

```
classify_issue      ← LLM: identifies issue_type + order_id
    ↓
route_after_classify
    ├─ issue needs order_id but none provided → request_order_id (END)
    └─ has order_id or doesn't need one → assess_severity
                    ↓
assess_severity     ← pure code: fetches order from DB, gets value + status
                    ↓
lookup_policy_node  ← tool: fetches policy text for this issue_type
                    ↓
route_severity
    ├─ HIGH or MEDIUM → escalation_handler_node (creates ticket)
    └─ LOW → draft_resolution (general query, no ticket)
                    ↓
escalation_handler_node  ← subgraph
                    ↓
draft_resolution    ← LLM: writes the resolution response
                    ↓
format_response     ← pure code: appends ticket ID if not already in response
```

### classify_issue
LLM identifies:
- `issue_type`: `damaged_product`, `wrong_item`, `refund`, `cancellation`, `general_query`
- `order_id`: resolved from message or from `order_statuses` in session_context
- `details`: one-sentence description

The prompt includes `order_hint` built from session_context:
```
Known orders by status:
  delivered: ORD-001, ORD-005
  delayed: ORD-004
```
This allows "I want to cancel my delayed order" to auto-resolve to ORD-004 if there's
only one delayed order.

### assess_severity
Fetches the order from DB. Validates ownership (order must belong to this user_id).
Determines severity:
- `damaged_product`, `wrong_item`, `refund` + order_value ≥ 10000 → HIGH
- `damaged_product`, `wrong_item`, `refund` + order_value < 10000 → MEDIUM
- `cancellation` → MEDIUM
- `general_query` → LOW

Also stores `order_status` (e.g. "delivered", "processing") in state.
This is critical for draft_resolution to know whether cancellation is actually possible.

### Escalation Handler Subgraph
`subgraphs/escalation_handler.py`

```
check_user_history_node → assign_priority → create_ticket_node
```

`check_user_history_node`: checks if user is a repeat complainant (has multiple prior tickets).

`assign_priority`:
```
HIGH severity + repeat  → URGENT
HIGH severity           → HIGH
MEDIUM + repeat         → HIGH
MEDIUM                  → MEDIUM
LOW                     → LOW
```

`create_ticket_node`: creates a record in `support_tickets` table. Returns `ticket_id`.

### draft_resolution
LLM writes the empathetic customer-facing response. The prompt includes:
- `issue_type`, `issue_details`, `order_id`
- **`order_status`** — actual status from DB (e.g. "delivered")
- `policy_text` — the policy for this issue type from the `policies` table
- `ticket_info` — ticket ID and priority if a ticket was created

Key instruction in the prompt:
```
IMPORTANT: Apply the policy strictly based on the actual order status.
If the order is "delivered" or "cancelled", do NOT say it can be cancelled
or that you will proceed with cancellation — state clearly it is not eligible.
```
This prevents the hallucination where the LLM says "happy to cancel" for
a delivered order just because the cancellation policy text mentions refunds.

### Issue Types and Their Policies
Stored in the `policies` table. Issue types:
- `damaged_product`: replacement or refund, report within 72 hours with photo
- `wrong_item`: full replacement or refund, report within 72 hours
- `refund`: processed within 5-7 business days
- `cancellation`: allowed within 24 hours of placement or if still in processing
- `general_query`: generic support response

---

## 10. Database Schema

### System Tables (managed by the app)

**users**
```
user_id (PK), is_authenticated, email, name, auth_provider, created_at, last_login_at, metadata (JSONB)
```

**sessions**
```
session_id (PK), user_id (FK), created_at, last_active_at, is_active,
agent_last_used, message_count, ended_reason,
session_context (JSONB)   ← structured memory for the conversation
```

**messages**
```
message_id (PK), session_id (FK), role (user/assistant), content,
agent_name, intent, intent_confidence, latency_ms, langfuse_trace_id,
token_usage (JSONB), created_at
```

### Domain Tables (business data)

**orders**
```
order_id (PK), user_id, status, items (JSONB), carrier, tracking_number,
order_date, expected_delivery, order_value
```
Each user gets 5 sample orders auto-created on first login:
ORD-XXXXXX-001 (delivered), 002 (shipped), 003 (processing), 004 (delayed), 005 (delivered)

**products**
```
product_id (PK), name, category, price, actual_price, discount_percent,
brand, rating, rating_count, description, features (JSONB), in_stock, product_type
```
Real product data imported from Kaggle (Indian e-commerce dataset).
`product_type` is a classified field (mouse, keyboard, headphones, etc.) used for
fast exact-match search before falling back to name keyword search.

**policies**
```
policy_id (PK), issue_type (unique), policy_text, updated_at
```

**carrier_tracking**
```
tracking_id (PK), tracking_number (unique), carrier_name, current_status,
current_location, events (JSONB), estimated_delivery, last_updated
```

### Runtime Tables

**support_tickets**
```
ticket_id (PK), user_id, order_id, issue_type, priority, status (open/closed), created_at
```

**agent_runs** (for analytics)
```
run_id (PK), session_id, agent_name, intent, duration_ms, success,
total_tokens, cost_usd, created_at
```

---

## 11. Authentication System

**Google OAuth flow:**
1. Frontend opens Google OAuth popup
2. User logs in with Google
3. Frontend gets a Google ID token (JWT)
4. Frontend sends JWT in `Authorization: Bearer <token>` header on every request
5. `resolve_user()` in routes.py calls `google.oauth2.id_token.verify_oauth2_token()` to validate
6. Extracts `google_sub` (Google's permanent user ID), creates/updates user in DB
7. On first login → auto-creates 5 sample orders for the user

**Guest flow:**
- Frontend generates a UUID as `guest_id`
- Sent in request body as `guest_id` field
- Stored as anonymous user in `users` table
- Can only use the Product Agent (order/support require auth)

---

## 12. LangFuse Observability

**File:** `langfuse_helpers/tracing.py`

Every request gets a **trace** (top-level container) with nested **spans** and **generations**.

### Hierarchy
```
chat_request (trace)
  ├── intent_router (span)
  │     └── intent_router LLM call (generation)
  ├── order_agent (span)           ← if order intent
  │     ├── validate_input (span)
  │     ├── fetch_order_data (span)
  │     ├── analyze_order_status (generation)
  │     ├── shipment_tracking_subgraph (span)
  │     │     ├── get_carrier_info (span)
  │     │     ├── fetch_tracking_data (span)
  │     │     └── parse_eta (span)
  │     └── generate_response (generation)
  └── ... (similarly for product/support agents)
```

### Key functions
- `create_trace()` — called once per request in routes.py
- `create_span(trace_id, name, parent_observation_id, input_data)` — for non-LLM nodes
- `end_span(span, output_data)` — must be called after node completes
- `create_generation(trace_id, name, model, prompt, response, usage)` — for every LLM call
- `get_prompt(name, label, fallback)` — fetches prompt from LangFuse (with in-memory cache);
  falls back to hardcoded string if LangFuse is unavailable
- `compile_prompt(template, **kwargs)` — replaces `{{variable}}` in LangFuse templates
- `flush()` — called after every request to send all events

### Prompt Management
Prompts can be stored in LangFuse and fetched at runtime. Each LLM call:
1. Calls `get_prompt("prompt_name", label="production", fallback=hardcoded_string)`
2. If LangFuse has the prompt → uses it (allows changing prompts without code deploy)
3. If LangFuse is down or prompt not found → uses the hardcoded fallback string
4. Token usage and cost are recorded in every generation

### Auto-Scoring (`langfuse_helpers/scoring.py`)
After every request, 5 heuristic scores (0.0–1.0) are submitted to LangFuse:
- `answer_relevancy` — word overlap between message and response
- `faithfulness` — response contains order IDs / prices / policy references (per agent)
- `completeness` — word count appropriate for question vs command
- `task_completion` — agent-specific signals (order ID + status, price symbol, ticket ID)
- `hallucination` — detects invented order IDs, ticket IDs without valid format, etc.

---

## 13. Prometheus Metrics

**File:** `monitoring/metrics.py`

Four metrics exported at `/metrics`:

| Metric | Type | Labels | What it measures |
|---|---|---|---|
| `chat_requests_total` | Counter | `agent_used`, `status` | Total requests per agent |
| `chat_request_latency_seconds` | Histogram | `agent_used` | End-to-end latency |
| `llm_tokens_total` | Counter | `agent_used`, `token_type` | Input/output tokens consumed |
| `chat_errors_total` | Counter | `error_type`, `agent_used` | Errors by type and agent |

Token counting is done in a background task (3 second delay after request completes)
because LangFuse needs time to process the trace before its token data is queryable.

---

## 14. API Endpoints

### POST /chat
**Headers:** `Authorization: Bearer <google_jwt>`, `X-Session-ID: <uuid>`
**Body:** `{"message": "...", "guest_id": "..."}`
**Response:** `{"response", "session_id", "user_id", "agent_used", "intent", "intent_confidence", "is_authenticated", "products"}`

`products` is a list of product dicts returned when the product agent responds.
The frontend uses this to render product cards separately from the text response.

### GET /health
Returns `{"status": "healthy", "database": "connected"}`. Used by Docker health checks.

### GET /config
Returns `{"google_client_id": "..."}`. Frontend fetches this to initialise Google Sign-In
without hardcoding the client ID in HTML.

### GET /metrics
Prometheus scrape endpoint (auto-exposed by `prometheus_fastapi_instrumentator`).

---

## 15. Tools Layer

Tools are LangChain `@tool` decorated functions. They are the interface between
agent graph nodes and the actual data services.

**Order tools** (`tools/order_tools.py`):
- `fetch_order_data` — fetches one order by order_id from DB
- `fetch_all_orders_for_user` — fetches all orders for a user_id
- `fetch_tracking_data` — fetches carrier_tracking record by tracking_number

**Product tools** (`tools/product_tools.py`):
- `search_products_tool` — calls `mock_product_api.search_products(filters)`
- `fetch_ratings_tool` — fetches rating + rating_count for product IDs
- `fetch_specs_tool` — fetches features + description for product IDs

**Support tools** (`tools/support_tools.py`):
- `lookup_policy_tool` — fetches policy text for a given issue_type from `policies` table
- `check_user_history_tool` — counts prior support tickets for user_id + order_id
- `create_ticket_tool` — inserts a new record into `support_tickets` table

---

## 16. Session Lifecycle

1. First request: no `X-Session-ID` header → new session created → `session_id` returned in response
2. Frontend stores `session_id` and sends it as `X-Session-ID` on every subsequent request
3. `resolve_session()` loads the session, checks if it's still active (not expired)
4. Session expires after 30 minutes of inactivity (configurable via `SESSION_EXPIRY_MINUTES`)
5. Expired session → `is_active=False`, new session created automatically
6. History window: last 10 messages loaded per request (configurable via `HISTORY_WINDOW_SIZE`)

---

## 17. Important Design Decisions and Why

### Why LangGraph instead of a simple LLM call?
LangGraph gives explicit control over the flow through nodes and edges. Each step
(classify → fetch → analyze → respond) is a separate function that can be debugged,
logged, and observed independently in LangFuse. A single "do everything" LLM call
would be harder to trace and less reliable.

### Why Structured Memory (session_context) instead of raw history?
Raw history injection (`history[-8:]`) into every prompt costs 800-1200 tokens per request
(the history is all previous messages in full). `format_context()` reduces this to ~15 tokens.
`format_recent_messages(n=2)` adds ~150 tokens max. Total context: ~165 tokens vs ~1200.
This also prevents the LLM from being biased by old context (e.g. old support topic).

### Why does order listing show only ID and product name?
Users just need to recognise which order is which. Price, status, carrier, delivery date
are all available when they ask about a specific order. The listing is a navigation step.
Showing everything makes the response visually noisy.

### Why is there a pre-LLM hard check for greetings?
LLMs cannot reliably ignore context. If `session_context` says `topic=support_query`
and the user says "hi", the LLM sometimes classifies it as `support_query`. A hard-coded
check for known greeting words runs before the LLM and always returns `unknown`.

### Why does assess_severity fetch the order from DB?
Two reasons: (1) validate that the order belongs to this user before processing a complaint;
(2) get the real order_value to determine severity (HIGH vs MEDIUM). As of the latest update,
it also fetches `order.status` so `draft_resolution` knows whether the order is delivered
and can apply the cancellation policy correctly.

### Why is scoring heuristic-based instead of LLM-based?
LLM-based scoring would add latency and cost on every request. Heuristic scoring is
instant and free. It catches obvious issues (missing order IDs, fabricated ticket IDs,
response too short) without being perfect. LangFuse stores all traces anyway for manual review.

---

## 18. Configuration (config/settings.py)

All config is loaded from `.env` via Pydantic `BaseSettings`:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | required | LLM API key |
| `LLM_MODEL_NAME` | `llama-3.3-70b-versatile` | Which Groq model to use |
| `DATABASE_URL` | required | PostgreSQL connection string |
| `LANGFUSE_SECRET_KEY` | placeholder | LangFuse auth |
| `LANGFUSE_PUBLIC_KEY` | placeholder | LangFuse auth |
| `LANGFUSE_HOST` | `http://localhost:3000` | LangFuse server |
| `GOOGLE_CLIENT_ID` | placeholder | Google OAuth |
| `SESSION_EXPIRY_MINUTES` | 30 | How long before a session expires |
| `HISTORY_WINDOW_SIZE` | 10 | How many messages to load per request |
| `INTENT_CONFIDENCE_THRESHOLD` | 0.7 | Below this → ask_clarification |
| `SUPPORT_HIGH_VALUE_THRESHOLD` | 10000.0 | Above this → HIGH severity |
| `PRODUCT_SEARCH_CANDIDATES` | 20 | Max products passed to rank_and_filter |
| `PRODUCT_RECOMMENDATION_COUNT` | 8 | Max products returned to user |
| `PRODUCT_PRICE_BROADEN_FACTOR` | 1.5 | Price multiplier when broadening search |
| `GROQ_INPUT_COST_PER_MILLION` | 0.59 | USD per million input tokens |
| `GROQ_OUTPUT_COST_PER_MILLION` | 0.79 | USD per million output tokens |
| `INTENT_ROUTER_PROMPT_LABEL` | `production` | LangFuse prompt label for router |
| `ORDER_RESPONSE_PROMPT_LABEL` | `production` | LangFuse prompt label for order/product |
| `ORDER_ANALYSIS_PROMPT_LABEL` | `production` | LangFuse prompt label for order analysis |

---

## 19. Common Bugs Fixed and Why

### "hi" after support query routes to Support Agent
**Root cause:** LLM reads `session_context` with `topic=support_query` and classifies
"hi" as a support message even when instructed not to.
**Fix:** Hard-coded pre-LLM greeting check in `intent_router` node.

### "tell me more about the delayed one" → error
**Root cause:** `validate_input` had no way to resolve "the delayed one" without
the user typing the order ID. `order_id_found=False` → `error_response`.
**Fix:** Step 2 in `validate_input` reads `order_statuses` from session_context and
resolves status references to actual order IDs.

### "how many orders i placed" → error
**Root cause:** The keyword list in `validate_input` didn't include "HOW MANY" and "I PLACED".
**Fix:** Extended `show_all_keywords` list.

### "any options with better rating?" → wrong products
**Root cause:** `extract_preferences` had no session context in its prompt. Follow-up
query "better rating" gave null product_type → search returned random high-rated products.
**Fix:** Added `format_context()` + `format_recent_messages()` + follow-up instruction
to the `extract_preferences` prompt.

### Support agent says "happy to cancel" a delivered order
**Root cause:** `assess_severity` fetched `order_value` but not `order_status`.
`draft_resolution` LLM only saw the cancellation policy text (allows within 24h)
with no knowledge that the order was actually delivered.
**Fix:** `assess_severity` now stores `order_status` in state.
`draft_resolution` prompt includes actual status + explicit instruction to deny
cancellation for delivered/cancelled orders.

### "masala dosa" returns electronics products
**Root cause:** Product search for `product_type="other"` returns an empty list
→ `broaden_search` removes filters → search returns top-rated products by default.
Even when `rank_and_filter` LLM returned `[]` (no relevant products), the code
was overriding it with the top search results as "partial matches".
**Fix:** When `rank_and_filter` returns `[]`, `ranked_products` is set to `[]`.
`format_recommendations` then says "I couldn't find any products".
For completely unknown product types (not in product_type enum) → `product_type="other"`
→ the keyword path in `search_products` is taken → returns empty if no keyword match.

---

## 20. File-to-Responsibility Map (Quick Reference)

| File | Primary responsibility |
|---|---|
| `main.py` | FastAPI app setup, CORS, Prometheus, startup |
| `api/routes.py` | /chat endpoint, user/session resolution, orchestration |
| `api/schemas.py` | Pydantic models for request/response validation |
| `config/settings.py` | All environment config, single `settings` object |
| `agents/intent_router.py` | Classify intent, route to correct agent |
| `agents/order_agent.py` | Handle order tracking and status queries |
| `agents/product_agent.py` | Handle product search and recommendations |
| `agents/support_agent.py` | Handle complaints, refunds, cancellations |
| `subgraphs/escalation_handler.py` | Check user history, assign priority, create ticket |
| `subgraphs/shipment_tracking.py` | Fetch carrier tracking data and ETA |
| `subgraphs/product_enrichment.py` | Enrich products with ratings, specs, scoring |
| `tools/order_tools.py` | LangChain tools wrapping order DB queries |
| `tools/product_tools.py` | LangChain tools wrapping product DB queries |
| `tools/support_tools.py` | LangChain tools wrapping policy and ticket DB queries |
| `services/mock_order_api.py` | Raw DB query functions for orders/tracking |
| `services/mock_product_api.py` | Raw DB query functions for products |
| `services/mock_support_api.py` | Raw DB query functions for policies/tickets |
| `database/models.py` | All SQLAlchemy ORM models (8 tables) |
| `database/connection.py` | Engine, SessionLocal, Base, create_tables() |
| `utils/memory.py` | format_context, format_recent_messages, merge_context |
| `langfuse_helpers/tracing.py` | All LangFuse API calls (trace, span, generation, prompt) |
| `langfuse_helpers/scoring.py` | Heuristic auto-scoring after each request |
| `monitoring/metrics.py` | Prometheus counter/histogram definitions |
| `monitoring/logging_config.py` | Structured JSON logging to stdout + file |
