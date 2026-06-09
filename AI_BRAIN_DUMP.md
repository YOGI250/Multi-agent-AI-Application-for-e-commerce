# E-Commerce AI Support System — Complete AI Brain Dump

> **How to use this file**: Paste this entire document into ChatGPT or Claude as your
> first message. Then say "I have shared my project context. Help me with [your task]."
> It covers everything — architecture, every node in every agent, every design decision,
> every known limitation, and how to talk about them in a demo.

---

## 1. WHAT THIS PROJECT IS

A **multi-agent AI customer support chatbot** for an e-commerce store. Users type
messages into a single chat interface. Depending on what they ask, one of three
specialist AI agents handles the request. A top-level Intent Router decides which
agent to call.

**Three capabilities:**
| Agent | What it handles |
|---|---|
| Order Agent | Order status, tracking, delivery dates, "how many orders do I have?" |
| Product Agent | Product search, recommendations, comparisons, filters by price/brand/rating |
| Support Agent | Complaints, refunds, damaged items, wrong items, cancellations |

**The chat interface:** Plain HTML/JS served by FastAPI at `http://localhost:8000/app`.
Users can log in with Google OAuth or browse as a guest. Guests can only search
products — they cannot access orders or raise support tickets.

---

## 2. TECH STACK — EVERY TOOL AND WHY

| Layer | Technology | Why chosen |
|---|---|---|
| Web framework | FastAPI (Python) | Async, fast, auto OpenAPI docs |
| AI orchestration | LangGraph (StateGraph) | Gives explicit control over agent flow as a graph |
| LLM | Groq API — `llama-3.3-70b-versatile` | Free tier, very fast inference |
| LLM client | LangChain `ChatGroq` | Standard interface, pluggable |
| Database | PostgreSQL via SQLAlchemy ORM | Relational, JSONB for flexible fields |
| Auth | Google OAuth 2.0 (JWT verify) | Standard, no password management |
| Observability | LangFuse | Tracing every LLM call, prompt versioning, LLM-as-judge eval |
| Metrics | Prometheus + Grafana | Real-time request/token/error dashboards |
| Log aggregation | Loki + Promtail | Ships container logs to Grafana |
| Frontend | Vanilla HTML/JS/CSS | No build step, works in any browser |
| Deployment | Docker Compose (local) + Helm (Kubernetes) | Reproducible, one-command startup |
| CI/CD | GitHub Actions (6 stages) | Lint → test → eval → Docker build → push → k8s deploy |

---

## 3. HIGH-LEVEL ARCHITECTURE — THE REQUEST FLOW

```
User message
    │
    ▼
FastAPI /chat endpoint  (api/routes.py)
    │
    ├── Auth: verify Google JWT or create anonymous user
    ├── Session: load/create session, load conversation history
    ├── LangFuse: create root trace
    │
    ▼
Intent Router Graph  (agents/intent_router.py)
    │
    ├── Node 1: intent_router (LLM)
    │       Classifies as: order_query / product_query / support_query / unknown
    │       Pre-checks (no LLM): greetings → unknown, pending support + order ID → support
    │
    ├── Edge: route_to_agent
    │       If confidence < threshold → ask_clarification
    │       If guest + not product_query → access_denied
    │
    ├── Node 2: run_order_agent → calls order_agent graph
    ├── Node 3: run_product_agent → calls product_agent graph
    ├── Node 4: run_support_agent → calls support_agent graph
    ├── Node 5: ask_clarification (no LLM — canned response)
    ├── Node 6: access_denied (no LLM — canned response)
    └── Node 7: fallback_response (no LLM — canned response)
    │
    ▼
FastAPI /chat endpoint (back in routes.py)
    │
    ├── Save messages to DB (users, sessions, messages tables)
    ├── Record metrics (Prometheus counters)
    ├── LangFuse: update trace with session/user, score response
    ├── Return JSON: {response, products, session_id, agent_used}
    │
    ▼
Frontend renders response + product cards (if any)
```

---

## 4. THE INTENT ROUTER — DETAILED

File: `agents/intent_router.py`

### State (RouterState)
Every LangGraph graph has a TypedDict state that flows through all nodes:
```
message, user_id, session_id, history, is_authenticated,
intent, confidence, reason, response, agent_used, products,
session_context, langfuse_trace_id, langfuse_parent_span_id,
total_input_tokens, total_output_tokens
```

### Node 1: intent_router (LLM)

Two deterministic pre-checks before any LLM call:
1. **Greeting detection** — if message is in `_GREETINGS` set (hi, hello, hey, etc.)
   or starts with a help-prefix → set intent=unknown, skip LLM entirely.
   Why: prevents "hello" from being classified as order_query because user's last
   message was about an order.

2. **Pending support context** — scans last 8 messages for the phrase
   "i'll need your order id". If found AND current message contains `ORD-XXXXX`
   → force intent=support_query, skip LLM.
   Why: prevents the user providing an order ID being routed to order_agent when
   they're in the middle of a support conversation.

After pre-checks: LLM call with recent history + session_context → JSON with
intent/confidence/reason.

### Edge: route_to_agent
Logic in `route_to_agent()`:
- `confidence < settings.intent_confidence_threshold` → ask_clarification
- Guest user + intent ≠ product_query → access_denied
- order_query → run_order_agent
- support_query → run_support_agent
- product_query → run_product_agent
- anything else → fallback_response

### session_context — the cross-agent memory
A small dict (`session_context`) flows through all agents and is persisted
in the `sessions` table JSONB column. Each agent reads and writes it.

**Keys written by each agent:**
- Intent Router: passes it through unchanged
- Product Agent: `topic="product_query"`, `last_product_type`, `last_max_price`, `products_shown`
- Order Agent: `topic="order_query"`, `order_id`, `order_count`, `order_statuses`
- Support Agent: `topic="support_query"`, `issue_type`, `order_id`, `ticket_id`

**Why it matters:**
- Product follow-up ("what about under 2000") → agent reads `last_product_type` from context
- Order follow-up ("when will I get the refund") → agent reads `order_id` from context even if set by support_agent
- Support agent routing ("my delivered order") → agent reads `order_statuses` to resolve which order_id

---

## 5. ORDER AGENT — DETAILED

File: `agents/order_agent.py`

### Graph: 8 nodes

```
validate_input
    │
    ├── (order_id found?) → fetch_order_data_node
    └── (not found)       → error_response → END
                                │
                        order_tool_node (ToolNode)
                                │
                        process_order_result
                                │
                    ├── (data found?) → analyze_order_status
                    └── (not found)   → error_response → END
                                            │
                                    shipment_tracking_node (subgraph)
                                            │
                                    generate_response (LLM for single order)
                                    OR Python-built response (for all orders list)
                                            │
                                           END
```

### Node 1: validate_input (pure code — 4-step cascade)

**Step 1**: Regex for `ORD-[A-Z0-9-]+` in message → direct match.

**Step 2**: `order_statuses` from session_context (dict: `{status: [order_ids]}`).
If user says "the delayed one" and there's exactly one delayed order → resolve it.

**Step 3**: Pattern regex for "show all orders" intent:
- "show me orders", "list the orders", "my orders", "order status", "how many orders", etc.

**Step 4**: Session context fallback:
- If `ctx.get("order_id")` exists → use it directly (works for support→order handoff)
- If no order ID but `ctx.get("topic")` in ("order_query", "support_query") → wants_all_orders
This is what makes "when will I get the refund" work — support_agent wrote `order_id` to context.

### Node 2: fetch_order_data_node + order_tool_node + process_order_result

Tool node pattern (used in all 3 agents):
1. Preparation node builds an `AIMessage` with `tool_calls` in it.
2. `ToolNode` (LangChain) executes the tool and puts result in a `ToolMessage`.
3. Processing node reads the `ToolMessage` and updates state.

Why this 3-node pattern instead of direct function call?
LangGraph's `ToolNode` handles the message routing correctly and is the standard
pattern. It also enables LangFuse span instrumentation around each tool call.

Two tools available:
- `fetch_order_data(order_id)` → returns single order dict
- `fetch_all_orders_for_user(user_id)` → returns list of all orders

Security check in `process_order_result`: verifies `order.user_id == user_id` —
prevents user A from seeing user B's order if they somehow guess an order ID.

### Node 3: analyze_order_status (LLM)

LLM call to classify order situation:
```json
{"issue_type": "delayed/on_track/delivered/cancelled/processing", "summary": "one sentence"}
```

### Node 4: shipment_tracking_node (subgraph)

Calls `shipment_tracking_subgraph` → 3 nodes:
- `get_carrier_info` → extract carrier + tracking_number from order data
- `fetch_tracking_data_node` → query `carrier_tracking` table for real-time location/ETA
- `parse_eta` → extract eta, current_location, events list

### Node 5: generate_response

**Case 1 — All orders list:**
Built entirely in Python, no LLM. Smart filtering:
- "not delivered" → filter by status ≠ delivered
- "shipped" → filter by shipped/in_transit
- "delayed" → filter delayed/late
- etc.
Product name lookup: user can say "status of my laptop stand" → iterates all orders, matches item name keywords.
Wants-detail detection: if user said "detailed information" + multiple orders → guide them to pick one.

**Case 2 — Single order:**
LLM generates natural language response using order data + tracking + analysis.

### Node 6: error_response

Two cases:
- Has order_id but not found → "I couldn't find order ORD-XXXX..."
- No order_id at all → "I need your order ID. Type 'show my orders'..."

---

## 6. PRODUCT AGENT — DETAILED

File: `agents/product_agent.py`

### Graph: 9 nodes

```
extract_preferences (LLM)
        │
search_products_node → product_tool_node → process_search_result
                                                    │
                        ┌───────────────────────────┤
                        │ (results?)                │
                  rank_and_filter (LLM)      broaden_search (code)
                        │                           │
             product_enrichment_node          search_products_node (retry)
             (subgraph: reviews+specs+score)
                        │
              format_recommendations          no_results_response
                        │
                       END
```

### Node 1: extract_preferences (LLM)

Purpose: convert user message to structured filters JSON:
```json
{
  "category": "Electronics",
  "product_type": "headphones",
  "max_price": 2000,
  "min_price": null,
  "brand": "boAt",
  "min_rating": 4.0
}
```

**Three-layer product_type resolution (applied in this order):**

**Layer 1 — LLM extraction**: LLM reads message + last 4 user messages and extracts filters.
Also has RULE 1 (explicit product name → ignore history) and RULE 2 (vague follow-up → use history).

**Layer 2 — Normalization dict**: After LLM, fix known plural/variant forms.
```python
"smartwatches" → "smartwatch", "mice" → "mouse", "geyser" → "water_heater", etc.
```
Why: LLM sometimes returns plural forms or regional synonyms that don't match the DB column.

**Layer 3 — KEYWORD_MAP (deterministic override)**: Checked against the user message itself.
If the message contains a keyword, it overrides whatever the LLM returned.
```python
KEYWORD_MAP = [
    ("smart watches", "smartwatch"),   # multi-word first — must come before "watch"
    ("ro water purifier", "water_purifier"),
    ("mouse pad", "mousepad"),         # must come before "mouse"
    ("laptop stand", "stand"),         # must come before "laptop"
    ("geyser", "water_heater"),
    ("mice", "mouse"),                 # must come before "mouse"
    ...
    ("fan", "fan"),                    # single-word entries come LAST
    ("pen", "pen"),
    ("ram", "ram"),
]
```

**Why multi-word entries come before single-word?**
"laptop stand" would match the "laptop" single-word entry first if "laptop" came
before "laptop stand". Multi-word first prevents early false matches.

**Word-boundary matching** (`_kw_match` function):
```python
def _kw_match(msg: str, kw: str) -> bool:
    if " " in kw:
        return kw in msg               # multi-word: plain substring
    return bool(re.search(r"\b" + re.escape(kw) + r"s?\b", msg))
```
The `\b...\b` word-boundary regex prevents:
- "pen" matching "spend" or "open"
- "fan" matching "fantastic"
- "ram" matching "programmable"

The `s?` at the end handles regular plurals: "fans" → "fan", "rams" → "ram".
Does NOT cover irregular plurals (watch→watches, mouse→mice) — those get explicit
entries in KEYWORD_MAP.

**Layer 4 — Session context fallback** (only if both LLM and KEYWORD_MAP returned nothing):
```python
last_ptype = (state.get("session_context") or {}).get("last_product_type")
```
Used for genuinely vague follow-ups like "under 2000" when no product is named.

### Node 2-3: search → tool → process

Same tool-node pattern as order agent.
`search_products_tool` calls `services/mock_product_api.py:search_products()`.

**Product search logic** (in `services/mock_product_api.py`):
1. Apply category, price, brand, rating filters to SQLAlchemy query.
2. If `product_type` given and ≠ "other":
   - Try exact match on `products.product_type` column (case-insensitive).
   - Also try `product_type + "s"` and `product_type.rstrip("s")` to handle plural/singular.
   - If no results → **name fallback**: `Product.name.ilike(f"%{singular}%")`.
     This catches colloquial terms: "geyser" won't find `water_heater` by type but
     may find "Bajaj Geyser 10L Water Heater" in the name column.
3. Return up to 20 products ordered by rating descending.

**Why `or_()` in SQLAlchemy?**
```python
query.filter(or_(
    Product.product_type.ilike(product_type),
    Product.product_type.ilike(singular),
    Product.product_type.ilike(singular + "s"),
))
```
`or_()` is SQLAlchemy's way of building `WHERE (a OR b OR c)`. Without it, you'd
get `WHERE a AND b AND c` which would never match.

### Node 3-edge: broaden_search (3 attempts before giving up)

When search returns 0 results, broaden_search is called with `attempts` counter:

**Attempt 0** (first retry): Drop the loosest constraint in order:
  - brand → max_price → min_rating → category
  - Why category last: KEYWORD_MAP already locked in the right product_type, so
    category contamination (e.g. wrong category from earlier search bleeding in)
    is more common than a wrong product_type at this stage.

**Attempt 1** (second retry): Drop product_type, restore original category.
  Handles the case where LLM returned an invalid type (e.g. "office" for "pen").

**Attempt 2** (third retry): Drop everything — pure keyword/text search.

After 3 failed attempts → `no_results_response`.

### Node 4: rank_and_filter (LLM)

Quality gate: LLM sees the 20 raw search results and ranks/filters them.
Returns a JSON array of 1-based indices: `[3, 1, 7, 2]`.

**Why strict prompt?**
Original prompt said "Return [] ONLY if entirely unrelated" → LLM returned fans for
an air conditioner request. New prompt:
```
STRICT: Do NOT include a product just because it is vaguely related.
If the user asked for an air conditioner, do not return fans or air purifiers.
Return [] if none of the listed products are genuinely what the user asked for.
When in doubt, return [].
```

**Rich request string**: If the user message is a vague follow-up ("under 2000"),
the agent builds a richer string: `"headphones — under 2000"` so rank_and_filter
LLM understands what product type is being filtered.

**Only LLM-selected products kept**: After ranking, we do NOT append unranked
products. Adding all unranked ones surfaces wrong results when search base is
contaminated (e.g. broad category search pulling in mousepads when asking about geysers).

### Node 5: product_enrichment_node (subgraph)

Calls `product_enrichment_subgraph` → 3 nodes:
- `fetch_reviews` → extract rating + rating_count (already in search results, no extra DB hit)
- `fetch_specs` → calls `get_specs(product_ids)` → fetches `description` and `features` JSONB
- `compute_score` → weighted scoring formula:

```
Rating quality    40%   rating / 5 * 0.40
Review trust      30%   min(rating_count / 1000, 1.0) * 0.30
Price fit         30%   (1.0 if no budget; 0 if over budget; 0.5-1.0 if within budget)
LLM rank bonus    10%   positional bonus preserving LLM's relevance order
```

Returns top 9 in-stock products sorted by score.

### Node 6/7: format_recommendations / no_results_response

`format_recommendations`: Builds the text response showing top 3 products with
name, price, rating, brand, and top 3 features. If more than 3 results, appends
"...and X more products available" (shown as cards in the frontend).

`no_results_response`: Shows an accurate catalog of what the store actually has —
40+ product types organized by category. This message was carefully tuned to match
what's actually in the database.

**Products proven to work reliably (safe for demo):**
laptops, headphones, keyboards, mice, speakers, smartwatches, fans, mixers,
kettles, irons, vacuum cleaners, water heaters (via "geyser" too), air purifiers,
water purifiers, room heaters, washing machines, cameras, tablets, pendrives, SSDs,
RAM, hard disks, memory cards, chargers, cables, USB hubs, routers, mousepads,
laptop bags, phone cases, monitors, webcams, trimmer, microwave, pens, notebooks.

**Products with known limitations:**
- "TV" or "television": ~391 products in DB are tagged `product_type="other"` (importer
  couldn't classify them). Searching "TV" may or may not work depending on whether the
  LLM guesses "tv" as type and name fallback finds them.
- "refrigerator", "AC", "air conditioner": not in the catalog at all. Will get
  the no-results message, which is correct behavior.

---

## 7. SUPPORT AGENT — DETAILED

File: `agents/support_agent.py`

### Graph: 9 nodes

```
classify_issue (LLM)
        │
        ├── (needs order ID + none given) → request_order_id → END
        └── (order ID present or not needed) → assess_severity
                        │
                lookup_policy_node → support_tool_node → process_policy_result
                                                                │
                            ┌───────────────────────────────────┤
                            │ (severity HIGH/MEDIUM)            │ (severity LOW)
                escalation_handler_node (subgraph)       draft_resolution (LLM)
                            │
                    draft_resolution (LLM)
                            │
                    format_response (code)
                            │
                           END
```

### Node 1: classify_issue (LLM)

Classifies into: `damaged_product / wrong_item / refund / cancellation / general_query`

Also extracts `order_id` from the message or from session_context.

**Critical**: explicit order ID regex always wins over LLM to avoid history confusion.
Support agent writes `topic="support_query"` and `order_id` to session_context so
the order agent can later pick it up (the "when will I get the refund" fix).

### Edge: route_after_classify

`ORDER_REQUIRED_ISSUES = {"damaged_product", "wrong_item", "refund", "cancellation"}`

If issue requires an order ID but none found → request_order_id.
The response text contains the exact phrase `"i'll need your order id"` which the
intent router's `_has_pending_support_context()` watches for.

### Node 2: assess_severity (pure code)

Fetches order from DB to get real order value and status.
Severity rules:
- damaged_product / wrong_item / refund: HIGH if order_value ≥ threshold, else MEDIUM
- cancellation: always MEDIUM
- general_query: always LOW

Also validates order ownership — if order not found under this user, clears `order_id`.

### Node 3: lookup_policy_node + tool + process

Tool: `lookup_policy_tool(issue_type)` → queries the `policies` table.
Policy table stores text like "Damaged items: contact within 7 days, full refund..."
LLM gets this policy in the draft_resolution prompt.

### Node 4: escalation_handler_node (subgraph)

Calls `escalation_handler_subgraph` → 2 nodes:
- `check_user_history`: queries `support_tickets` table. Returns `is_duplicate`, `days_open`, `existing_ticket_id`
- `create_ticket`: if not duplicate → INSERT new ticket; if duplicate → skip creation

**Duplicate handling by age:**
- 0 days → "Your ticket was recently created, allow 48 hours"
- 1-3 days → "Being actively reviewed, update within 24 hours"
- 4+ days (overdue) → Strong apology, "escalated to senior team"

**Why exact response strings for duplicates?**
LLM was generating inconsistent duplicate responses (sometimes saying "new ticket
created" even for duplicates). Exact pre-computed strings in the prompt prevent deviation.

### Node 5: draft_resolution (LLM)

Writes the final response using:
- Policy text
- Ticket info
- Duplicate instruction (exact string or "standard empathetic response")
- Actual order status → prevents saying "we will cancel your order" if it's already delivered

### Node 6: format_response (pure code)

If ticket ID was created but LLM forgot to mention it → appends it.
Makes ticket ID presence guaranteed in the final output.

---

## 8. DATABASE SCHEMA — ALL 8 TABLES

File: `database/models.py`

### System tables (manage users, sessions, conversation)

**users**: `user_id (PK), is_authenticated, email, name, auth_provider, created_at, last_login_at, metadata_`
- Anonymous users get UUID, `auth_provider="anonymous"`
- Google users get their email/name, `auth_provider="google"`

**sessions**: `session_id (PK), user_id (FK), created_at, last_active_at, is_active, agent_last_used, message_count, ended_reason, session_context (JSONB)`
- `session_context` JSONB column is the cross-agent memory dict (topic, order_id, last_product_type, etc.)
- Sessions expire after 30 minutes of inactivity (configurable)
- Index on `(user_id, is_active)` for fast lookup of active session per user

**messages**: `message_id, session_id (FK), role, content, agent_name, intent, intent_confidence, created_at, token_usage (JSONB), latency_ms, langfuse_trace_id, error`
- Every user message and assistant response is stored here
- `langfuse_trace_id` links DB record to LangFuse trace

### Domain tables (business data)

**orders**: `order_id (PK), user_id (FK), status, items (JSONB), carrier, tracking_number, order_date, expected_delivery, order_value`
- `items` is JSONB array of product names
- `status` values: "processing", "shipped", "in_transit", "delayed", "delivered", "cancelled"

**products**: `product_id (PK), name, category, price, actual_price, discount_percent, brand, rating, rating_count, description, features (JSONB), in_stock, product_type`
- 1,351 products imported from Kaggle Amazon dataset
- `product_type` is the classification field used for search (mouse, keyboard, headphones, etc.)
- ~391 products have `product_type="other"` (importer couldn't classify them — known gap)
- `features` is JSONB array of bullet point strings

**policies**: `policy_id, issue_type (unique), policy_text, updated_at`
- One row per issue type (damaged_product, wrong_item, refund, cancellation, general_query)
- LLM reads these as context for support responses

**carrier_tracking**: `tracking_id, tracking_number (unique), carrier_name, current_status, current_location, events (JSONB), estimated_delivery, last_updated`
- Mock tracking data for demo orders
- `events` is JSONB array of `{date, location, event}` objects

### Runtime tables (operational)

**support_tickets**: `ticket_id, user_id (FK), order_id (FK), issue_type, priority, status, created_at`
- Created by escalation_handler_subgraph when priority is HIGH/MEDIUM
- Used to detect duplicates (user filing same complaint twice)
- `priority` field holds "HIGH" or "MEDIUM" — **there is no `severity` or `description` field**
- `ticket_id` is a UUID (e.g. `05B09C4C-...`) included in the agent's final response text

**agent_runs**: `run_id, session_id, message_id, agent_name, intent, duration_ms, success, total_tokens, cost_usd, created_at`
- Every agent invocation recorded for analytics
- Used to compute cost per request (Groq pricing × token count)

---

## 9. SERVICES / MOCK APIs

These are Python modules that abstract database access. They are called by tools,
not by agents directly. This separates "how to query data" from "what to do with data".

**`services/mock_product_api.py`**:
- `search_products(filters)` — multi-filter query with product_type + name fallback
- `get_specs(product_ids)` — batch fetch description + features for enrichment

**`services/mock_order_api.py`**:
- `get_order(order_id)` — single order fetch
- `get_orders_by_user(user_id)` — all orders for user

**`services/mock_support_api.py`**:
- `lookup_policy(issue_type)` — get policy text
- `check_user_history(user_id, order_id)` — detect duplicate tickets, get days_open
- `create_ticket(user_id, issue_type, priority, order_id)` — insert new ticket

---

## 10. TOOLS — THE LANGGRAPH BRIDGE

File: `tools/` directory

Tools are LangChain `@tool` decorated functions. They are the bridge between
agent nodes and service functions. Each tool:
1. Is passed to `ToolNode` which handles message routing
2. Is instrumented with LangFuse span tracking

**Product tools**: `search_products_tool`, `fetch_specs_tool`
**Order tools**: `fetch_order_data`, `fetch_all_orders_for_user`, `fetch_tracking_data`
**Support tools**: `lookup_policy_tool`, `check_user_history_tool`, `create_ticket_tool`

---

## 11. OBSERVABILITY STACK

### LangFuse

**Two environments:**
- **Local/Docker Compose**: `http://localhost:3000` — login: `admin@langfuse.com` / `password`
- **CI/Cloud**: `https://cloud.langfuse.com` — credentials stored in GitHub Secrets (`LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`)

The k8s deployment uses cloud.langfuse.com (configured via k8s Secret `ecommerce-secrets`).
All LangFuse traces from the live k8s app appear in the cloud account.

**Tracing**: Every `/chat` request creates a root trace. Each agent node that makes
an LLM call creates a "generation" span. Tool calls and subgraphs create spans.
Token usage (input + output) is recorded at every generation.

**Prompt management**: All agent system prompts are versioned in LangFuse.
Agents call `get_prompt("prompt_name", label="production")` — if LangFuse is
reachable, it fetches the latest version; if not, it uses the `fallback` string
hardcoded in the Python file. This means you can update prompts without redeploying.

**Scoring**: After every response, 3 automatic scores are computed:
- `answer_relevancy` — is the response relevant to the question?
- `task_completion` — did the agent complete the task?
- `correctness` — is the information factually correct?

**Evaluation dataset**: `ecommerce-eval-dataset` — always exactly **15 items** (CI full-refreshes it).
Each CI run that completes Stage 3 adds one new experiment row (accumulates over time).
Running locally: `python eval/run_eval.py` sends cases to the live API and records scores.

### Prometheus + Grafana (http://localhost:3001)
Default login: `admin` / password from `.env`

**Custom metrics exposed at `/metrics`:**
- `chat_requests_total` (counter, labels: agent_used, status)
- `chat_request_latency_seconds` (histogram, labels: agent_used)
- `llm_tokens_total` (counter, labels: agent_used, token_type)
- `chat_errors_total` (counter, labels: error_type, agent_used)

**Grafana dashboard panels:**
- Request volume over time (by agent)
- Agent selection distribution (pie chart)
- Average latency per agent
- Error rate
- LLM token consumption
- Live logs from Loki (filterable by agent_used, session_id, request_id)

### Loki + Promtail
Promtail ships container logs to Loki. JSON log fields (`level`, `agent_used`,
`session_id`, `request_id`) are extracted as Loki labels for filtering in Grafana.
In Docker Compose, Promtail reads the container log file.
In Kubernetes, Promtail runs as a DaemonSet and reads `/var/log/pods/`.

---

## 12. AUTHENTICATION FLOW

1. Frontend loads Google Sign-In button using `GOOGLE_CLIENT_ID` from `/api/config`.
2. User clicks Sign In → Google returns a JWT ID token.
3. Frontend sends JWT in `Authorization: Bearer <token>` header with every request.
4. FastAPI middleware (`api/routes.py`) calls `google.oauth2.id_token.verify_oauth2_token()`.
5. If valid → resolve or create User record with email/name.
6. If invalid / no token → create anonymous user, `is_authenticated=False`.
7. `is_authenticated` flows into RouterState and gates order/support access.

**Docker/network fallback**: If the Google certificate endpoint is unreachable
(happens in isolated Docker networks), falls back to local JWT decode.

---

## 13. SESSION MANAGEMENT

- Sessions are identified by `X-Session-ID` header (UUID).
- First request: new session + new user created.
- Subsequent requests: session looked up, conversation history loaded from `messages` table.
- Session expires after 30 minutes of inactivity → marked `is_active=False`.
- `session_context` JSONB is loaded from `sessions` table, passed to agents, agents update it, saved back.
- The frontend stores `session_id` from the response body in localStorage.

---

## 14. CI/CD PIPELINE

6 GitHub Actions stages (`.github/workflows/`):

| Stage | Runner | What |
|---|---|---|
| 1 — Lint & Format | ubuntu-latest | flake8 (style) + black --check (formatting) |
| 2 — Unit Tests | ubuntu-latest | pytest with ≥80% coverage (`--cov-fail-under=80`) |
| 3 — LLM Evaluation | ubuntu-latest | Real GROQ calls + LangFuse cloud — 15 eval cases, saves HTML/JSON report artifact |
| 4 — Docker Build | ubuntu-latest | Builds image (does not push) |
| 5 — Push to Registry | ubuntu-latest | Pushes to Docker Hub (main branch only) |
| 6 — Deploy to K8s | self-hosted | helm upgrade + smoke test /health + auto-rollback |

Stage 6 runs on a **self-hosted runner** on the local machine where the **kind** (Kubernetes IN Docker) cluster runs.
On smoke test failure, `helm rollback` reverts to previous Helm release automatically.

**Stage 6 gate**: Only runs when `vars.DEPLOY_ENABLED == 'true'` is set in GitHub repo variables (manual toggle).

**Stage 3 details** (3 steps inside the eval job):
1. `seed_eval_db.py` — creates eval user + sample orders in Neon DB (idempotent)
2. `seed_dataset.py` — **full refresh**: deletes all existing LangFuse dataset items then re-adds 15 from `eval/dataset.json`; prevents item accumulation from repeated CI runs
3. `run_eval.py` — sends each case to the live FastAPI app (using `NEON_DATABASE_URL`), scores with LLM-as-judge, saves reports to `eval/reports/`; results visible in LangFuse cloud → Datasets → ecommerce-eval-dataset

**Token cost warning**: Stage 3 uses real GROQ tokens (2-3 LLM calls × 15 cases ≈ 60+ calls).
Do NOT push to main immediately before a demo — running CI can exhaust the 100k daily token limit.

---

## 15. KUBERNETES / HELM DEPLOYMENT

File: `helm/` directory

**Cluster**: kind (Kubernetes IN Docker) — cluster name `ecommerce`, context `kind-ecommerce`.
Pods run inside Docker containers. They cannot be reached directly from a browser
— you need `kubectl port-forward` tunnels to connect.

**Port-forward setup (required after every restart)**:
```bash
# One-liner to forward all services (auto-reconnects on disconnect):
scripts/start.sh

# Or install as a persistent systemd service (survives login):
scripts/setup-autostart.sh
```
`scripts/start.sh` forwards:
- FastAPI: `localhost:8000` → svc/ecommerce-fastapi port 80
- Grafana: `localhost:3001` → svc/ecommerce-grafana port 3001
- Prometheus: `localhost:9090` → svc/ecommerce-prometheus port 9090

**Helm chart deploys:**
- FastAPI app (Deployment + Service)
- Prometheus (with scrape config pointing at `/metrics`)
- Grafana (with pre-provisioned dashboards as ConfigMaps)
- Loki (log aggregation)
- Promtail (DaemonSet — reads `/var/log/pods/` and ships to Loki)

**Not in Helm (runs externally):**
- LangFuse → uses `https://cloud.langfuse.com` (hosted); credentials in k8s Secret
- PostgreSQL → Neon serverless DB (cloud); connection string in k8s Secret

**Neon cold-start**: Neon scales to zero after ~5 min idle. First DB connection after
idle takes 5-10 seconds. Helm probes have `timeoutSeconds: 10` to tolerate this.

The app reads all config from environment variables (set in Helm `values.yaml` →
Kubernetes `Secret` / `ConfigMap`). `config/settings.py` loads via Pydantic Settings.

**Secrets are NOT in Helm values** — they live in a k8s Secret named `ecommerce-secrets`
created by CI Stage 6 (or manually patched). To patch a secret without redeploying:
```bash
NEW_VAL=$(echo -n "actual-value" | base64)
kubectl patch secret ecommerce-secrets -n ecommerce \
  --type='json' -p="[{\"op\":\"replace\",\"path\":\"/data/groq-api-key\",\"value\":\"$NEW_VAL\"}]"
kubectl rollout restart deployment/ecommerce-fastapi -n ecommerce
```

---

## 16. DOCKER COMPOSE SERVICES

All defined in `docker-compose.yml`:

| Service | Container name | Port |
|---|---|---|
| FastAPI app | ecommerce-fastapi | 8000 |
| PostgreSQL | ecommerce-postgres | 5432 |
| Prometheus | ecommerce-prometheus | 9090 |
| Grafana | ecommerce-grafana | 3001 |
| Loki | ecommerce-loki | 3100 |
| Promtail | ecommerce-promtail | — |
| LangFuse server | langfuse-server | 3000 |
| LangFuse worker | langfuse-worker | — |
| LangFuse DB (Postgres) | langfuse-db | — |
| LangFuse MinIO (object storage) | langfuse-minio | — |
| LangFuse Redis | langfuse-redis | — |
| LangFuse ClickHouse (analytics) | langfuse-clickhouse | — |

**The FastAPI container** (`Dockerfile`):
- Base: Python 3.12-slim
- COPY requirements.txt → pip install → COPY code
- Entrypoint: `entrypoint.sh` → runs `alembic upgrade head` then `uvicorn main:app`
- Health check: GET `/health` must return 200

**Rebuild ritual:**
```bash
docker compose build fastapi          # re-copies code into image
docker compose up -d fastapi          # restart container with new image
docker compose ps                     # verify (healthy) status
docker compose logs fastapi --tail=50 # check startup logs if needed
```

---

## 17. OPERATIONAL RUNBOOK (THINGS THAT BREAK AND HOW TO FIX THEM)

### "Cannot connect to server" in the UI

**Root cause A — GROQ daily token limit exhausted (most common)**
GROQ free tier: 100,000 tokens/day per key. Each `/chat` request = 2-3 LLM calls.
Running CI or automated tests before a demo can exhaust this.

Fix:
1. Create a new key at console.groq.com
2. Patch the k8s secret (do NOT add a trailing newline — that breaks the key):
```bash
NEW_KEY="gsk_xxxx..."
NEW_B64=$(printf '%s' "$NEW_KEY" | base64 -w 0)
kubectl patch secret ecommerce-secrets -n ecommerce \
  --type='json' -p="[{\"op\":\"replace\",\"path\":\"/data/groq-api-key\",\"value\":\"$NEW_B64\"}]"
kubectl rollout restart deployment/ecommerce-fastapi -n ecommerce
```
3. Also update the `GROQ_API_KEY` GitHub Secret — otherwise next CI deploy reverts to old key.

Limit resets at midnight UTC.

**Root cause B — port-forward tunnel died**
kubectl port-forward is a temporary tunnel — it dies when the terminal closes.
Fix: run `scripts/start.sh` (has an auto-reconnect loop) or install the systemd
service with `scripts/setup-autostart.sh`.

**Root cause C — Neon DB cold start**
Neon scales to zero after ~5 min idle. First request after idle takes 5-10s.
This is normal — the second request will be fast. No fix needed.

### Google Sign-In "invalid_client" error

Root cause: trailing newline in the `google-client-id` k8s secret.
Fix: patch with `printf '%s'` (NOT `echo`) to avoid adding `\n`:
```bash
ID="your-client-id.apps.googleusercontent.com"
B64=$(printf '%s' "$ID" | base64 -w 0)
kubectl patch secret ecommerce-secrets -n ecommerce \
  --type='json' -p="[{\"op\":\"replace\",\"path\":\"/data/google-client-id\",\"value\":\"$B64\"}]"
kubectl rollout restart deployment/ecommerce-fastapi -n ecommerce
```

### CI Stage 6 (Deploy) fails

Stage 6 only runs if `vars.DEPLOY_ENABLED == 'true'` and on the main branch.
Common failure modes:
- Smoke test timeout → check `kubectl describe pod -l component=fastapi -n ecommerce`
- Image not loading into kind → kind cluster may have restarted; re-pull and load
- Helm secret conflict → CI Stage 6 recreates the secret with `--dry-run | kubectl apply -f -`

### LangFuse dataset showing 0 experiments after a CI run

Stage 3 creates a new experiment run with each CI push. If you see 0 experiments,
the eval job likely hit a GROQ token limit or crashed during `run_eval.py`.
Check the "Stage 3 — LLM Evaluation" GitHub Actions log.

Normal state:
- Dataset `ecommerce-eval-dataset`: exactly **15 items** (full-refreshed on every CI run)
- Experiments: one entry per CI run (accumulate over time — this is expected)

### How to check what's running in k8s

```bash
kubectl get pods -n ecommerce                    # list all pods
kubectl logs -l component=fastapi -n ecommerce --tail=50   # app logs
kubectl get secret ecommerce-secrets -n ecommerce -o yaml  # verify secret values
kubectl port-forward svc/ecommerce-fastapi 8000:80 -n ecommerce  # manual tunnel
```

---

## 18. NON-OBVIOUS CODE DECISIONS (THE "WHY") 

These are the things that aren't obvious from reading the code:

### Why KEYWORD_MAP is an ordered list, not a dict
Python dicts are unordered (well, insertion-ordered in 3.7+ but conceptually unordered
for this use case). The order matters: multi-word entries must be checked before their
single-word components. A list of tuples preserves the intended priority.

### Why `or_()` in SQLAlchemy product search
SQLAlchemy's `.filter()` chains ANDs conditions. To express `WHERE a OR b OR c`,
you must use `sqlalchemy.or_()`. The product search checks `product_type ILIKE X
OR product_type ILIKE singular OR product_type ILIKE singular+"s"` to handle
plural/singular variants.

### Why `\bs?` instead of matching both "fan" and "fans" separately
`\bfans?\b` means: word boundary, then "fan", then optional "s", then word boundary.
This matches both "fan" and "fans" in one regex. Without `\b`, "fan" would match
"fancy", "fantastic", "infant". Without `s?`, you'd need two entries per word.

### Why session_context uses topic="support_query" not topic="order_query"
The support agent sets `topic="support_query"` (not "order_query") because the
conversation is about a support issue. When the user later asks "when will I get
the refund", the message is routed to the order agent. The order agent's
`validate_input` checks `ctx.get("order_id")` directly, which works regardless of
topic. It also checks `ctx.get("topic") in ("order_query", "support_query")` for
the all-orders fallback.

### Why the LLM rank_and_filter prompt has explicit counter-examples
LLMs with soft instructions ("return [] if unrelated") are too eager to return
something. Without explicit counter-examples like "if user asked for an air
conditioner, do not return fans", the LLM would return fans because they both
"cool the air". Counter-examples anchor the LLM to the stricter interpretation.

### Why order_agent builds the all-orders response in Python (no LLM)
An earlier version used an LLM to format the order list, which was slow (+1.5 sec)
and introduced variability. A flat list of orders is structured data — it never
needs an LLM. Building it in Python is instant, deterministic, and zero-cost.

### Why LangFuse prompt fetching has a fallback
`get_prompt("name", label="production", fallback=fallback_prompt)`:
If LangFuse is down or the prompt name doesn't exist, the hardcoded fallback_prompt
is used. This makes the system resilient — it degrades gracefully rather than
crashing when LangFuse is unavailable.

### Why duplicate support ticket handling uses exact pre-computed strings
Prompting the LLM with "if duplicate, don't say a new ticket was created" is
unreliable — LLMs sometimes deviate. Instead, the code constructs the exact
response string before calling the LLM and injects it as:
`"Use EXACTLY this response: '...'"`
This gives the LLM no room to improvise on compliance-sensitive content.

### Why classify_issue strips order_id from context before sending to LLM
```python
context_str = format_context({k: v for k, v in ctx.items() if k != "order_id"})
```
If the old order_id is in context (from a previous support query), the LLM might
use that stale order_id instead of extracting the fresh one from the current message.
Stripping it forces the LLM to work only from the current message.

### Why product search skips `product_type="other"`
The DB has ~391 products with `product_type="other"` — TVs, calculators, batteries
that the Kaggle importer couldn't classify. If we include "other" in a product_type
search, we'd return random unrelated items. Skipping it is the correct behavior.
Users searching for TVs might still find them via the name fallback if the LLM
generates `product_type="tv"`.

---

## 19. KNOWN LIMITATIONS AND HOW TO TALK ABOUT THEM IN A DEMO

### "Why does [product] not show results?"

The product search relies on:
1. The LLM correctly extracting a product_type
2. That product_type matching one of the 40+ types in the DB

If the LLM returns a made-up type, the KEYWORD_MAP catches most common terms but
can't catch everything. Demo strategy: stick to products in the catalog list
(Section 6, "Products proven to work reliably"). If an interviewer asks for a
product not in that list, the system correctly shows the catalog message — this is
expected behavior.

### "Why does the no-results message show the same list every time?"

The catalog message is hardcoded because it reflects what's actually in the database.
It's intentional — accurate fallback is better than showing wrong products.
The improvement path: dynamic catalog query showing only in-stock categories.

### "Why don't you use embeddings / RAG for product search?"

Current approach: exact product_type + name ILIKE. This is intentional for this scale.
With 1,351 products, a well-classified DB + deterministic search is faster, cheaper,
and more explainable than semantic search. At 100K+ products with rich descriptions,
embeddings would start to make sense.

### "The LLM sometimes gets product type wrong"

Mitigation in place: KEYWORD_MAP (deterministic override), normalization dict,
3-attempt broadening, and rank_and_filter (second LLM quality gate).
Root cause: product catalog quality — better `product_type` data would help more
than any code change.

### "Why not use GPT-4 / Claude?"

Groq API was chosen for speed (ultra-low latency inference) and because it's
completely free for demo usage. The architecture is LLM-agnostic — swapping
`ChatGroq` for `ChatAnthropic` or `ChatOpenAI` requires changing one line per agent.

---

## 20. PROJECT STRUCTURE — EVERY FILE AND WHAT IT DOES

```
ecommerce-agent/
│
├── main.py                        FastAPI app init, CORS, static files, Prometheus middleware
│
├── api/
│   ├── routes.py                  /chat, /health, /config, /metrics endpoints + session logic
│   └── schemas.py                 Pydantic models: ChatRequest, ChatResponse, etc.
│
├── agents/
│   ├── intent_router.py           Top-level router — classifies intent, calls sub-agents
│   ├── order_agent.py             Order tracking, status, listing
│   ├── product_agent.py           Product search, recommendations, filtering
│   └── support_agent.py           Complaints, refunds, ticket creation
│
├── subgraphs/
│   ├── product_enrichment.py      Fetch specs + compute relevance score for products
│   ├── shipment_tracking.py       Get carrier info + ETA from carrier_tracking table
│   └── escalation_handler.py      Check duplicates + create support ticket
│
├── tools/
│   ├── product_tools.py           search_products_tool, fetch_specs_tool
│   ├── order_tools.py             fetch_order_data, fetch_all_orders_for_user, fetch_tracking_data
│   └── support_tools.py           lookup_policy_tool, check_user_history_tool, create_ticket_tool
│
├── services/
│   ├── mock_product_api.py        DB queries for products (type search + name fallback)
│   ├── mock_order_api.py          DB queries for orders
│   └── mock_support_api.py        DB queries for policies + tickets
│
├── database/
│   ├── models.py                  All 8 SQLAlchemy models (users, sessions, messages, orders, products, policies, carrier_tracking, support_tickets, agent_runs)
│   ├── connection.py              SQLAlchemy engine, SessionLocal, Base
│   ├── import_kaggle_data.py      Imports Amazon product CSV, classifies into product_type, seeds orders/tracking/policies
│   └── classify_products.py       Product classification logic used during import
│
├── config/
│   └── settings.py               Pydantic Settings — all config from .env
│
├── langfuse_helpers/
│   ├── tracing.py                 create_span, end_span, create_generation, get_prompt
│   ├── scoring.py                 Post-response scoring (relevancy, completion, correctness)
│   └── evaluation.py             Evaluation run helpers
│
├── monitoring/
│   ├── metrics.py                 Prometheus counter/histogram definitions
│   ├── logging_config.py          JSON structured logging setup
│   └── (grafana/prometheus/loki configs in yaml files)
│
├── utils/
│   ├── memory.py                  format_context, format_recent_messages, merge_context helpers
│   └── (other shared utils)
│
├── eval/
│   ├── run_eval.py               Main eval runner (mocked or live)
│   ├── seed_dataset.py           Create evaluation dataset in LangFuse
│   ├── seed_prompts.py           Push versioned prompts to LangFuse
│   └── setup_langfuse_evaluators.py  Register LLM-as-judge evaluator configs
│
├── tests/                         pytest unit tests (≥85% coverage required for CI)
│
├── frontend/
│   └── index.html                 Single-file HTML/JS chat UI with Google Sign-In
│
├── helm/                          Kubernetes Helm chart
├── Dockerfile                     Container build (Python 3.12-slim)
├── docker-compose.yml             Full stack: app + postgres + monitoring + langfuse
├── entrypoint.sh                  DB migration (alembic) + uvicorn startup
└── requirements.txt               All Python dependencies
```

---

## 21. SUMMARY FOR PPT / ARCHITECTURE DIAGRAM

**Project name**: Multi-Agent AI Customer Support System for E-Commerce

**One-line pitch**: An AI chatbot that routes customer queries to specialized agents for orders, product search, and support — with full observability and automated evaluation.

**Key numbers**:
- 3 specialist agents + 1 intent router
- 3 subgraphs (product enrichment, shipment tracking, escalation handler)
- 8 database tables
- 1,351 products in catalog
- 40+ searchable product types
- 6-stage CI/CD pipeline (lint → test → eval → build → push → deploy)
- 15 eval test cases (LangFuse dataset, refreshed on every CI run)
- LangFuse traces every LLM call with token usage + cost
- Prometheus + Grafana dashboard with 6 real-time panels
- Automated LLM-as-judge evaluation on every CI run
- Kubernetes deployment on kind cluster (local self-hosted runner)
- Neon serverless PostgreSQL (cloud) for production DB

**Architecture layers (for diagram):**
```
[Frontend HTML/JS]
      ↓ HTTP
[FastAPI]  ←→  [PostgreSQL]
      ↓
[Intent Router (LangGraph)]
   ↙    ↓    ↘
[Order] [Product] [Support]  ← 3 LangGraph StateGraphs
   ↓       ↓        ↓
[Subgraphs: Shipment / Enrichment / Escalation]
      ↓
[Groq LLM — llama-3.3-70b-versatile]
      ↓
[LangFuse]  [Prometheus → Grafana]  [Loki → Grafana]
```

**What makes this production-grade (beyond just making it work):**
1. Prompt versioning in LangFuse — update prompts without redeployment
2. Automated LLM-as-judge evaluation in CI
3. Full distributed tracing of every LLM call with token cost
4. Deterministic keyword layer over LLM (KEYWORD_MAP) for reliability
5. 3-tier product search with broadening fallback
6. Cross-agent session_context dict for multi-turn conversation coherence
7. Kubernetes deployment with Helm + auto-rollback on smoke test failure
8. Google OAuth with anonymous fallback
9. Support ticket deduplication logic
10. Order ownership validation (user can't see another user's order)
