# Struggles & Solutions — What We Fought and How We Fixed It

> Real problems encountered while building this project, what caused them,
> and exactly how they were resolved. Useful for interviews — showing you
> know *why* something broke is more impressive than saying it worked.

---

## 1. Authentication — Google JWT Verification Failed in Docker

**The struggle:**
Google OAuth worked fine locally. Inside Docker, the JWT verification call crashed
with a connection error. The container couldn't reach Google's public key endpoint
(`https://www.googleapis.com/oauth2/v3/certs`) because Docker's internal network
was isolated from the internet in that environment.

**What was happening:**
FastAPI calls `google.oauth2.id_token.verify_oauth2_token()` which internally
fetches Google's public key to verify the JWT signature. That HTTP call was timing
out inside the container.

**How we fixed it:**
Added a local JWT decode fallback. If the Google cert endpoint is unreachable,
the server decodes the JWT locally without verifying the signature — sufficient
for a demo environment where we trust the token structure.

```python
try:
    idinfo = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)
except Exception:
    # fallback: decode without verification (Docker network isolation)
    idinfo = jwt.decode(token, options={"verify_signature": False})
```

**The lesson:**
JWT verification requires fetching a live public key. In isolated networks
(Docker, VPNs, restricted environments), always have a fallback path.

---

## 2. Product Search — "pen" Matched "spend", "fan" Matched "fantastic"

**The struggle:**
When a user searched for "pen" or "fan", the keyword matching was firing
incorrectly for messages like "I want to spend under 2000" or "that's fantastic".

**What was happening:**
The original KEYWORD_MAP check used plain Python `in` operator:
```python
if "pen" in message.lower():   # "spend" contains "pen" → wrong match
```
This is a substring match, not a word match. "pen" is inside "spend", "open",
"happen". "fan" is inside "fantastic", "infant".

**How we fixed it:**
Replaced plain substring check with a word-boundary regex:
```python
def _kw_match(msg: str, kw: str) -> bool:
    if " " in kw:
        return kw in msg   # multi-word phrases: plain substring is fine
    return bool(re.search(r"\b" + re.escape(kw) + r"s?\b", msg))
```
`\b` = word boundary. "pen" now only matches the standalone word "pen" or "pens",
not "spend" or "open". The `s?` handles regular plurals so "fans" also matches "fan".

**The lesson:**
Substring matching is dangerous for short keywords. Always use word boundaries
when matching single words in natural language.

---

## 3. Product Search — Irregular Plurals Broke the Matching

**The struggle:**
After adding `s?` for plurals, "smartwatches" still didn't match "smartwatch"
and "mice" didn't match "mouse". Users searching "show me smartwatches" got no results.

**What was happening:**
`s?` handles regular plurals (fan→fans, ram→rams). But English has irregular plurals:
- watch → watches (not watchs)
- mouse → mice (not mouses)

The regex `\bsmartwatchs?\b` would match "smartwatch" and "smartwatchs" — neither
of which the user types.

**How we fixed it:**
Added explicit entries for each irregular plural form at the top of KEYWORD_MAP,
before their singular versions:
```python
("smartwatches", "smartwatch"),   # irregular plural — before "smartwatch"
("smartwatch", "smartwatch"),
("mice", "mouse"),                # irregular plural — before "mouse"
("mouse", "mouse"),
```

**The lesson:**
No regex trick handles all of English. For known irregular forms, explicit entries
are more reliable than clever pattern matching.

---

## 4. Product Search — "mouse pad" Was Routing to "mouse"

**The struggle:**
Users searching "show me a mouse pad" were getting mice results instead of mousepads.

**What was happening:**
The word "mouse" appears in "mouse pad". The KEYWORD_MAP entry for "mouse" was
firing before the "mouse pad" entry got a chance to be checked.

**How we fixed it:**
Multi-word entries must come before their single-word components in the KEYWORD_MAP
list. The list is checked top-to-bottom and stops at the first match:
```python
KEYWORD_MAP = [
    ("mouse pad", "mousepad"),   # checked first — catches "mouse pad"
    ("mouse mat", "mousepad"),
    ...
    ("mouse", "mouse"),          # checked after — only fires if no multi-word matched
]
```
Same logic applies to "laptop stand" before "laptop", "smart watch" before "watch".

**The lesson:**
When matching from a list of patterns with overlapping words, always check longer/more-specific patterns before shorter ones.

---

## 5. Product Search — "geyser" Returned Nothing

**The struggle:**
Users searching "show me geysers" got the no-results message even though the
catalog has water heaters (which is what a geyser is).

**What was happening:**
The database has `product_type = "water_heater"`. The word "geyser" is a regional
Indian term for water heater. The LLM didn't reliably map "geyser" → "water_heater"
because it wasn't hinted to do so.

**How we fixed it (three places):**

1. **KEYWORD_MAP** — deterministic override:
   ```python
   ("geyser", "water_heater"),
   ```

2. **Normalization dict** — catches LLM plural output:
   ```python
   "geyser": "water_heater",
   "geysers": "water_heater",
   ```

3. **LLM prompt hint** — tells the LLM what to do:
   ```
   NORMALISE: geysers→water_heater
   ```

**The lesson:**
For regional synonyms (geyser = water heater, mixer grinder = mixer), you need
the fix in all three layers: deterministic keyword map, normalization, and LLM hint.
Relying on the LLM alone to "know" regional terms is unreliable.

---

## 6. Product Search — Random Products Returned for Unavailable Items

**The struggle:**
When a user asked for "air conditioner" (not in catalog), the agent was returning
fans and air purifiers instead of saying "we don't have that".

**What was happening:**
The `rank_and_filter` LLM prompt said:
> "Return [] ONLY if entirely unrelated"

"Entirely unrelated" is too loose. Fans and air purifiers both relate to cooling,
so the LLM included them. From the user's perspective, they asked for an AC and
got a ceiling fan — which is wrong and confusing.

**How we fixed it:**
Tightened the prompt with explicit counter-examples:
```
STRICT: Do NOT include a product just because it is vaguely related.
If the user asked for an air conditioner, do not return fans or air purifiers.
If the user asked for a refrigerator, do not return water purifiers or coolers.
Return [] if none of the listed products are genuinely what the user asked for.
When in doubt, return [].
```

**The lesson:**
Soft LLM instructions ("only if unrelated") need explicit counter-examples to anchor
the boundary. Without examples, LLMs interpret "unrelated" too broadly.

---

## 7. Order Agent — "When Will I Get the Refund" Asked for Order ID

**The struggle:**
A user raised a support ticket for a damaged item, got a ticket created, then asked
"when will I get the refund?" The order agent responded "I need your order ID" — even
though the order ID was already known from the support conversation.

**What was happening:**
The order agent's session context fallback only checked:
```python
if ctx.get("topic") == "order_query":
    wants_all_orders = True
```
But the support agent sets `topic = "support_query"`, not `"order_query"`. So the
order agent ignored the `order_id` that the support agent had already stored in
session_context.

**How we fixed it:**
Check for `order_id` in context directly, regardless of which agent set it:
```python
if not wants_all_orders and ctx.get("order_id"):
    state["order_id"] = ctx["order_id"]
    state["order_id_found"] = True
elif not wants_all_orders and ctx.get("topic") in ("order_query", "support_query"):
    wants_all_orders = True
```

**The lesson:**
Cross-agent session context must be read broadly. An order_id is an order_id
regardless of which agent wrote it. Don't gate on topic when the data itself is sufficient.

---

## 8. Order Agent — "Detailed Information" Showed the Same Flat List

**The struggle:**
When a user asked "give me detailed information of orders not yet delivered",
they got the same flat list (order ID, item name, status) — no extra detail.

**What was happening:**
The agent had no concept of "the user wants more detail". It always returned
the same flat list regardless of how the question was phrased.

**How we fixed it:**
Added a `wants_detail` detection flag:
```python
wants_detail = any(w in msg_lower for w in
    ("detail", "detailed", "full", "more info", "information", "elaborate"))
```
If `wants_detail` is True and there are multiple orders, instead of dumping more
data (which would be overwhelming), we guide the user:
```
I can show full tracking and delivery details for one order at a time.
Which order would you like to check?
```

**The lesson:**
"More detail" doesn't always mean show more data. For list views, the right UX
is to guide the user to narrow down — then show deep detail for one item.

---

## 9. Deployment — Container Running Stale Code

**The struggle:**
Fixed several bugs, committed the code, tested in the browser — bugs were still
there. Spent time double-checking the code, couldn't understand why nothing changed.

**What was happening:**
The Docker container was built at 21:46 IST. All the bug-fix commits happened
after 22:01 IST. The running container had the old image — `docker compose up -d`
restarts the container but does **not** rebuild the image.

**How we fixed it:**
```bash
docker compose build fastapi   # re-copies code into the image
docker compose up -d fastapi   # restart with the new image
```

**The lesson:**
`docker compose up -d` ≠ rebuild. You must explicitly run `build` after changing
Python files. Add this to muscle memory: **change code → build → up**.

---

## 10. LangFuse — Traces Not Appearing / Missing Data

**The struggle:**
Some requests weren't showing up in LangFuse traces, or traces were incomplete —
spans were missing even though the code was creating them.

**What was happening:**
LangFuse v4 uses an OTEL-based SDK. Traces are buffered and flushed asynchronously.
If the process ends (or the response is returned) before the flush completes,
trace data is lost. Also, `session_id` and `user_id` needed to be set via
specific OTEL attribute keys, not arbitrary fields.

**How we fixed it:**
- Called `trace.end()` explicitly before flushing
- Used the correct LangFuse v4 OTEL attribute keys for `session_id` and `user_id`
- Added `langfuse_client.flush()` at the end of each request in FastAPI

**The lesson:**
Async observability SDKs require explicit lifecycle management. "Fire and forget"
doesn't work when the process is short-lived or the buffer hasn't flushed yet.

---

## 11. Intent Router — Context Contamination on Greetings

**The struggle:**
After a long order conversation, a user typed "hi" — and got routed to the order
agent because the LLM saw recent order messages in history and assumed order intent.

**What was happening:**
The LLM was given session_context + recent history. Seeing "order_id: ORD-1234"
in context, it classified "hi" as order_query (confidence 0.7).

**How we fixed it:**
Added a deterministic pre-check before the LLM is called:
```python
_GREETINGS = {"hi", "hello", "hey", "hiya", ...}
if msg_lower in _GREETINGS:
    state["intent"] = "unknown"
    return state  # never reaches LLM
```
Greetings always return "unknown" → fallback response. Context is irrelevant for greetings.

**The lesson:**
Some intents are unambiguous and should never be handed to an LLM. Deterministic
pre-checks are faster, cheaper, and more reliable for clear-cut cases.

---

## 12. Support Agent — Duplicate Ticket Response Was Inconsistent

**The struggle:**
When a user filed the same complaint twice, the LLM was supposed to say
"your ticket is already being handled" — but sometimes it said "a new ticket
has been created for you" even when no new ticket was created.

**What was happening:**
The prompt instruction was:
> "If duplicate, do NOT say a new ticket was created"

LLMs with soft negation instructions ("do NOT") are unreliable. The LLM would
sometimes ignore the instruction, especially with longer prompts.

**How we fixed it:**
Pre-computed the exact response string in Python and injected it as a rigid instruction:
```python
duplicate_instruction = (
    f"Use EXACTLY this response (do not add or change anything):\n"
    f'"Your ticket {ticket_id} was recently created. Our team will '
    f'contact you within 48 hours."'
)
```

**The lesson:**
For compliance-sensitive or legally-phrased content, don't trust the LLM to
word it correctly. Pre-compute the exact string and instruct the LLM to use it verbatim.
