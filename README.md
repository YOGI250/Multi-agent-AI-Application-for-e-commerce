## Known Improvements

### 1. Session Context Extraction
Currently conversation history is passed to LLMs as raw text.
A session context extractor would scan history and build structured
facts (mentioned order IDs, ticket IDs, last product type) before
calling any agent. This would allow agents to resolve vague references
like "my first complaint" or "that ticket" without the user repeating
IDs every time. Ordinal references like "third order" would still
require clarification when multiple orders are in context.

### 2. Ticket Lifecycle Management
Tickets currently stay open indefinitely. Production implementation
would include ticket closure via agent action, customer confirmation,
or auto-closure after inactivity. SLA-based escalation would become
meaningful once tickets can be resolved.

### 3. Token Refresh
JWT tokens expire after 1 hour. A refresh token mechanism would
silently obtain new tokens without interrupting the user session.
Currently handled by frontend detection with a clear re-login prompt.

### 4. Carrier-Specific Tracking
All carriers (BlueDart, FedEx, DTDC, DHL) use the same tracking logic.
Production would integrate each carrier's specific API.

### 5. LLM-Based Product Classification
The classify_products.py script uses keyword rules for one-time
product type classification. LLM-based classification in batches
would be more accurate especially for accessory products whose
names mention the main product.

### 6. Product Name Based Order Lookup
Currently the Order Agent requires an explicit order ID (ORD-XXXXX format).
If a user says "what is the status of my laptop" without an order ID,
the intent router may misclassify it as product_query and the Order Agent
cannot fetch by product name.

Full fix requires three changes:
1. Intent router — distinguish "my laptop" (order context) from 
   "show me laptop" (shopping context) using ownership keywords
2. Order Agent validate_input — extract product name when no order ID found
3. Order Agent fetch — query orders table by product name + user_id,
   then smart filter to surface the undelivered order automatically

Example scenario: user ordered 4 laptops, 3 delivered, 1 in transit.
"what is the status of my laptop" should automatically find and return
the in-transit order without asking for an order ID.