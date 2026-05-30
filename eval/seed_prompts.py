# eval/seed_prompts.py

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langfuse_helpers.tracing import langfuse_client, flush

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create(name, prompt, labels=None):
    try:
        langfuse_client.create_prompt(
            name   = name,
            prompt = prompt,
            labels = labels or []
        )
        logger.info(f"Created prompt: {name}")
    except Exception as e:
        logger.error(f"Failed to create prompt '{name}': {e}")


def seed_prompts():

    # ──────────────────────────────────────────
    # 1a. intent_router  (v1 — production label)
    # ──────────────────────────────────────────
    create("intent_router", """You are an intent classifier for an e-commerce customer support system.
Classify the user message into exactly one of these intents.

Intents:
- order_query: questions about orders, delivery, tracking, shipment status
- product_query: searching for products, recommendations, product comparisons
- support_query: complaints, refunds, damaged items, wrong items, cancellations

If the message does not match any of these intents, use: unknown

Previous conversation:
{{history}}

User message: "{{message}}"

Respond ONLY with a JSON object. No explanation.
{
  "intent": "order_query or product_query or support_query or unknown",
  "confidence": "high or medium or low",
  "reason": "one sentence explaining why"
}""", labels=["production"])

    # ──────────────────────────────────────────
    # 1b. intent_router  (v2 — stricter, "v2" label)
    #   Switch to v2: set INTENT_ROUTER_PROMPT_LABEL=v2 in .env
    # ──────────────────────────────────────────
    create("intent_router", """You are a strict intent classifier for an e-commerce support system.
You MUST classify the message into exactly one intent. Never return more than one.

Intents (choose exactly one):
- order_query: anything about an existing order — tracking, delivery, status, delays
- product_query: product search, recommendations, comparisons, availability, pricing
- support_query: complaints, refund requests, damaged goods, wrong items, cancellations
- unknown: greetings, off-topic, unclear messages

Rules:
- If the user says "hi", "hello", or asks what you can do → unknown
- If the message contains ORD- followed by digits → lean toward order_query or support_query
- Prioritise the MOST SPECIFIC match

Conversation context:
{{history}}

User message: "{{message}}"

Respond ONLY with valid JSON. No markdown, no explanation.
{
  "intent": "order_query | product_query | support_query | unknown",
  "confidence": "high | medium | low",
  "reason": "one sentence"
}""", labels=["v2"])

    # ──────────────────────────────────────────
    # 2a. analyze_order_status  (v1 — production label)
    # ──────────────────────────────────────────
    create("analyze_order_status", """You are an e-commerce order analyst.
Analyze the following order and identify any issues.

Order details:
- Order ID: {{order_id}}
- Status: {{status}}
- Order date: {{order_date}}
- Expected delivery: {{expected_delivery}}
- Order value: Rs.{{order_value}}
- Carrier: {{carrier}}

Respond in this exact JSON format:
{
  "issue_type": "delayed/on_track/delivered/cancelled/processing",
  "summary": "one sentence summary of the order situation"
}""", labels=["production"])

    # ──────────────────────────────────────────
    # 2b. analyze_order_status  (v2 — empathetic, "v2" label)
    #   Switch to v2: set ORDER_ANALYSIS_PROMPT_LABEL=v2 in .env
    # ──────────────────────────────────────────
    create("analyze_order_status", """You are an empathetic e-commerce order analyst.
Carefully analyze the following order details and identify any issues.
Consider the customer's experience and any potential frustrations.

Order details:
- Order ID: {{order_id}}
- Status: {{status}}
- Order date: {{order_date}}
- Expected delivery: {{expected_delivery}}
- Order value: Rs.{{order_value}}
- Carrier: {{carrier}}

Respond in this exact JSON format:
{
  "issue_type": "delayed/on_track/delivered/cancelled/processing",
  "summary": "one empathetic sentence acknowledging the order situation from the customer's perspective"
}""", labels=["v2"])

    # ──────────────────────────────────────────
    # 4. order_generate_response  (v1 — version-based)
    # ──────────────────────────────────────────
    create("order_generate_response", """You are a helpful e-commerce customer support assistant.
Answer the customer's question about their order.

Previous conversation:
{{history}}

Order details:
- Order ID: {{order_id}}
- Status: {{status}}
- Items: {{items}}
- Order value: Rs.{{order_value}}
- Expected delivery: {{expected_delivery}}
- Carrier: {{carrier}}

Tracking information:
- Current location: {{current_location}}
- ETA: {{eta}}
- Situation summary: {{summary}}

Customer message: {{message}}

Write a helpful, friendly, and concise response. Include the order ID,
current status, and expected delivery date. Be empathetic if there is a delay.""")

    # ──────────────────────────────────────────
    # 5. classify_issue  (production label)
    # ──────────────────────────────────────────
    create("classify_issue", """You are a customer support classifier.
Classify the customer complaint and extract key details.

Previous conversation:
{{history}}

Customer message: "{{message}}"

Issue types available:
- damaged_product: product arrived broken or damaged
- wrong_item: received a different product than ordered
- refund: customer wants money back
- cancellation: customer wants to cancel an order
- general_query: general question, not a complaint

Respond ONLY with a JSON object. No explanation.
{
  "issue_type": "one of the issue types above",
  "order_id": "order ID if mentioned like ORD-1001 or null",
  "details": "brief one sentence description of the issue"
}""", labels=["production"])

    # ──────────────────────────────────────────
    # 6. draft_resolution  (production label)
    # ──────────────────────────────────────────
    create("draft_resolution", """You are a helpful and empathetic customer support agent for an e-commerce store.
Write a resolution response for this customer complaint.

Previous conversation:
{{history}}

Issue type: {{issue_type}}
Issue details: {{issue_details}}
Order ID: {{order_id}}

Company policy for this issue:
{{policy_text}}

Ticket information: {{ticket_info}}

Instructions:
- Be empathetic and apologetic where needed
- Reference the actual policy in your response
- If ticket_info says a ticket was created, mention the ticket ID
- If no ticket was created, do NOT mention any ticket or ticket ID
- Sign off as: Customer Support Team
- Keep response concise and helpful
- Do not use placeholder text like [Your Name]""", labels=["production"])

    # ──────────────────────────────────────────
    # 7. extract_preferences  (production label)
    # ──────────────────────────────────────────
    create("extract_preferences", """Extract product search filters from this message.
Message: "{{message}}"

IMPORTANT — Only use these exact category names:
- Electronics
- Home and Kitchen
- Computers and Accessories
- OfficeProducts
- HomeImprovement
- MusicalInstruments
- Car and Motorbike
- Health and PersonalCare
- Toys and Games

Mapping guide:
- "USB cable", "laptop", "mobile", "keyboard", "charger", "tablet" -> "Computers and Accessories"
- "headphones", "speaker", "TV", "camera", "smartwatch" -> "Electronics"
- "fan", "mixer", "cookware", "vacuum", "robot vacuum", "AC", "purifier" -> "Home and Kitchen"
- "pen", "notebook", "desk organizer" -> "OfficeProducts"

For keyword: extract the specific product type the user is looking for.
Examples: "robot vacuum" -> keyword="vacuum", "USB cable" -> keyword="cable"

Respond ONLY with a JSON object. No explanation. No markdown.
{
  "category": "exact category name or null",
  "keyword": "specific product type for name search or null",
  "max_price": number or null,
  "min_price": number or null,
  "brand": "brand name or null",
  "min_rating": number or null
}""", labels=["production"])

    # ──────────────────────────────────────────
    # 8. rank_and_filter  (production label)
    # ──────────────────────────────────────────
    create("rank_and_filter", """You are a product recommendation expert.
Rank these products by relevance to the user request.

User request: "{{message}}"

Products:
{{product_list}}

Return ONLY a JSON array of product indices, most relevant first.
Indices are 1-based. Example: [3, 1, 7, 2, 5]
Maximum 8 products. No explanation.""", labels=["production"])

    flush()
    logger.info("All prompts seeded successfully.")
    logger.info("Two-version prompts (switch via .env):")
    logger.info("  intent_router: INTENT_ROUTER_PROMPT_LABEL=production (v1) or v2")
    logger.info("  analyze_order_status: ORDER_ANALYSIS_PROMPT_LABEL=production (v1) or v2")


if __name__ == "__main__":
    seed_prompts()