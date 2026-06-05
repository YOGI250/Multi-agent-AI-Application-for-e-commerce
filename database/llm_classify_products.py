#!/usr/bin/env python3
"""
database/llm_classify_products.py

Replaces classify_products.py.
Uses Groq LLM to classify each product by name into a product_type.
Runs once — writes results back to PostgreSQL.

Usage:
    python3 -m database.llm_classify_products
"""

import json
import logging
import time
import re

from langchain_groq import ChatGroq
from database.connection import SessionLocal
from database.models import Product
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_TYPES = [
    "mouse", "keyboard", "headphones", "speaker", "laptop", "tablet",
    "smartwatch", "monitor", "webcam", "router", "cable", "charger",
    "usb_hub", "pendrive", "ssd", "hard_disk", "ram", "memory_card",
    "printer", "stand", "mousepad", "laptop_bag", "phone_case", "extension",
    "fan", "mixer", "iron", "kettle", "water_heater", "room_heater",
    "vacuum", "washing_machine", "air_purifier", "water_purifier",
    "camera", "pen", "notebook", "light", "trimmer", "microwave", "other",
]

BATCH_SIZE = 20


def classify_batch(llm: ChatGroq, batch: list[tuple[str, str]]) -> dict[str, str]:
    """
    batch: list of (product_id, product_name)
    Returns: dict of product_id -> product_type
    """
    numbered = "\n".join(
        f"{i+1}. {name[:120]}" for i, (_, name) in enumerate(batch)
    )

    prompt = f"""You are a product classification expert for an Indian e-commerce platform.

Classify each product into exactly ONE product type from this list:
{", ".join(VALID_TYPES)}

Rules:
- "cable" = any USB, HDMI, charging, data, or lightning cable
- "charger" = power adapters, wall chargers, wireless chargers (NOT cables)
- "ram" = ONLY actual RAM/memory modules (DDR4, DDR5). NOT phones that mention "4GB RAM"
- "other" = anything not in the list (phones, TVs, rice cookers, calculators, batteries, etc.)
- When in doubt, use "other"

Products to classify:
{numbered}

Respond ONLY with a JSON object mapping number to product_type.
Example: {{"1": "fan", "2": "cable", "3": "other"}}"""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.warning("No JSON found in LLM response — marking batch as 'other'")
            return {pid: "other" for pid, _ in batch}

        raw = json.loads(match.group())
        result = {}
        for i, (pid, _) in enumerate(batch):
            key = str(i + 1)
            ptype = raw.get(key, "other").strip().lower()
            result[pid] = ptype if ptype in VALID_TYPES else "other"
        return result

    except Exception as e:
        logger.error(f"LLM batch failed: {e} — marking batch as 'other'")
        return {pid: "other" for pid, _ in batch}


def run():
    db = SessionLocal()
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    try:
        products = db.query(Product.product_id, Product.name).all()
        total = len(products)
        logger.info(f"Classifying {total} products using Groq LLM...")

        all_results: dict[str, str] = {}
        batches = [products[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Batch {batch_num}/{len(batches)} ({len(batch)} products)...")
            result = classify_batch(llm, batch)
            all_results.update(result)

            # Groq free tier: avoid hitting rate limits
            if batch_num % 5 == 0:
                time.sleep(2)

        # Write all results back to DB in one pass
        logger.info("Writing classifications to PostgreSQL...")
        updated = 0
        type_counts: dict[str, int] = {}

        for product in db.query(Product).all():
            ptype = all_results.get(product.product_id, "other")
            product.product_type = ptype
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
            updated += 1

        db.commit()
        logger.info(f"Updated {updated} products.")

        logger.info("\nProduct type distribution:")
        for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {count:4d}x  {ptype}")

    except Exception as e:
        logger.error(f"Error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run()
