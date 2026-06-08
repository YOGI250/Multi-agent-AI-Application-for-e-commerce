

import json
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langfuse_helpers.tracing import langfuse_client, flush

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed():
    dataset_name = "ecommerce-eval-dataset"

    with open("eval/dataset.json", "r") as f:
        items = json.load(f)

    # create dataset  
    try:
        langfuse_client.create_dataset(
            name        = dataset_name,
            description = "Ecommerce agent evaluation — 15 cases covering all 3 agents"
        )
        logger.info(f"Dataset created: {dataset_name}")
    except Exception as e:
        logger.info(f"Dataset already exists or error: {e}")

    # create items — skip if a matching message already exists (idempotent)
    try:
        existing = langfuse_client.get_dataset(dataset_name)
        existing_messages = {it.input.get("message", "") for it in existing.items}
    except Exception:
        existing_messages = set()

    added = 0
    for i, item in enumerate(items):
        msg = item["input"].get("message", "")
        if msg in existing_messages:
            logger.info(f"  Item {i+1}/{len(items)} already exists — skipping")
            continue
        langfuse_client.create_dataset_item(
            dataset_name    = dataset_name,
            input           = item["input"],
            expected_output = item["expected_output"]
        )
        logger.info(f"  Item {i+1}/{len(items)} added: {msg[:50]}")
        added += 1

    flush()
    logger.info(f"Done — {added} new items added to '{dataset_name}'")


if __name__ == "__main__":
    seed()
    