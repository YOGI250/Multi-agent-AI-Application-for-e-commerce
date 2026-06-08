

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
            description = "Ecommerce agent evaluation — 5 cases covering all 3 agents"
        )
        logger.info(f"Dataset created: {dataset_name}")
    except Exception as e:
        logger.info(f"Dataset already exists or error: {e}")

    # create items
    for i, item in enumerate(items):
        langfuse_client.create_dataset_item(
            dataset_name    = dataset_name,
            input           = item["input"],
            expected_output = item["expected_output"]
        )
        logger.info(f"  Item {i+1}/5 added: {item['input']['message'][:50]}")

    flush()
    logger.info(f"Done — {len(items)} items in '{dataset_name}'")


if __name__ == "__main__":
    seed()
    