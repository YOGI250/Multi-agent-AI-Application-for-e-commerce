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
        conversations = json.load(f)

    # flatten conversations into individual scored turns —
    # context-building turns without expected_output are not pushed as items
    items = []
    for conv in conversations:
        user_id = conv.get("user_id", "eval_guest_001")
        for turn_idx, turn in enumerate(conv["turns"], start=1):
            if not turn.get("expected_output"):
                continue
            items.append(
                {
                    "input": {
                        "message": turn["message"],
                        "user_id": user_id,
                        "conversation_id": conv.get("conversation_id"),
                        "turn": turn_idx,
                    },
                    "expected_output": turn["expected_output"],
                }
            )

    # create dataset if it doesn't exist
    try:
        langfuse_client.create_dataset(
            name=dataset_name,
            description="Ecommerce agent evaluation — multi-turn conversations covering all 3 agents"
        )
        logger.info(f"Dataset created: {dataset_name}")
    except Exception as e:
        logger.info(f"Dataset already exists: {e}")

    # full refresh — delete all existing items then re-add from dataset.json
    # ensures LangFuse stays in sync with dataset.json on every CI run
    try:
        existing = langfuse_client.get_dataset(dataset_name)
        existing_items = existing.items or []
        if existing_items:
            for it in existing_items:
                try:
                    langfuse_client.api.dataset_items.delete(it.id)
                except Exception as del_err:
                    logger.warning(f"  Could not delete item {it.id}: {del_err}")
            logger.info(f"Deleted {len(existing_items)} existing item(s)")
        else:
            logger.info("No existing items to delete")
    except Exception as e:
        logger.warning(f"Could not fetch/clear existing items: {e}")

    # re-add all items from dataset.json
    added = 0
    for i, item in enumerate(items):
        msg = item["input"].get("message", "")
        try:
            langfuse_client.create_dataset_item(
                dataset_name=dataset_name,
                input=item["input"],
                expected_output=item["expected_output"]
            )
            logger.info(f"  Item {i+1}/{len(items)} added: {msg[:50]}")
            added += 1
        except Exception as e:
            logger.error(f"  Failed to add item {i+1}: {e}")

    flush()
    logger.info(f"Done — {added}/{len(items)} items in '{dataset_name}'")


if __name__ == "__main__":
    seed()
