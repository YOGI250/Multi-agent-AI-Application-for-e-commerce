"""
eval/setup_langfuse_evaluators.py

One-shot setup script that wires up the LangFuse Evaluator section:
  1. Creates score configs (metric schemas) via LangFuse REST API
  2. Creates evaluator prompt templates (LLM-as-judge prompts) as managed
     prompts in LangFuse — these are what you select when creating an
     Evaluator in the UI under Evaluations → Evaluators → New Evaluator

The evaluation dataset itself is seeded separately by eval/seed_dataset.py
(reads eval/dataset.json, run in CI Stage 3).

Usage:
    python eval/setup_langfuse_evaluators.py

After running this script, go to LangFuse UI:
  1. Settings → LLM Connections → New Connection
       - Provider: OpenAI (compatible)
       - Base URL: https://router.huggingface.co/v1
       - API Key: a Hugging Face access token (huggingface.co/settings/tokens)
       - Model: e.g. deepseek-ai/DeepSeek-V4-Flash (or any "Inference Available" chat model)
     This keeps the judge LLM's usage separate from the app's GROQ_API_KEY quota.
  2. Evaluations → Evaluators → New Evaluator
  3. Pick a template created by this script (e.g. eval_answer_relevancy)
  4. Set score name, pick the Hugging Face connection as the model, sampling, and save
"""

import sys
import os
import base64
import logging
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from langfuse import Langfuse

langfuse_client = Langfuse(
    public_key = settings.langfuse_public_key,
    secret_key = settings.langfuse_secret_key,
    host       = settings.langfuse_host,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── REST API helpers ──────────────────────────────────────────────────────────

LANGFUSE_HOST = settings.langfuse_host
_token = base64.b64encode(
    f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
).decode()
HEADERS = {
    "Authorization": f"Basic {_token}",
    "Content-Type": "application/json",
}


def _post(path: str, body: dict) -> dict | None:
    url = f"{LANGFUSE_HOST}{path}"
    try:
        r = requests.post(url, headers=HEADERS, json=body, timeout=10)
        if r.status_code in (200, 201):
            return r.json()
        if r.status_code == 409:
            logger.info(f"  Already exists: {path} ({body.get('name', '')})")
            return None
        logger.warning(f"  POST {path} → {r.status_code}: {r.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"  POST {path} failed: {e}")
        return None


# ── 1. Score configs ──────────────────────────────────────────────────────────

SCORE_CONFIGS = [
    {
        "name":      "answer_relevancy",
        "dataType":  "NUMERIC",
        "minValue":  0.0,
        "maxValue":  1.0,
        "description": "How relevant is the response to the user query? (0–1)"
    },
    {
        "name":      "faithfulness",
        "dataType":  "NUMERIC",
        "minValue":  0.0,
        "maxValue":  1.0,
        "description": "Is the response grounded in real data, no invented facts? (0–1)"
    },
    {
        "name":      "task_completion",
        "dataType":  "NUMERIC",
        "minValue":  0.0,
        "maxValue":  1.0,
        "description": "Did the agent fully complete the requested task? (0–1)"
    },
    {
        "name":      "correctness",
        "dataType":  "NUMERIC",
        "minValue":  0.0,
        "maxValue":  1.0,
        "description": "Does the response contain the correct expected information? (0–1)"
    },
    {
        "name":      "hallucination_free",
        "dataType":  "NUMERIC",
        "minValue":  0.0,
        "maxValue":  1.0,
        "description": "Are all order IDs, prices, and ticket IDs grounded in context? (0–1)"
    },
]


def create_score_configs():
    logger.info("Creating score configs …")
    for cfg in SCORE_CONFIGS:
        result = _post("/api/public/score-configs", cfg)
        if result:
            logger.info(f"  ✓ {cfg['name']}")
    logger.info("Score configs done.\n")


# ── 2. Evaluator prompt templates ─────────────────────────────────────────────

EVAL_TEMPLATES = [
    {
        "name": "eval_answer_relevancy",
        "prompt": (
            "You are an evaluation assistant.\n\n"
            "Score how relevant the AI response is to the user query.\n\n"
            "User query:\n{{input}}\n\n"
            "AI response:\n{{output}}\n\n"
            "Scoring guide (0–1):\n"
            "  1.0 — response directly and fully addresses the query\n"
            "  0.7 — mostly relevant, minor off-topic content\n"
            "  0.4 — tangentially related but misses the main point\n"
            "  0.0 — completely irrelevant\n\n"
            "Return ONLY valid JSON, no markdown:\n"
            '{{"score": <0.0-1.0>, "reason": "<one sentence>"}}'
        ),
        "labels": ["production"],
    },
    {
        "name": "eval_faithfulness",
        "prompt": (
            "You are an evaluation assistant.\n\n"
            "Check whether the AI response is grounded in the provided context "
            "and does not invent facts (order IDs, prices, product names, ticket IDs).\n\n"
            "User query:\n{{input}}\n\n"
            "AI response:\n{{output}}\n\n"
            "Scoring guide (0–1):\n"
            "  1.0 — every claim in the response is verifiable from the query/context\n"
            "  0.7 — mostly grounded, one minor unsupported detail\n"
            "  0.4 — several unverifiable claims\n"
            "  0.0 — response invents significant facts\n\n"
            "Return ONLY valid JSON, no markdown:\n"
            '{{"score": <0.0-1.0>, "reason": "<one sentence>"}}'
        ),
        "labels": ["production"],
    },
    {
        "name": "eval_task_completion",
        "prompt": (
            "You are an evaluation assistant.\n\n"
            "Assess whether the AI agent fully completed what the user asked for.\n\n"
            "User query:\n{{input}}\n\n"
            "AI response:\n{{output}}\n\n"
            "Scoring guide (0–1):\n"
            "  1.0 — task fully completed (order status provided, products listed, "
            "ticket created, etc.)\n"
            "  0.7 — task mostly completed, missing a minor detail\n"
            "  0.4 — partial completion, important parts missing\n"
            "  0.0 — task not completed at all\n\n"
            "Return ONLY valid JSON, no markdown:\n"
            '{{"score": <0.0-1.0>, "reason": "<one sentence>"}}'
        ),
        "labels": ["production"],
    },
    {
        "name": "eval_hallucination_free",
        "prompt": (
            "You are an evaluation assistant checking for hallucinations.\n\n"
            "Carefully check if the AI response invents any specific factual claims "
            "— especially order IDs (ORD-…), prices (₹…), ticket IDs, or product names "
            "— that are NOT present in the user query.\n\n"
            "User query:\n{{input}}\n\n"
            "AI response:\n{{output}}\n\n"
            "Scoring guide (0–1):\n"
            "  1.0 — no hallucinations, all specifics match the query context\n"
            "  0.5 — one possibly hallucinated detail\n"
            "  0.0 — response clearly invents order IDs, prices, or ticket IDs\n\n"
            "Return ONLY valid JSON, no markdown:\n"
            '{{"score": <0.0-1.0>, "reason": "<one sentence>"}}'
        ),
        "labels": ["production"],
    },
]


def create_eval_templates():
    logger.info("Creating evaluator prompt templates …")
    for tpl in EVAL_TEMPLATES:
        try:
            langfuse_client.create_prompt(
                name   = tpl["name"],
                prompt = tpl["prompt"],
                labels = tpl["labels"],
            )
            logger.info(f"  ✓ {tpl['name']}")
        except Exception as e:
            logger.info(f"  Already exists or error ({tpl['name']}): {e}")
    logger.info("Evaluator templates done.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("LangFuse Evaluator Setup")
    logger.info("=" * 60 + "\n")

    create_score_configs()
    create_eval_templates()

    logger.info("=" * 60)
    logger.info("Setup complete. Next steps in LangFuse UI (cloud.langfuse.com):")
    logger.info("")
    logger.info("  1. Settings → LLM Connections → New Connection")
    logger.info("       Provider: OpenAI (compatible)")
    logger.info("       Base URL: https://router.huggingface.co/v1")
    logger.info("       API Key:  a Hugging Face access token")
    logger.info("       Model:    e.g. deepseek-ai/DeepSeek-V4-Flash")
    logger.info("     (separate from GROQ_API_KEY — keeps judge LLM usage off the app's quota)")
    logger.info("")
    logger.info("  2. Evaluations → Evaluators → New Evaluator")
    logger.info("     For each metric, pick the matching template:")
    logger.info("       • eval_answer_relevancy  → score: answer_relevancy")
    logger.info("       • eval_faithfulness      → score: faithfulness")
    logger.info("       • eval_task_completion   → score: task_completion")
    logger.info("       • eval_hallucination_free → score: hallucination_free")
    logger.info("     Set sampling: 1.0, model: the Hugging Face connection from step 1")
    logger.info("")
    logger.info("  3. Evaluations → Datasets → ecommerce-eval-dataset")
    logger.info("     (seeded by eval/seed_dataset.py) — run the dataset against your pipeline")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
