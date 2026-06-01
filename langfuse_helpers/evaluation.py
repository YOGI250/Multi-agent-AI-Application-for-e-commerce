# langfuse_helpers/evaluation.py

import datetime
import logging

from langfuse_helpers.tracing import langfuse_client, flush
from agents.intent_router import intent_router_graph
from langfuse_helpers.tracing import create_trace
from config.settings import settings

logger = logging.getLogger(__name__)


def run_evaluation(dataset_name: str = "ecommerce-eval-dataset"):
    """Runs one evaluation against the LangFuse dataset using the v4 run_experiment API."""
    logger.info(f"Starting evaluation run on dataset: {dataset_name}")

    dataset = langfuse_client.get_dataset(dataset_name)
    logger.info(f"Dataset loaded: {len(dataset.items)} items")

    run_name = f"eval-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def task(*, item, **kwargs):
        input_data = item.input if hasattr(item, "input") else item.get("input", {})
        expected = item.expected_output if hasattr(item, "expected_output") else item.get("expected_output", {})
        message = input_data.get("message", "")
        user_id = input_data.get("user_id", settings.eval_user_id) or "eval_guest"

        trace = create_trace(
            session_id=f"eval-{message[:20]}",
            user_id=user_id,
            is_authenticated=True,
            message=message,
        )

        result = intent_router_graph.invoke(
            {
                "message": message,
                "user_id": user_id,
                "session_id": f"eval-{message[:20]}",
                "history": [],
                "is_authenticated": True,
                "langfuse_trace_id": trace.id,
                "langfuse_parent_span_id": None,
            }
        )

        response = result.get("response", "")
        agent_used = result.get("agent_used", "unknown")
        intent = result.get("intent", "unknown")

        trace._span.set_trace_io(output={"response": response, "agent_used": agent_used})

        expected_contains = (expected or {}).get("should_contain", [])
        matches = sum(1 for kw in expected_contains if kw.lower() in response.lower())
        keyword_score = matches / len(expected_contains) if expected_contains else 1.0

        expected_intent = (expected or {}).get("intent", "")
        intent_correct = 1.0 if intent == expected_intent else 0.0

        langfuse_client.create_score(
            trace_id=trace.id,
            name="keyword_coverage",
            value=keyword_score,
            comment=f"matched {matches}/{len(expected_contains)} keywords",
        )
        langfuse_client.create_score(
            trace_id=trace.id,
            name="intent_accuracy",
            value=intent_correct,
            comment=f"expected={expected_intent} got={intent}",
        )

        flush()
        logger.info(f"agent={agent_used} | keyword={keyword_score:.2f} | intent={intent_correct:.0f}")
        return {"response": response, "trace_id": trace.id}

    langfuse_client.run_experiment(
        name=dataset_name,
        run_name=run_name,
        data=dataset.items,
        task=task,
    )

    logger.info("Evaluation run completed")
    logger.info(f"View results at: http://localhost:3000 → Datasets → {dataset_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
