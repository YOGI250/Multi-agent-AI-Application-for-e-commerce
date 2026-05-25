# langfuse_helpers/evaluation.py

import logging
from langfuse_helpers.tracing import langfuse_client, flush
from agents.intent_router import intent_router_graph
from langfuse_helpers.tracing import create_trace

logger = logging.getLogger(__name__)


def run_evaluation(dataset_name: str = "ecommerce-eval-dataset"):
    """
    Runs one evaluation against the LangFuse dataset.
    For each item:
      - runs the full agent pipeline
      - scores the response
      - links the run to the dataset item
    """
    logger.info(f"Starting evaluation run on dataset: {dataset_name}")

    # fetch dataset from LangFuse
    dataset = langfuse_client.get_dataset(dataset_name)
    logger.info(f"Dataset loaded: {len(dataset.items)} items")

    import datetime
    run_name = f"eval-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    for item in dataset.items:
        logger.info(f"Evaluating item: {item.id}")

        input_data    = item.input
        expected      = item.expected_output
        message       = input_data.get("message", "")
        user_id       = input_data.get(
            "user_id", "google_105309025092043620678"
        )

        # create a trace for this eval run
        trace = create_trace(
            session_id       = f"eval-{item.id}",
            user_id          = user_id,
            is_authenticated = True,
            message          = message
        )

        # run the agent
        try:
            result = intent_router_graph.invoke({
                "message":                 message,
                "user_id":                 user_id,
                "session_id":              f"eval-{item.id}",
                "history":                 [],
                "is_authenticated":        True,
                "langfuse_trace_id":       trace.id,
                "langfuse_parent_span_id": None
            })

            response   = result.get("response", "")
            agent_used = result.get("agent_used", "unknown")
            intent     = result.get("intent", "unknown")

            langfuse_client.trace(
                id     = trace.id,
                output = {"response": response, "agent_used": agent_used}
            )

            # score — check if expected keywords appear in response
            expected_contains = expected.get("should_contain", [])
            matches = sum(
                1 for keyword in expected_contains
                if keyword.lower() in response.lower()
            )
            keyword_score = (
                matches / len(expected_contains)
                if expected_contains else 1.0
            )

            # score — check if intent matches
            expected_intent = expected.get("intent", "")
            intent_correct  = 1.0 if intent == expected_intent else 0.0

            # submit scores to LangFuse
            langfuse_client.score(
                trace_id = trace.id,
                name     = "keyword_coverage",
                value    = keyword_score,
                comment  = f"matched {matches}/{len(expected_contains)} keywords"
            )

            langfuse_client.score(
                trace_id = trace.id,
                name     = "intent_accuracy",
                value    = intent_correct,
                comment  = f"expected={expected_intent} got={intent}"
            )

            # link run to dataset item
            item.link(
                trace,
                run_name,
                run_description = "Automated evaluation run 1"
            )

            logger.info(
                f"Item {item.id} | agent={agent_used} | "
                f"keyword_score={keyword_score:.2f} | "
                f"intent_correct={intent_correct:.0f}"
            )

        except Exception as e:
            logger.error(f"Error evaluating item {item.id}: {e}")

    flush()
    logger.info("Evaluation run completed")
    logger.info(
        f"View results at: "
        f"http://localhost:3000 → Datasets → {dataset_name}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()