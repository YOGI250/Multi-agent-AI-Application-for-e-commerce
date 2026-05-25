# langfuse_helpers/scoring.py

import logging
from langfuse_helpers.tracing import langfuse_client

logger = logging.getLogger(__name__)


def score_response(
    trace_id:   str,
    agent_used: str,
    message:    str,
    response:   str
):
    """
    Auto-scores every agent response on 4 metrics.
    Called after every agent run in api/routes.py.
    Scores are visible in LangFuse UI per trace.

    Metrics:
    1. answer_relevancy  — is response relevant to message?
    2. faithfulness      — is response grounded in data?
    3. completeness      — does response fully answer question?
    4. task_completion   — did agent complete the task?
    """
    try:
        scores = _calculate_scores(agent_used, message, response)

        for metric_name, score_value in scores.items():
            langfuse_client.score(
                trace_id = trace_id,
                name     = metric_name,
                value    = score_value,
                comment  = f"Auto-scored for {agent_used}"
            )

        logger.info(
            f"Scores submitted for trace {trace_id}: {scores}"
        )

    except Exception as e:
        logger.error(f"Error scoring response: {e}")


def _calculate_scores(
    agent_used: str,
    message:    str,
    response:   str
) -> dict:
    """
    Calculates scores using simple heuristics.
    In production these would use DeepEval or LLM judges.
    For now — rule based scoring for demo.
    """
    scores = {}

    # ── 1. Answer Relevancy ──
    # Does response contain keywords from the message?
    message_words = set(message.lower().split())
    response_words = set(response.lower().split())
    common_words = message_words & response_words
    relevancy = min(len(common_words) / max(len(message_words), 1), 1.0)
    scores["answer_relevancy"] = round(max(relevancy, 0.5), 2)

    # ── 2. Faithfulness ──
    # Did agent respond (not error out)?
    if response and len(response) > 50:
        scores["faithfulness"] = 1.0
    elif response and len(response) > 20:
        scores["faithfulness"] = 0.7
    else:
        scores["faithfulness"] = 0.3

    # ── 3. Completeness ──
    # Is the response long enough to be complete?
    if len(response) > 200:
        scores["completeness"] = 1.0
    elif len(response) > 100:
        scores["completeness"] = 0.8
    elif len(response) > 50:
        scores["completeness"] = 0.6
    else:
        scores["completeness"] = 0.3

    # ── 4. Task Completion ──
    # Agent specific checks
    if agent_used == "order_agent":
        # good if response mentions order status keywords
        keywords = ["delivered", "shipped", "delayed",
                    "cancelled", "processing", "order"]
        found = sum(1 for k in keywords if k in response.lower())
        scores["task_completion"] = min(found / 3, 1.0)

    elif agent_used == "product_agent":
        # good if response contains price and rating info
        has_price  = "₹" in response or "price" in response.lower()
        has_rating = "⭐" in response or "rating" in response.lower()
        score = 0.5
        if has_price:  score += 0.25
        if has_rating: score += 0.25
        scores["task_completion"] = score

    elif agent_used == "support_agent":
        # good if response mentions policy or ticket
        has_policy = "policy" in response.lower()
        has_ticket = "ticket" in response.lower()
        score = 0.5
        if has_policy: score += 0.25
        if has_ticket: score += 0.25
        scores["task_completion"] = score

    else:
        scores["task_completion"] = 0.8

    return scores
