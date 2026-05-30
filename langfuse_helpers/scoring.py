# langfuse_helpers/scoring.py
import re
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
            langfuse_client.create_score(
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
    Calculates scores using heuristics per agent type.
    Scores are between 0.0 and 1.0.

    1. answer_relevancy  — does response address the message topic?
    2. faithfulness      — does response reference actual data?
    3. completeness      — does response fully answer the question?
    4. task_completion   — did agent complete its specific task?
    5. hallucination     — did agent make up data not grounded in context?
    """
    scores = {}

    message_lower  = message.lower()
    response_lower = response.lower()

    # ── 1. Answer Relevancy ──
    # Check if key topic words from message appear in response
    # Filter out stop words first for meaningful overlap
    stop_words = {"the", "is", "a", "an", "my", "i", "me",
                  "for", "to", "of", "in", "it", "do", "can",
                  "you", "we", "and", "or", "with", "this",
                  "that", "what", "how", "why", "when", "where"}

    message_words  = set(message_lower.split()) - stop_words
    response_words = set(response_lower.split()) - stop_words
    common_words   = message_words & response_words

    if message_words:
        relevancy = min(len(common_words) / len(message_words), 1.0)
        scores["answer_relevancy"] = round(max(relevancy, 0.4), 2)
    else:
        scores["answer_relevancy"] = 0.8

    # ── 2. Faithfulness ──
    # Is the response grounded in actual data?
    # Check for concrete data references — not just length
    if agent_used == "order_agent":
        # faithful if response mentions actual order ID or status
        has_order_id = bool(
            re.search(r"ORD-[A-Z0-9-]+", response)
        )
        has_status = any(
            s in response_lower
            for s in ["delivered", "shipped", "delayed",
                      "cancelled", "processing", "transit"]
        )
        if has_order_id and has_status:
            scores["faithfulness"] = 1.0
        elif has_order_id or has_status:
            scores["faithfulness"] = 0.7
        else:
            scores["faithfulness"] = 0.4

    elif agent_used == "product_agent":
        # faithful if response mentions actual prices and ratings
        has_price  = "₹" in response
        has_rating = "⭐" in response
        if has_price and has_rating:
            scores["faithfulness"] = 1.0
        elif has_price or has_rating:
            scores["faithfulness"] = 0.7
        else:
            scores["faithfulness"] = 0.4

    elif agent_used == "support_agent":
        # faithful if response references policy or ticket ID
        has_policy    = "policy" in response_lower
        has_ticket_id = bool(
            re.search(r"[A-Z0-9]{8}", response)
        )
        if has_policy and has_ticket_id:
            scores["faithfulness"] = 1.0
        elif has_policy or has_ticket_id:
            scores["faithfulness"] = 0.7
        else:
            scores["faithfulness"] = 0.5

    else:
        scores["faithfulness"] = 0.8

    # ── 3. Completeness ──
    # Does the response actually answer what was asked?
    # Check for question-answer alignment not just length
    question_words = ["what", "where", "when", "how", "why",
                      "which", "who", "show", "list", "find"]
    is_question = any(w in message_lower for w in question_words)

    if is_question:
        # response should be substantive for a question
        word_count = len(response.split())
        if word_count > 80:
            scores["completeness"] = 1.0
        elif word_count > 40:
            scores["completeness"] = 0.8
        elif word_count > 20:
            scores["completeness"] = 0.6
        else:
            scores["completeness"] = 0.3
    else:
        # for commands like "cancel my order", shorter responses are fine
        scores["completeness"] = 0.9 if len(response) > 50 else 0.5

    # ── 4. Task Completion ──
    # Did the agent complete its specific task?
    if agent_used == "order_agent":
        has_order_ref  = bool(
            re.search(r"ORD-[A-Z0-9-]+", response)
        )
        has_status_info = any(
            s in response_lower
            for s in ["delivered", "shipped", "delayed",
                      "processing", "transit", "expected",
                      "arrival", "delivery"]
        )
        score = 0.4
        if has_order_ref:   score += 0.3
        if has_status_info: score += 0.3
        scores["task_completion"] = round(score, 2)

    elif agent_used == "product_agent":
        has_recommendations = any(
            w in response_lower
            for w in ["recommend", "here are", "top", "best"]
        )
        has_price  = "₹" in response
        has_rating = "⭐" in response
        score = 0.3
        if has_recommendations: score += 0.2
        if has_price:           score += 0.25
        if has_rating:          score += 0.25
        scores["task_completion"] = round(score, 2)

    elif agent_used == "support_agent":
        has_acknowledgement = any(
            w in response_lower
            for w in ["sorry", "apolog", "understand",
                      "assist", "help", "resolve"]
        )
        has_action = any(
            w in response_lower
            for w in ["ticket", "refund", "replacement",
                      "cancel", "policy", "team"]
        )
        score = 0.3
        if has_acknowledgement: score += 0.3
        if has_action:          score += 0.4
        scores["task_completion"] = round(score, 2)

    else:
        scores["task_completion"] = 0.8

    # ── 5. Hallucination ──
    # Did the agent make up data not grounded in actual context?
    # Check for fabricated specifics per agent type
    if agent_used == "order_agent":
        # hallucination if response invents order details
        # good sign: mentions real order ID from message
        order_id_in_message = re.search(
            r"ORD-[A-Z0-9-]+", message
        )
        order_id_in_response = re.search(
            r"ORD-[A-Z0-9-]+", response
        )
        if order_id_in_message and order_id_in_response:
            # both have order ID — check they match
            if order_id_in_message.group() == order_id_in_response.group():
                scores["hallucination"] = 0.0  # 0 = no hallucination
            else:
                scores["hallucination"] = 1.0  # invented different order ID
        elif order_id_in_response and not order_id_in_message:
            # response mentions order ID not in message — possible hallucination
            scores["hallucination"] = 0.5
        else:
            scores["hallucination"] = 0.0

    elif agent_used == "product_agent":
        # hallucination if response mentions prices without ₹ symbol
        # or invents ratings not in standard format
        has_proper_price  = "₹" in response
        has_proper_rating = "⭐" in response
        if has_proper_price and has_proper_rating:
            scores["hallucination"] = 0.0
        elif "price" in response_lower and not has_proper_price:
            scores["hallucination"] = 0.5
        else:
            scores["hallucination"] = 0.0

    elif agent_used == "support_agent":
        # hallucination if response invents a ticket ID
        # that doesn't match expected format
        ticket_match = re.search(
            r"\b[A-Z0-9]{8}\b", response
        )
        if ticket_match:
            # ticket ID present — likely grounded (created by system)
            scores["hallucination"] = 0.0
        elif "ticket" in response_lower and not ticket_match:
            # mentions ticket but no valid ID — possible hallucination
            scores["hallucination"] = 0.3
        else:
            scores["hallucination"] = 0.0

    else:
        scores["hallucination"] = 0.0

    return scores