# langfuse_helpers/scoring.py
import re
import json
import logging
from langchain_groq import ChatGroq
from langfuse_helpers.tracing import langfuse_client
from monitoring.metrics import record_error
from config.settings import settings

logger = logging.getLogger(__name__)

_SCORER_MODEL = "llama-3.1-8b-instant"

_SCORE_PROMPT = """\
You are an AI evaluation judge. Score the assistant response on 5 metrics.

USER MESSAGE: {message}
AGENT USED: {agent_used}
ASSISTANT RESPONSE: {response}

Score each metric from 0.0 to 1.0 based on the actual response quality:

- answer_relevancy: Does the response directly address what the user asked? (high = relevant)
- faithfulness: Is the response grounded in real data such as prices with ₹, ratings with ⭐, order IDs, ticket IDs?
- completeness: Does the response fully answer the question?
- task_completion: Did the agent complete its specific task? Order agent = gave order status. Product agent = gave recommendations with prices. Support agent = gave resolution or created ticket.
- hallucination: Did the agent invent facts not in the conversation? (0.0 = no hallucination (GOOD), 1.0 = invented facts (BAD))

Respond with ONLY a JSON object. Example format (fill in real scores, not these placeholder values):
{{"answer_relevancy": <score>, "faithfulness": <score>, "completeness": <score>, "task_completion": <score>, "hallucination": <score>}}"""


def score_response(trace_id: str, agent_used: str, message: str, response: str, observation_id: str = None):
    """
    Groq LLM-as-judge scoring. Called after every agent run in api/routes.py.
    Scores are visible in LangFuse UI per trace, linked to the generation span when observation_id is provided.
    """
    try:
        scores = _llm_judge(agent_used, message, response)

        for metric_name, score_value in scores.items():
            langfuse_client.create_score(
                trace_id=trace_id,
                observation_id=observation_id,
                name=metric_name,
                value=score_value,
                comment=f"LLM-judged (Groq {_SCORER_MODEL}) for {agent_used}",
            )

        logger.info(f"Scores submitted for trace {trace_id}: {scores}")

    except Exception as e:
        logger.error(f"Error scoring response: {e}")
        record_error(error_type=type(e).__name__, agent_used=agent_used)


def _llm_judge(agent_used: str, message: str, response: str) -> dict:
    """Call Groq LLM to score the response on 5 metrics."""
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=_SCORER_MODEL,
        temperature=0,
    )

    prompt = _SCORE_PROMPT.format(
        message=message,
        agent_used=agent_used,
        response=response[:1500],
    )

    result = llm.invoke(prompt)
    raw = result.content.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    scores = json.loads(raw)

    expected_keys = {"answer_relevancy", "faithfulness", "completeness", "task_completion", "hallucination"}
    if not expected_keys.issubset(scores.keys()):
        raise ValueError(f"LLM returned incomplete keys: {scores.keys()}")

    return {k: round(max(0.0, min(1.0, float(scores[k]))), 2) for k in expected_keys}
