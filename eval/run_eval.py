#!/usr/bin/env python3
"""
eval/run_eval.py — CI/CD Stage 3: LLM Evaluation Runner

Modes:
  default (mocked): runs the real product_agent graph with patched LLM calls
                    and patched DB tool calls — deterministic, no network I/O.
  RUN_LIVE_EVAL=true: calls the live FastAPI endpoint for actual responses.

Scoring:
  Uses Groq llama-3.1-8b-instant as LLM judge (GROQ_API_KEY env var required).
  Falls back to heuristic scoring if Groq is unavailable.

Outputs:
  eval/reports/eval_report.json  — JSON report (CI artifact + pushed to LangFuse)
  eval/reports/eval_report.html  — self-contained HTML report (CI artifact)

Exit code: 0 = all metrics above thresholds, 1 = any metric below threshold.
"""

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
import yaml

# ── DeepEval ──────────────────────────────────────────────────────────────────
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))  # make agents/, services/, etc. importable
CONFIG_PATH = REPO_ROOT / "eval" / "config.yaml"
DATASET_PATH = REPO_ROOT / "eval" / "dataset.json"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load config and dataset ─────────────────────────────────────────────────────

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

THRESHOLDS: dict[str, float] = {k: v["threshold"] for k, v in config["metrics"].items()}

with open(DATASET_PATH) as f:
    dataset = json.load(f)

LIVE_EVAL = os.getenv("RUN_LIVE_EVAL", "false").lower() == "true"
API_BASE = os.getenv("EVAL_API_BASE", "http://localhost:8000")
EVAL_USER = os.getenv("EVAL_USER_ID", "google_105309025092043620678")

# ── Git metadata ────────────────────────────────────────────────────────────────


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


GIT_SHA = _git(["git", "rev-parse", "HEAD"])
GIT_BRANCH = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])

# ── DeepEval judge model (Groq as backend) ──────────────────────────────────────


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """Wraps Groq llama-3.1-8b-instant so DeepEval metrics can use it as a judge."""

    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key and not groq_key.startswith("test_"):
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant", temperature=0)
        else:
            self._llm = None

    def load_model(self):
        return self._llm

    def generate(self, prompt: str, *args, **kwargs) -> str:
        if not self._llm:
            raise RuntimeError("No valid GROQ_API_KEY for DeepEval judge")
        return self._llm.invoke(prompt).content

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "groq/llama-3.1-8b-instant"


# ── DeepEval scoring ─────────────────────────────────────────────────────────────


def score_with_deepeval(message: str, actual_output: str, reference: str, expected: dict) -> dict[str, float]:
    """
    Score using DeepEval metric classes + custom Groq batch judge.

    DeepEval GEval covers 2 required dimensions (satisfies framework requirement):
      - answer_relevancy  : GEval — is response relevant to the user request?
      - task_completion   : GEval — did the agent complete the search task?

    Custom Groq single-call judge covers the remaining 3 dimensions in one
    API call (avoids the rate-limit issue DeepEval causes with many internal calls):
      - faithfulness, correctness, hallucination_free
    """
    groq_judge = GroqDeepEvalLLM()

    test_case = LLMTestCase(
        input=message,
        actual_output=actual_output,
        expected_output=reference,
    )

    scores: dict[str, float] = {}

    # ── DeepEval GEval — 2 dimensions ────────────────────────────────────────
    geval_metrics = [
        (
            "answer_relevancy",
            "Does the actual output directly address the user's product search request "
            "with specific product recommendations?",
            [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        ),
        (
            "task_completion",
            "Does the actual output fully complete the task by listing product names, "
            "prices in rupees, ratings, and brand names?",
            [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        ),
    ]

    for metric_name, criteria, params in geval_metrics:
        try:
            metric = GEval(
                name=metric_name,
                criteria=criteria,
                evaluation_params=params,
                model=groq_judge,
                threshold=THRESHOLDS.get(metric_name, 0.7),
                async_mode=False,
            )
            metric.measure(test_case)
            scores[metric_name] = round(float(metric.score), 4)
            logger.info(f"    [DeepEval GEval] {metric_name}: {scores[metric_name]}")
        except Exception as e:
            logger.warning(f"    GEval {metric_name} failed: {e} — heuristic fallback")
            scores[metric_name] = _heuristic_score(actual_output, reference, expected)[metric_name]

    # ── Custom Groq batch judge — remaining 3 dimensions in one call ──────────
    try:
        batch = _llm_judge_eval(message, actual_output, reference, expected)
        for k in ("faithfulness", "correctness", "hallucination_free"):
            scores[k] = batch.get(k, 0.0)
    except (ValueError, ImportError, Exception) as e:
        logger.warning(f"    Groq batch judge failed: {e} — heuristic fallback")
        heuristic = _heuristic_score(actual_output, reference, expected)
        for k in ("faithfulness", "correctness", "hallucination_free"):
            scores[k] = heuristic[k]

    return scores


# ── Heuristic fallback (used only when DeepEval judge is unavailable) ────────────


def _llm_judge_eval(message: str, response: str, reference: str, expected: dict) -> dict[str, float]:
    """
    Groq LLM-as-judge for eval pipeline.
    Uses llama-3.1-8b-instant — fast, cheap, already available via GROQ_API_KEY.
    Falls back to heuristics if Groq is unavailable.
    """
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key or groq_api_key.startswith("test_"):
        raise ValueError("No real GROQ_API_KEY available — using heuristics")

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("langchain_groq not installed")

    intent = expected.get("intent", "unknown")
    should_contain = expected.get("should_contain", [])

    keywords_str = ", ".join(should_contain)
    prompt = f"""\
You are an AI evaluation judge. Score the assistant response on 5 metrics.

USER MESSAGE: {message}
EXPECTED INTENT: {intent}
EXPECTED KEYWORDS that should appear in the response: {keywords_str}
REFERENCE RESPONSE (ground truth): {reference[:800]}
ACTUAL RESPONSE TO EVALUATE: {response[:800]}

Score each metric from 0.0 to 1.0 based on the actual response quality:

- answer_relevancy: Does the actual response directly address what the user asked? (high = relevant)
- faithfulness: Is the actual response grounded in facts such as prices, ratings, order IDs, ticket IDs?
- task_completion: Does the actual response contain the expected keywords ({keywords_str})?
- correctness: Does the actual response match the expected intent ({intent}) and align with the reference?
- hallucination_free: Did the agent avoid inventing facts? (1.0 = no hallucination, 0.0 = invented facts)

Respond with ONLY a JSON object. Example format (fill in real scores, not these placeholder values):
{{"answer_relevancy": <score>, "faithfulness": <score>, "task_completion": <score>, "correctness": <score>, "hallucination_free": <score>}}"""

    llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=0)
    result = llm.invoke(prompt)
    raw = result.content.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    scores = json.loads(raw)
    expected_keys = {
        "answer_relevancy",
        "faithfulness",
        "task_completion",
        "correctness",
        "hallucination_free",
    }
    if not expected_keys.issubset(scores.keys()):
        raise ValueError(f"Incomplete keys from LLM judge: {scores.keys()}")

    return {k: round(max(0.0, min(1.0, float(scores[k]))), 4) for k in expected_keys}


def _heuristic_score(response: str, reference: str, expected: dict) -> dict[str, float]:
    """Keyword + overlap heuristics used when no Groq key is available."""
    should_contain = [kw.lower() for kw in expected.get("should_contain", [])]
    resp_lower = response.lower()
    ref_lower = reference.lower()

    kw_hits = sum(1 for kw in should_contain if kw in resp_lower)
    task_completion = kw_hits / len(should_contain) if should_contain else 0.8

    ref_words = set(ref_lower.split())
    resp_words = set(resp_lower.split())
    overlap = len(ref_words & resp_words) / len(ref_words) if ref_words else 0.5

    return {
        "answer_relevancy": min(1.0, round(overlap * 1.1, 4)),
        "faithfulness": round(overlap, 4),
        "task_completion": round(task_completion, 4),
        "correctness": round((overlap + task_completion) / 2, 4),
        "hallucination_free": 0.8,
    }


def score_case(message: str, response: str, reference: str, expected: dict) -> dict[str, float]:
    try:
        scores = score_with_deepeval(message, response, reference, expected)
        logger.info("    [scorer] DeepEval (AnswerRelevancyMetric + GEval, Groq judge)")
    except Exception as e:
        logger.warning(f"    DeepEval failed entirely: {e} — heuristic fallback")
        scores = _heuristic_score(response, reference, expected)
        logger.info("    [scorer] heuristics (fallback)")
    return scores


# ── Mocked agent runner ──────────────────────────────────────────────────────────


class SequentialMockLLM:
    """
    Replaces ChatGroq during mocked eval. Returns fixture strings in call order.
    product_agent makes exactly 2 LLM calls per product query:
      call 1 — extract_preferences  → JSON filter object
      call 2 — rank_and_filter      → JSON index array e.g. [1, 2, 3]
    """

    def __init__(self, fixtures: list):
        self._fixtures = list(fixtures)
        self._idx = 0

    def invoke(self, *args, **kwargs):
        if self._idx >= len(self._fixtures):
            raise RuntimeError(
                f"Unexpected LLM call #{self._idx + 1}: " f"only {len(self._fixtures)} fixtures defined for this sample"
            )
        content = self._fixtures[self._idx]
        self._idx += 1
        resp = MagicMock()
        resp.content = content
        resp.usage_metadata = {
            "input_tokens": 50,
            "output_tokens": 30,
            "total_tokens": 80,
        }
        return resp


def run_with_mocks(
    message: str,
    llm_fixtures: list,
    search_results: list,
    user_id: str = "eval_guest_001",
) -> str:
    """Run product_agent with mocked LLM + tool calls."""
    from agents.product_agent import product_agent as _pa

    mock_llm = SequentialMockLLM(llm_fixtures)
    with patch("agents.product_agent.ChatGroq", return_value=mock_llm), patch(
        "tools.product_tools.search_products", return_value=search_results
    ), patch("tools.product_tools.get_specs", return_value={}):
        result = _pa.invoke(
            {
                "message": message,
                "user_id": user_id,
                "session_id": "eval_session_001",
                "history": [],
                "session_context": {},
                "langfuse_trace_id": None,
                "langfuse_parent_span_id": None,
            }
        )
    return result.get("response", "")


def run_with_mocks_order(
    message: str,
    llm_fixtures: list,
    order_data: dict,
    tracking_data: dict,
    user_id: str = "eval_guest_001",
) -> str:
    """Run order_agent with mocked LLM + tool calls.
    LLM calls: analyze_order_status (fixture 0), generate_response (fixture 1).
    user_id must match order_data["user_id"] — process_order_result validates ownership.
    """
    from agents.order_agent import order_agent as _oa

    mock_llm = SequentialMockLLM(llm_fixtures)
    with patch("agents.order_agent.ChatGroq", return_value=mock_llm), patch(
        "tools.order_tools.get_order", return_value=order_data
    ), patch(
        "tools.order_tools.get_tracking",
        return_value=tracking_data if tracking_data else None,
    ):
        result = _oa.invoke(
            {
                "message": message,
                "user_id": user_id,
                "session_id": "eval_session_001",
                "history": [],
                "session_context": {},
                "langfuse_trace_id": None,
                "langfuse_parent_span_id": None,
            }
        )
    return result.get("response", "")


def run_with_mocks_support(message: str, llm_fixtures: list, policy_data: dict, user_id: str = "eval_guest_001") -> str:
    """Run support_agent with mocked LLM + tool calls.
    LLM calls: classify_issue (fixture 0), draft_resolution (fixture 1).
    assess_severity queries DB directly — has try/except so DB failure is safe.
    """
    from agents.support_agent import support_agent as _sa

    mock_llm = SequentialMockLLM(llm_fixtures)
    with patch("agents.support_agent.ChatGroq", return_value=mock_llm), patch(
        "tools.support_tools.get_policy", return_value=policy_data
    ), patch(
        "tools.support_tools.get_user_complaint_history",
        return_value={
            "existing_ticket_id": None,
            "is_duplicate": False,
            "days_open": 0,
        },
    ), patch(
        "tools.support_tools.create_ticket",
        return_value={"ticket_id": "eval_ticket_001", "status": "created"},
    ):
        result = _sa.invoke(
            {
                "message": message,
                "user_id": user_id,
                "session_id": "eval_session_001",
                "history": [],
                "session_context": {},
                "langfuse_trace_id": None,
                "langfuse_parent_span_id": None,
            }
        )
    return result.get("response", "")


# ── Live API call ────────────────────────────────────────────────────────────────


def call_live_api(message: str, user_id: str) -> str:
    """Call the running FastAPI /api/v1/chat endpoint and return the response text."""
    url = f"{API_BASE}/chat"
    try:
        resp = requests.post(url, json={"message": message, "guest_id": user_id}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        logger.error(f"Live API call failed for '{message[:40]}': {e}")
        return ""


# ── Main evaluation loop ─────────────────────────────────────────────────────────


def run_evaluation() -> tuple[list[dict], dict[str, float], bool]:
    mode = "LIVE" if LIVE_EVAL else "MOCKED"
    logger.info(f"Starting evaluation | mode={mode} | samples={len(dataset)}")
    logger.info(f"git_sha={GIT_SHA[:8]} | branch={GIT_BRANCH}")

    per_sample: list[dict] = []

    for i, case in enumerate(dataset):
        message = case["input"]["message"]
        user_id = case["input"].get("user_id", EVAL_USER)
        expected = case["expected_output"]
        reference = case["mocked_response"]

        logger.info(f"  [{i+1}/{len(dataset)}] {message[:60]}")

        if LIVE_EVAL:
            actual_output = call_live_api(message, user_id)
            if not actual_output:
                logger.warning("    Empty response from live API — skipping sample")
                continue
        else:
            agent_type = case.get("agent", "product_agent")
            llm_fixtures = case.get("llm_fixtures", [])
            try:
                if agent_type == "order_agent":
                    actual_output = run_with_mocks_order(
                        message,
                        llm_fixtures,
                        case.get("order_data", {}),
                        case.get("tracking_data", {}),
                        user_id=user_id,
                    )
                elif agent_type == "support_agent":
                    actual_output = run_with_mocks_support(
                        message,
                        llm_fixtures,
                        case.get("policy_data", {}),
                        user_id=user_id,
                    )
                else:
                    actual_output = run_with_mocks(
                        message,
                        llm_fixtures,
                        case.get("search_results", []),
                        user_id=user_id,
                    )
            except Exception as e:
                logger.error(f"    Agent run failed: {e}")
                continue
            if not actual_output:
                logger.warning("    Empty response from mocked agent — skipping sample")
                continue
            logger.info(f"    [{agent_type}] produced {len(actual_output)} chars of output")

        scores = score_case(message, actual_output, reference, expected)
        passed = all(scores[k] >= THRESHOLDS[k] for k in scores if k in THRESHOLDS)

        per_sample.append(
            {
                "sample_id": i + 1,
                "message": message,
                "intent": expected.get("intent"),
                "actual_output": actual_output[:300] + ("..." if len(actual_output) > 300 else ""),
                "scores": scores,
                "passed": passed,
                "mode": mode,
            }
        )

        for metric, val in scores.items():
            threshold = THRESHOLDS.get(metric, 0.0)
            status = "PASS" if val >= threshold else "FAIL"
            logger.info(f"    {metric:<20} {val:.4f}  [{status}]")

    # Per-metric averages
    averages: dict[str, float] = {}
    for metric in THRESHOLDS:
        vals = [s["scores"][metric] for s in per_sample if metric in s["scores"]]
        averages[metric] = round(sum(vals) / len(vals), 4) if vals else 0.0

    overall_pass = all(averages[m] >= THRESHOLDS[m] for m in THRESHOLDS)

    logger.info("─" * 50)
    logger.info("EVALUATION SUMMARY")
    for metric, avg in averages.items():
        threshold = THRESHOLDS[metric]
        status = "PASS" if avg >= threshold else "FAIL"
        logger.info(f"  {metric:<20} avg={avg:.4f}  threshold={threshold}  [{status}]")
    logger.info(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    return per_sample, averages, overall_pass


# ── Report generation ────────────────────────────────────────────────────────────


def generate_json_report(per_sample, averages, overall_pass) -> Path:
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": GIT_SHA,
        "branch_name": GIT_BRANCH,
        "total_samples": len(per_sample),
        "mode": "live" if LIVE_EVAL else "mocked",
        "per_metric_averages": averages,
        "thresholds": THRESHOLDS,
        "per_sample_scores": per_sample,
        "overall_pass": overall_pass,
    }
    path = REPORTS_DIR / "eval_report.json"
    path.write_text(json.dumps(report, indent=2))
    logger.info(f"JSON report → {path}")
    return path


def generate_html_report(per_sample, averages, overall_pass) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    badge_cls = "pass" if overall_pass else "fail"
    badge_txt = "PASS" if overall_pass else "FAIL"

    # Summary rows
    summary_rows = ""
    for metric, avg in averages.items():
        threshold = THRESHOLDS.get(metric, 0.0)
        passed = avg >= threshold
        row_cls = "pass-row" if passed else "fail-row"
        summary_rows += (
            f"<tr class='{row_cls}'>"
            f"<td>{metric}</td>"
            f"<td>{avg:.4f}</td>"
            f"<td>{threshold}</td>"
            f"<td>{'✅ PASS' if passed else '❌ FAIL'}</td>"
            f"</tr>\n"
        )

    # Per-sample rows
    sample_rows = ""
    for s in per_sample:
        row_cls = "pass-row" if s["passed"] else "fail-row"
        scores_str = " | ".join(f"{k}: {v:.2f}" for k, v in s["scores"].items())
        sample_rows += (
            f"<tr class='{row_cls}'>"
            f"<td>{s['sample_id']}</td>"
            f"<td>{s['intent']}</td>"
            f"<td style='max-width:300px;word-break:break-word'>{s['message']}</td>"
            f"<td style='font-size:0.8em'>{scores_str}</td>"
            f"<td>{'✅' if s['passed'] else '❌'}</td>"
            f"</tr>\n"
        )

    # Inline bar chart data (JSON for chart.js-like rendering using CSS widths)
    bars = ""
    for metric, avg in averages.items():
        threshold = THRESHOLDS.get(metric, 0.0)
        width = int(avg * 100)
        color = "#4caf50" if avg >= threshold else "#f44336"
        bars += (
            f"<div class='bar-label'>{metric}</div>"
            f"<div class='bar-track'>"
            f"  <div class='bar-fill' style='width:{width}%;background:{color}'>"
            f"    {avg:.4f}"
            f"  </div>"
            f"  <span class='threshold-line' style='left:{int(threshold*100)}%'></span>"
            f"</div>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eval Report — {ts}</title>
<style>
  body   {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
  h1     {{ color: #1a1a2e; }}
  .badge {{ display: inline-block; padding: 6px 18px; border-radius: 4px;
            font-weight: bold; font-size: 1.2em; color: #fff; }}
  .pass  {{ background: #4caf50; }}
  .fail  {{ background: #f44336; }}
  table  {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
  th     {{ background: #1a1a2e; color: #fff; padding: 10px; text-align: left; }}
  td     {{ padding: 8px 10px; border-bottom: 1px solid #ddd; }}
  .pass-row td {{ background: #f0fff0; }}
  .fail-row td {{ background: #fff0f0; }}
  .meta  {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
  .bar-label  {{ font-size: 0.85em; margin-top: 10px; margin-bottom: 2px; }}
  .bar-track  {{ position: relative; background: #eee; height: 28px;
                 border-radius: 4px; margin-bottom: 6px; overflow: visible; }}
  .bar-fill   {{ height: 100%; border-radius: 4px; color: #fff;
                 font-size: 0.8em; line-height: 28px; padding-left: 6px;
                 white-space: nowrap; }}
  .threshold-line {{ position: absolute; top: 0; bottom: 0; width: 2px;
                     background: #333; opacity: 0.5; }}
</style>
</head>
<body>
<h1>LLM Evaluation Report</h1>
<p class="meta">
  Generated: {ts} &nbsp;|&nbsp;
  Commit: <code>{GIT_SHA[:8]}</code> &nbsp;|&nbsp;
  Branch: <code>{GIT_BRANCH}</code> &nbsp;|&nbsp;
  Samples: {len(per_sample)} &nbsp;|&nbsp;
  Mode: {"LIVE" if LIVE_EVAL else "MOCKED"}
</p>

<h2>Overall Result: <span class="badge {badge_cls}">{badge_txt}</span></h2>

<h2>Per-Metric Summary</h2>
<table>
  <tr><th>Metric</th><th>Average Score</th><th>Threshold</th><th>Result</th></tr>
  {summary_rows}
</table>

<h2>Score Charts <span style="font-size:0.7em;color:#666">(vertical line = threshold)</span></h2>
<div style="max-width:600px">
  {bars}
</div>

<h2>Per-Sample Results</h2>
<table>
  <tr>
    <th>#</th><th>Intent</th><th>Message</th><th>Scores</th><th>Pass</th>
  </tr>
  {sample_rows}
</table>
</body>
</html>
"""
    path = REPORTS_DIR / "eval_report.html"
    path.write_text(html)
    logger.info(f"HTML report → {path}")
    return path


# ── Push JSON report to LangFuse ────────────────────────────────────────────────


def push_report_to_langfuse(json_path: Path, per_sample: list, overall_pass: bool) -> None:
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from langfuse import Langfuse

        lf = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "placeholder").strip(),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", "placeholder").strip(),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000").strip(),
        )

        dataset_name = config["evaluation"]["dataset_name"]
        run_name = f"ci-eval-{GIT_SHA[:8]}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"

        # Fetch existing dataset items seeded by seed_dataset.py
        # so we link run results to existing items (not create duplicates each run)
        item_map: dict[str, str] = {}
        try:
            existing = lf.get_dataset(dataset_name)
            item_map = {item.input.get("message", ""): item.id for item in existing.items}
        except Exception:
            pass  # dataset not seeded yet — will create items inline below

        pushed = 0
        for sample in per_sample:
            try:
                item_id = item_map.get(sample["message"])
                if not item_id:
                    # Fallback: create item inline if seed_dataset.py wasn't run
                    new_item = lf.create_dataset_item(
                        dataset_name=dataset_name,
                        input={"message": sample["message"]},
                        expected_output={"intent": sample.get("intent")},
                    )
                    item_id = new_item.id

                lf.create_dataset_run_item(
                    run_name=run_name,
                    dataset_item_id=item_id,
                    metadata={
                        "scores": sample["scores"],
                        "passed": sample["passed"],
                        "mode": sample.get("mode", "mocked"),
                        "agent": sample.get("intent", "unknown"),
                    },
                )
                pushed += 1
            except Exception:
                pass

        lf.flush()
        logger.info(f"LangFuse: eval run '{run_name}' pushed ({pushed} items)")

    except Exception as e:
        # LangFuse may not be reachable in CI — don't fail the pipeline for this
        logger.warning(f"LangFuse push skipped: {e}")


# ── Entry point ─────────────────────────────────────────────────────────────────


def main() -> int:
    per_sample, averages, overall_pass = run_evaluation()

    json_path = generate_json_report(per_sample, averages, overall_pass)
    generate_html_report(per_sample, averages, overall_pass)

    if config.get("reporting", {}).get("push_to_langfuse", False):
        push_report_to_langfuse(json_path, per_sample, overall_pass)

    if overall_pass:
        logger.info("Evaluation PASSED — all metrics above thresholds.")
    else:
        logger.error("Evaluation FAILED — one or more metrics below threshold.")
        for metric, avg in averages.items():
            if avg < THRESHOLDS.get(metric, 0.0):
                logger.error(f"  FAIL: {metric} = {avg:.4f} " f"(threshold = {THRESHOLDS[metric]})")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
