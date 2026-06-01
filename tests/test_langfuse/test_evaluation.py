# tests/test_langfuse/test_evaluation.py

from unittest.mock import patch, MagicMock


def _make_mock_item(item_id="item_001", keyword="laptop", intent="product_query"):
    item = MagicMock()
    item.id = item_id
    item.input = {"message": "show me laptops", "user_id": "eval_user"}
    item.expected_output = {
        "should_contain": [keyword],
        "intent": intent,
    }
    return item


def _make_mock_trace():
    trace = MagicMock()
    trace.id = "trace_xyz"
    trace._span = MagicMock()
    return trace


def _fake_run_experiment(mock_client):
    """
    Side-effect for mock_client.run_experiment that actually calls the task
    function for each item, mirroring what the real v4 SDK does internally.
    """

    def _run(*, name, run_name, data, task, **kwargs):
        for item in data:
            try:
                task(item=item)
            except Exception:
                pass

    mock_client.run_experiment.side_effect = _run


class TestRunEvaluation:

    def test_invokes_graph_for_each_item(self):
        from langfuse_helpers import evaluation

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.items = [_make_mock_item("i1"), _make_mock_item("i2")]
        mock_client.get_dataset.return_value = mock_dataset
        _fake_run_experiment(mock_client)

        with patch.object(evaluation, "langfuse_client", mock_client), \
             patch.object(evaluation, "create_trace", return_value=_make_mock_trace()), \
             patch.object(evaluation, "intent_router_graph") as mock_graph, \
             patch.object(evaluation, "flush"):

            mock_graph.invoke.return_value = {
                "response": "Here are laptops",
                "agent_used": "product_agent",
                "intent": "product_query",
            }
            evaluation.run_evaluation("test-dataset")

        assert mock_graph.invoke.call_count == 2

    def test_submits_two_scores_per_item(self):
        from langfuse_helpers import evaluation

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.items = [_make_mock_item()]
        mock_client.get_dataset.return_value = mock_dataset
        _fake_run_experiment(mock_client)

        with patch.object(evaluation, "langfuse_client", mock_client), \
             patch.object(evaluation, "create_trace", return_value=_make_mock_trace()), \
             patch.object(evaluation, "intent_router_graph") as mock_graph, \
             patch.object(evaluation, "flush"):

            mock_graph.invoke.return_value = {
                "response": "Here are laptops",
                "agent_used": "product_agent",
                "intent": "product_query",
            }
            evaluation.run_evaluation("test-dataset")

        assert mock_client.create_score.call_count == 2

    def test_keyword_score_1_when_all_keywords_match(self):
        from langfuse_helpers import evaluation

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.items = [_make_mock_item(keyword="laptop")]
        mock_client.get_dataset.return_value = mock_dataset
        _fake_run_experiment(mock_client)
        score_values = []

        def capture_score(**kwargs):
            score_values.append(kwargs["value"])

        mock_client.create_score.side_effect = capture_score

        with patch.object(evaluation, "langfuse_client", mock_client), \
             patch.object(evaluation, "create_trace", return_value=_make_mock_trace()), \
             patch.object(evaluation, "intent_router_graph") as mock_graph, \
             patch.object(evaluation, "flush"):

            mock_graph.invoke.return_value = {
                "response": "Here are the best laptops",
                "agent_used": "product_agent",
                "intent": "product_query",
            }
            evaluation.run_evaluation("test-dataset")

        keyword_score = score_values[0]
        assert keyword_score == 1.0

    def test_intent_score_0_when_intent_mismatch(self):
        from langfuse_helpers import evaluation

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.items = [_make_mock_item(intent="product_query")]
        mock_client.get_dataset.return_value = mock_dataset
        _fake_run_experiment(mock_client)
        score_values = []

        def capture_score(**kwargs):
            score_values.append(kwargs["value"])

        mock_client.create_score.side_effect = capture_score

        with patch.object(evaluation, "langfuse_client", mock_client), \
             patch.object(evaluation, "create_trace", return_value=_make_mock_trace()), \
             patch.object(evaluation, "intent_router_graph") as mock_graph, \
             patch.object(evaluation, "flush"):

            mock_graph.invoke.return_value = {
                "response": "Here are laptops",
                "agent_used": "product_agent",
                "intent": "order_query",  # wrong intent
            }
            evaluation.run_evaluation("test-dataset")

        intent_score = score_values[1]
        assert intent_score == 0.0

    def test_agent_exception_is_caught(self):
        from langfuse_helpers import evaluation

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.items = [_make_mock_item()]
        mock_client.get_dataset.return_value = mock_dataset
        _fake_run_experiment(mock_client)

        with patch.object(evaluation, "langfuse_client", mock_client), \
             patch.object(evaluation, "create_trace", return_value=_make_mock_trace()), \
             patch.object(evaluation, "intent_router_graph") as mock_graph, \
             patch.object(evaluation, "flush"):

            mock_graph.invoke.side_effect = Exception("LLM timeout")
            evaluation.run_evaluation("test-dataset")  # must not raise

        mock_client.create_score.assert_not_called()

    def test_empty_dataset_runs_without_error(self):
        from langfuse_helpers import evaluation

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.items = []
        mock_client.get_dataset.return_value = mock_dataset
        _fake_run_experiment(mock_client)

        with patch.object(evaluation, "langfuse_client", mock_client), \
             patch.object(evaluation, "flush"):
            evaluation.run_evaluation("empty-dataset")

        mock_client.create_score.assert_not_called()

    def test_flush_called_after_all_items(self):
        from langfuse_helpers import evaluation

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.items = [_make_mock_item()]
        mock_client.get_dataset.return_value = mock_dataset
        _fake_run_experiment(mock_client)
        mock_flush = MagicMock()

        with patch.object(evaluation, "langfuse_client", mock_client), \
             patch.object(evaluation, "create_trace", return_value=_make_mock_trace()), \
             patch.object(evaluation, "intent_router_graph") as mock_graph, \
             patch.object(evaluation, "flush", mock_flush):

            mock_graph.invoke.return_value = {
                "response": "laptops",
                "agent_used": "product_agent",
                "intent": "product_query",
            }
            evaluation.run_evaluation("test-dataset")

        mock_flush.assert_called_once()

    def test_run_experiment_called_with_dataset_items(self):
        from langfuse_helpers import evaluation

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_item = _make_mock_item()
        mock_dataset.items = [mock_item]
        mock_client.get_dataset.return_value = mock_dataset

        with patch.object(evaluation, "langfuse_client", mock_client), \
             patch.object(evaluation, "flush"):
            evaluation.run_evaluation("test-dataset")

        mock_client.run_experiment.assert_called_once()
        call_kwargs = mock_client.run_experiment.call_args.kwargs
        assert call_kwargs["name"] == "test-dataset"
        assert call_kwargs["data"] == [mock_item]
        assert callable(call_kwargs["task"])
