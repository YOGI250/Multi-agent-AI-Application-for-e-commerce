# tests/test_langfuse/test_tracing.py

from unittest.mock import patch, MagicMock


class TestCalculateCost:

    def test_zero_tokens(self):
        from langfuse_helpers.tracing import calculate_cost
        assert calculate_cost(0, 0) == 0.0

    def test_positive_tokens_returns_float(self):
        from langfuse_helpers.tracing import calculate_cost
        cost = calculate_cost(1_000_000, 1_000_000)
        assert isinstance(cost, float)
        assert cost > 0

    def test_only_input_tokens(self):
        from langfuse_helpers.tracing import calculate_cost
        cost = calculate_cost(500_000, 0)
        assert cost > 0

    def test_only_output_tokens(self):
        from langfuse_helpers.tracing import calculate_cost
        cost = calculate_cost(0, 500_000)
        assert cost > 0


class TestExtractTokenUsage:

    def test_normal_usage_metadata(self):
        from langfuse_helpers.tracing import extract_token_usage
        mock_resp = MagicMock()
        mock_resp.usage_metadata = {
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15
        }
        result = extract_token_usage(mock_resp)
        assert result == {"input": 10, "output": 5, "total": 15}

    def test_empty_metadata_returns_zeros(self):
        from langfuse_helpers.tracing import extract_token_usage
        mock_resp = MagicMock()
        mock_resp.usage_metadata = {}
        result = extract_token_usage(mock_resp)
        assert result == {"input": 0, "output": 0, "total": 0}

    def test_invalid_metadata_triggers_fallback(self):
        from langfuse_helpers.tracing import extract_token_usage
        mock_resp = MagicMock()
        # A string has no .get() — triggers AttributeError inside try
        mock_resp.usage_metadata = "not-a-dict"
        result = extract_token_usage(mock_resp)
        assert result == {"input": 0, "output": 0, "total": 0}


class TestTraceHandle:

    def test_id_attribute_is_set(self):
        from langfuse_helpers.tracing import TraceHandle
        mock_span = MagicMock()
        handle = TraceHandle(mock_span, "trace-abc")
        assert handle.id == "trace-abc"
        assert handle._span is mock_span

    def test_update_delegates_to_span(self):
        from langfuse_helpers.tracing import TraceHandle
        mock_span = MagicMock()
        handle = TraceHandle(mock_span, "trace-abc")
        handle.update(output={"key": "val"})
        mock_span.update.assert_called_once_with(output={"key": "val"})

    def test_end_delegates_to_span(self):
        from langfuse_helpers.tracing import TraceHandle
        mock_span = MagicMock()
        handle = TraceHandle(mock_span, "trace-abc")
        handle.end()
        mock_span.end.assert_called_once()


class TestCreateTrace:

    def test_returns_trace_handle_with_correct_id(self):
        from langfuse_helpers import tracing
        from langfuse_helpers.tracing import TraceHandle

        mock_client = MagicMock()
        mock_span   = MagicMock()
        mock_client.create_trace_id.return_value   = "abcdef123"
        mock_client.start_observation.return_value = mock_span

        with patch.object(tracing, "langfuse_client", mock_client):
            result = tracing.create_trace("sess1", "user1", True, "hello world")

        assert isinstance(result, TraceHandle)
        assert result.id == "abcdef123"
        assert result._span is mock_span

    def test_sets_session_and_user_attributes(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_span   = MagicMock()
        mock_client.create_trace_id.return_value   = "abc"
        mock_client.start_observation.return_value = mock_span

        with patch.object(tracing, "langfuse_client", mock_client):
            tracing.create_trace("s1", "u1", False, "test message")

        calls = [str(c) for c in mock_span._otel_span.set_attribute.call_args_list]
        assert any("session.id" in c for c in calls)
        assert any("user.id" in c for c in calls)

    def test_passes_message_to_start_observation(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_span   = MagicMock()
        mock_client.create_trace_id.return_value   = "tid"
        mock_client.start_observation.return_value = mock_span

        with patch.object(tracing, "langfuse_client", mock_client):
            tracing.create_trace("sess", "usr", True, "find me a laptop")

        call_kwargs = mock_client.start_observation.call_args[1]
        assert call_kwargs["input"] == {"message": "find me a laptop"}


class TestCreateSpan:

    def test_returns_span_object(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_span   = MagicMock()
        mock_client.start_observation.return_value = mock_span

        with patch.object(tracing, "langfuse_client", mock_client):
            result = tracing.create_span("trace123", "my_span", input_data={"q": "v"})

        assert result is mock_span

    def test_with_parent_observation_id(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_span   = MagicMock()
        mock_client.start_observation.return_value = mock_span

        with patch.object(tracing, "langfuse_client", mock_client):
            tracing.create_span("trace123", "child", parent_observation_id="parent_456")

        # TraceContext is a TypedDict in LangFuse v4 — access as dict
        call_kwargs = mock_client.start_observation.call_args[1]
        trace_ctx   = call_kwargs["trace_context"]
        parent_id   = trace_ctx.parent_span_id if hasattr(trace_ctx, "parent_span_id") \
                      else trace_ctx["parent_span_id"]
        assert parent_id == "parent_456"

    def test_without_input_data(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_span   = MagicMock()
        mock_client.start_observation.return_value = mock_span

        with patch.object(tracing, "langfuse_client", mock_client):
            result = tracing.create_span("trace123", "span_no_input")

        assert result is mock_span


class TestEndSpan:

    def test_end_with_output_calls_update_then_end(self):
        from langfuse_helpers.tracing import end_span
        mock_span = MagicMock()
        end_span(mock_span, {"result": "ok"})
        mock_span.update.assert_called_once_with(output={"result": "ok"})
        mock_span.end.assert_called_once()

    def test_end_without_output_skips_update(self):
        from langfuse_helpers.tracing import end_span
        mock_span = MagicMock()
        end_span(mock_span)
        mock_span.update.assert_not_called()
        mock_span.end.assert_called_once()

    def test_end_swallows_exception(self):
        from langfuse_helpers.tracing import end_span
        mock_span = MagicMock()
        mock_span.end.side_effect = Exception("span error")
        end_span(mock_span)  # must not raise


class TestCreateGeneration:

    def test_returns_generation_and_calls_end(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_gen    = MagicMock()
        mock_client.start_observation.return_value = mock_gen

        with patch.object(tracing, "langfuse_client", mock_client):
            result = tracing.create_generation(
                trace_id   = "t123",
                name       = "llm_call",
                model      = "llama-70b",
                prompt     = "Hello",
                response   = "World",
                usage      = {"input": 10, "output": 5},
                agent_used = "product_agent"
            )

        assert result is mock_gen
        mock_gen.end.assert_called_once()

    def test_usage_details_passed_correctly(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_gen    = MagicMock()
        mock_client.start_observation.return_value = mock_gen

        with patch.object(tracing, "langfuse_client", mock_client):
            tracing.create_generation(
                trace_id  = "t1",
                name      = "gen",
                model     = "llama",
                prompt    = "p",
                response  = "r",
                usage     = {"input": 100, "output": 50}
            )

        call_kwargs = mock_client.start_observation.call_args[1]
        assert call_kwargs["usage_details"]["input"]  == 100
        assert call_kwargs["usage_details"]["output"] == 50

    def test_prompt_metadata_included_when_provided(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_gen    = MagicMock()
        mock_client.start_observation.return_value = mock_gen

        with patch.object(tracing, "langfuse_client", mock_client):
            tracing.create_generation(
                trace_id       = "t1",
                name           = "gen",
                model          = "llama",
                prompt         = "p",
                response       = "r",
                prompt_name    = "product_prompt",
                prompt_version = 2
            )

        call_kwargs = mock_client.start_observation.call_args[1]
        assert call_kwargs["metadata"]["prompt_name"] == "product_prompt"

    def test_no_metadata_when_no_prompt_name(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_gen    = MagicMock()
        mock_client.start_observation.return_value = mock_gen

        with patch.object(tracing, "langfuse_client", mock_client):
            tracing.create_generation(
                trace_id  = "t1",
                name      = "gen",
                model     = "llama",
                prompt    = "p",
                response  = "r"
            )

        call_kwargs = mock_client.start_observation.call_args[1]
        assert call_kwargs["metadata"] is None


class TestGetPrompt:

    def test_cache_hit_returns_cached_value(self):
        from langfuse_helpers import tracing
        tracing._prompt_cache["cache_test::latest"] = ("cached text", "latest")
        try:
            result = tracing.get_prompt("cache_test")
            assert result == ("cached text", "latest")
        finally:
            tracing._prompt_cache.pop("cache_test::latest", None)

    def test_fetches_latest_from_langfuse(self):
        from langfuse_helpers import tracing

        mock_client        = MagicMock()
        mock_prompt_obj    = MagicMock()
        mock_prompt_obj.prompt = "LangFuse prompt text"
        mock_client.get_prompt.return_value = mock_prompt_obj

        with patch.object(tracing, "langfuse_client", mock_client):
            result = tracing.get_prompt("fetch_test_prompt")

        assert result[0] == "LangFuse prompt text"
        tracing._prompt_cache.pop("fetch_test_prompt::latest", None)

    def test_fetches_with_label(self):
        from langfuse_helpers import tracing

        mock_client        = MagicMock()
        mock_prompt_obj    = MagicMock()
        mock_prompt_obj.prompt = "prod prompt"
        mock_client.get_prompt.return_value = mock_prompt_obj

        with patch.object(tracing, "langfuse_client", mock_client):
            result = tracing.get_prompt("label_prompt", label="production")

        assert result[0] == "prod prompt"
        tracing._prompt_cache.pop("label_prompt::production", None)

    def test_fetches_with_version(self):
        from langfuse_helpers import tracing

        mock_client        = MagicMock()
        mock_prompt_obj    = MagicMock()
        mock_prompt_obj.prompt = "v3 prompt"
        mock_client.get_prompt.return_value = mock_prompt_obj

        with patch.object(tracing, "langfuse_client", mock_client):
            result = tracing.get_prompt("ver_prompt", version=3)

        assert result[0] == "v3 prompt"
        tracing._prompt_cache.pop("ver_prompt::v3", None)

    def test_returns_fallback_on_langfuse_error(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_client.get_prompt.side_effect = Exception("LangFuse unavailable")

        with patch.object(tracing, "langfuse_client", mock_client):
            result = tracing.get_prompt("bad_prompt", fallback="default text")

        assert result == ("default text", "fallback")

    def test_returns_empty_string_fallback_when_none(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        mock_client.get_prompt.side_effect = Exception("error")

        with patch.object(tracing, "langfuse_client", mock_client):
            result = tracing.get_prompt("no_fallback_prompt")

        assert result == ("", "fallback")


class TestCompilePrompt:

    def test_replaces_single_variable(self):
        from langfuse_helpers.tracing import compile_prompt
        result = compile_prompt("Hello {{name}}!", name="Alice")
        assert result == "Hello Alice!"

    def test_replaces_multiple_variables(self):
        from langfuse_helpers.tracing import compile_prompt
        result = compile_prompt("{{a}} and {{b}}", a="X", b="Y")
        assert result == "X and Y"

    def test_none_value_replaced_with_empty_string(self):
        from langfuse_helpers.tracing import compile_prompt
        result = compile_prompt("Hello {{name}}", name=None)
        assert result == "Hello "

    def test_static_text_unchanged(self):
        from langfuse_helpers.tracing import compile_prompt
        result = compile_prompt("No variables here")
        assert result == "No variables here"


class TestFlush:

    def test_calls_client_flush(self):
        from langfuse_helpers import tracing

        mock_client = MagicMock()
        with patch.object(tracing, "langfuse_client", mock_client):
            tracing.flush()

        mock_client.flush.assert_called_once()
