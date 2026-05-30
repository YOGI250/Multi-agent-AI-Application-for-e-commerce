# tests/test_utils/test_memory.py

import pytest
from utils.memory import format_context, format_recent_messages, merge_context


class TestFormatContext:

    def test_empty_dict_returns_new_conversation(self):
        assert format_context({}) == "[new conversation]"

    def test_none_returns_new_conversation(self):
        assert format_context(None) == "[new conversation]"

    def test_topic_only(self):
        result = format_context({"topic": "laptops"})
        assert "topic=laptops" in result

    def test_order_id_included(self):
        result = format_context({"order_id": "ORD-123"})
        assert "order_id=ORD-123" in result

    def test_issue_type_included(self):
        result = format_context({"issue_type": "damaged"})
        assert "issue=damaged" in result

    def test_ticket_id_included(self):
        result = format_context({"ticket_id": "TKT00001"})
        assert "ticket=TKT00001" in result

    def test_orders_listed_with_count(self):
        result = format_context({"orders_listed": True, "order_count": 3})
        assert "orders_listed=true(3 orders)" in result

    def test_orders_listed_without_count(self):
        result = format_context({"orders_listed": True})
        assert "orders_listed=true" in result

    def test_products_shown(self):
        result = format_context({"products_shown": ["p1", "p2", "p3"]})
        assert "products_shown=3" in result

    def test_multiple_fields_joined_with_pipe(self):
        result = format_context({"topic": "shoes", "order_id": "ORD-999"})
        assert "|" in result
        assert "topic=shoes" in result
        assert "order_id=ORD-999" in result

    def test_no_matching_fields_returns_new_conversation(self):
        result = format_context({"unrelated_key": "value"})
        assert result == "[new conversation]"

    def test_result_wrapped_in_brackets(self):
        result = format_context({"topic": "phones"})
        assert result.startswith("[") and result.endswith("]")


class TestFormatRecentMessages:

    def test_empty_history_returns_no_previous(self):
        assert format_recent_messages([]) == "No previous messages"

    def test_single_message_returned(self):
        history = [{"role": "user", "content": "hello"}]
        result = format_recent_messages(history)
        assert "USER: hello" in result

    def test_role_uppercased(self):
        history = [{"role": "assistant", "content": "hi there"}]
        result = format_recent_messages(history)
        assert "ASSISTANT: hi there" in result

    def test_long_content_truncated(self):
        long_msg = "a" * 200
        history = [{"role": "user", "content": long_msg}]
        result = format_recent_messages(history, max_chars=150)
        assert result.endswith("...")
        assert len(result) < 200

    def test_short_content_not_truncated(self):
        history = [{"role": "user", "content": "short"}]
        result = format_recent_messages(history)
        assert "..." not in result

    def test_only_last_n_turns_returned(self):
        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "resp3"},
        ]
        result = format_recent_messages(history, n=1)
        assert "msg1" not in result
        assert "msg3" in result

    def test_multiple_messages_joined_by_newline(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = format_recent_messages(history)
        assert "\n" in result


class TestMergeContext:

    def test_new_facts_added_to_empty(self):
        result = merge_context({}, {"topic": "shoes"})
        assert result["topic"] == "shoes"

    def test_new_facts_overwrite_old(self):
        result = merge_context({"topic": "shoes"}, {"topic": "phones"})
        assert result["topic"] == "phones"

    def test_none_values_do_not_overwrite(self):
        result = merge_context({"order_id": "ORD-1"}, {"order_id": None})
        assert result["order_id"] == "ORD-1"

    def test_old_keys_preserved_when_not_in_new(self):
        result = merge_context({"topic": "shoes", "order_id": "ORD-1"}, {"topic": "phones"})
        assert result["order_id"] == "ORD-1"

    def test_returns_new_dict_not_mutating_old(self):
        old = {"topic": "shoes"}
        merge_context(old, {"topic": "phones"})
        assert old["topic"] == "shoes"
