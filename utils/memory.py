# utils/memory.py

def format_context(ctx: dict) -> str:
    """Compact one-line summary of session context for LLM prompts."""
    if not ctx:
        return "[new conversation]"
    parts = []
    if ctx.get("topic"):
        parts.append(f"topic={ctx['topic']}")
    if ctx.get("order_id"):
        parts.append(f"order_id={ctx['order_id']}")
    if ctx.get("issue_type"):
        parts.append(f"issue={ctx['issue_type']}")
    if ctx.get("ticket_id"):
        parts.append(f"ticket={ctx['ticket_id']}")
    if ctx.get("orders_listed"):
        count = ctx.get("order_count")
        parts.append(f"orders_listed=true({count} orders)" if count else "orders_listed=true")
    if ctx.get("products_shown"):
        parts.append(f"products_shown={len(ctx['products_shown'])}")
    return "[" + " | ".join(parts) + "]" if parts else "[new conversation]"


def format_recent_messages(history: list, n: int = 2, max_chars: int = 150) -> str:
    """Last n turns of history, each message truncated to max_chars."""
    if not history:
        return "No previous messages"
    recent = history[-(n * 2):]
    lines = []
    for m in recent:
        content = m["content"]
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        lines.append(f"{m['role'].upper()}: {content}")
    return "\n".join(lines)


def merge_context(old: dict, new_facts: dict) -> dict:
    """Merge new_facts into old context; only overwrite with non-None values."""
    merged = {**old}
    for k, v in new_facts.items():
        if v is not None:
            merged[k] = v
    return merged
