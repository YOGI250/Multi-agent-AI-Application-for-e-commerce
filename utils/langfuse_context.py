from contextvars import ContextVar

_trace_id: ContextVar = ContextVar("langfuse_trace_id", default=None)
_span_id: ContextVar = ContextVar("langfuse_span_id", default=None)


def set_trace_context(trace_id: str, span_id: str = None):
    _trace_id.set(trace_id)
    _span_id.set(span_id)


def get_trace_context():
    return _trace_id.get(), _span_id.get()
