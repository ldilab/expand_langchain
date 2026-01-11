"""Local tracing module for LangGraph/LangChain pipelines.

This module provides:
1. LocalTraceCallback - A LangChain callback handler for capturing traces
2. TraceStore - Storage backend for traces (JSON/JSONL)
3. TracingConfig - Configuration for tracing behavior
4. viewer - CLI tool for viewing traces

Usage:
    from expand_langchain.tracing import LocalTraceCallback, TracingConfig

    config = TracingConfig(
        run_name="my-run",
        results_dir="results",
        enable_realtime_log=True,
    )
    callback = LocalTraceCallback(config)

    # Use with Generator
    generator = Generator(...)
    generator.run(..., callbacks=[callback])
"""

from .callback import LocalTraceCallback
from .config import TracingConfig
from .store import TraceStore

__all__ = [
    "LocalTraceCallback",
    "TracingConfig",
    "TraceStore",
]
