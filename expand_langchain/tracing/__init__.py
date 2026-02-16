"""Local tracing module for LangGraph/LangChain pipelines.

This module provides:
1. LocalTraceCallback - A LangChain callback handler for capturing traces
2. TraceStore - Storage backend for traces (JSON/JSONL/YAML)
3. TracingConfig - Configuration for tracing behavior
4. Hierarchical trace builder - AI-friendly debug traces
5. viewer - CLI tool for viewing traces

Usage:
    from expand_langchain.tracing import LocalTraceCallback, TracingConfig

    config = TracingConfig(
        run_name="my-run",
        results_dir="results",
        generate_ai_summary=True,  # Generate hierarchical debug trace
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
