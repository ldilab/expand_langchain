"""Allow running viewer as a module.

Usage:
    uv run python -m expand_langchain.tracing results/run-name/traces/
    uv run python -m expand_langchain.tracing.viewer results/run-name/traces/
"""

from .viewer import main

if __name__ == "__main__":
    main()
