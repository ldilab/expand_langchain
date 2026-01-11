"""Configuration for local tracing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TracingConfig:
    """Configuration for local tracing.

    Attributes:
        run_name: Name of the run, used for directory naming
        results_dir: Base directory for results (default: "results")
        enable_realtime_log: Whether to write to a realtime log file for tail -f
        log_llm_io: Whether to log LLM inputs and outputs
        log_tool_io: Whether to log tool inputs and outputs
        log_graph_state: Whether to log LangGraph state changes
        max_content_length: Maximum length for logged content (truncated if longer)
        pretty_print: Whether to pretty-print JSON output
    """

    run_name: str
    results_dir: str = "results"
    enable_realtime_log: bool = True
    log_llm_io: bool = True
    log_tool_io: bool = True
    log_graph_state: bool = True
    max_content_length: int = 10000
    pretty_print: bool = True

    # Internal fields
    _traces_dir: Optional[Path] = field(default=None, repr=False)
    _realtime_log_path: Optional[Path] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize directory paths."""
        base_dir = Path(self.results_dir) / self.run_name
        self._traces_dir = base_dir / "traces"
        self._realtime_log_path = base_dir / "traces" / "realtime.log"

    @property
    def traces_dir(self) -> Path:
        """Get the traces directory path."""
        if self._traces_dir is None:
            raise ValueError("TracingConfig not properly initialized")
        return self._traces_dir

    @property
    def realtime_log_path(self) -> Path:
        """Get the realtime log file path."""
        if self._realtime_log_path is None:
            raise ValueError("TracingConfig not properly initialized")
        return self._realtime_log_path

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.traces_dir.mkdir(parents=True, exist_ok=True)
