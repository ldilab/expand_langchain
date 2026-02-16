"""Configuration for local tracing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Set

from .models import TraceEventType


def _default_event_types() -> Set[str]:
    return {
        TraceEventType.LLM_END.value,
        TraceEventType.TOOL_START.value,
        TraceEventType.TOOL_END.value,
        TraceEventType.TOOL_ERROR.value,
    }


def _normalize_event_types(event_types: Iterable[object]) -> Set[str]:
    normalized: Set[str] = set()
    for event_type in event_types:
        if isinstance(event_type, TraceEventType):
            normalized.add(event_type.value)
        else:
            normalized.add(str(event_type))
    return normalized


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
        write_full_run_histories: Whether to write a compact run history YAML
        write_json: Whether to write session JSON files
        write_jsonl: Whether to write event JSONL files
        write_yaml_trace: Whether to write YAML trace events
        generate_ai_summary: Whether to generate hierarchical debug trace (task_id_debug.yaml)
        summary_max_content_length: (Deprecated, not used)
        event_types: Allowed event types to record (default excludes chain_start)
    """

    run_name: str
    results_dir: str = "results"
    enable_realtime_log: bool = False
    log_llm_io: bool = True
    log_tool_io: bool = True
    log_graph_state: bool = True
    max_content_length: int = 0
    pretty_print: bool = True
    write_full_run_histories: bool = True
    write_json: bool = False
    write_jsonl: bool = False
    write_yaml_trace: bool = True
    generate_ai_summary: bool = True
    summary_max_content_length: int = 5000
    event_types: Optional[Set[str]] = field(default_factory=_default_event_types)

    # Internal fields
    _traces_dir: Optional[Path] = field(default=None, repr=False)
    _realtime_log_path: Optional[Path] = field(default=None, repr=False)
    _trace_yaml_path: Optional[Path] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize directory paths."""
        if self.event_types is None:
            self.event_types = _default_event_types()
        else:
            self.event_types = _normalize_event_types(self.event_types)
        base_dir = Path(self.results_dir) / self.run_name
        self._traces_dir = base_dir / "traces"
        self._realtime_log_path = base_dir / "traces" / "realtime.log"
        self._trace_yaml_path = base_dir / "traces" / "trace.yaml"
        # Force-disable realtime and JSON/JSONL outputs for YAML-only tracing.
        self.enable_realtime_log = False
        self.write_json = False
        self.write_jsonl = False

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

    @property
    def trace_yaml_path(self) -> Path:
        """Get the YAML trace file path."""
        if self._trace_yaml_path is None:
            raise ValueError("TracingConfig not properly initialized")
        return self._trace_yaml_path

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.traces_dir.mkdir(parents=True, exist_ok=True)
