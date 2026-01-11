"""Storage backend for traces."""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .config import TracingConfig
from .models import TraceEvent, TraceSession

logger = logging.getLogger(__name__)


class TraceStore:
    """Storage backend for trace events.

    Supports both JSON (full session) and JSONL (append-only events) formats.
    Thread-safe for concurrent writes.
    """

    def __init__(self, config: TracingConfig):
        """Initialize the trace store.

        Args:
            config: Tracing configuration
        """
        self.config = config
        self._lock = threading.Lock()
        self._sessions: Dict[str, TraceSession] = {}
        self._realtime_file = None

        # Ensure directories exist
        config.ensure_directories()

        # Open realtime log file if enabled
        if config.enable_realtime_log:
            self._realtime_file = open(config.realtime_log_path, "a", encoding="utf-8")
            logger.info(f"Realtime log enabled: {config.realtime_log_path}")

    def __del__(self):
        """Cleanup resources."""
        self.close()

    def close(self):
        """Close open file handles."""
        if self._realtime_file:
            self._realtime_file.close()
            self._realtime_file = None

    def get_or_create_session(self, task_id: str) -> TraceSession:
        """Get or create a trace session for a task.

        Args:
            task_id: The task identifier

        Returns:
            The trace session
        """
        with self._lock:
            if task_id not in self._sessions:
                self._sessions[task_id] = TraceSession(
                    task_id=task_id,
                    run_name=self.config.run_name,
                    start_time=datetime.now(),
                )
            return self._sessions[task_id]

    def add_event(self, event: TraceEvent):
        """Add a trace event.

        Args:
            event: The trace event to add
        """
        task_id = event.task_id or "default"
        session = self.get_or_create_session(task_id)

        with self._lock:
            session.add_event(event)

            # Write to JSONL file (append mode)
            jsonl_path = self.config.traces_dir / f"{task_id}.jsonl"
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(event.to_json(pretty=False) + "\n")

            # Write to realtime log if enabled
            if self._realtime_file and self.config.enable_realtime_log:
                self._write_realtime_log(event)

    def _write_realtime_log(self, event: TraceEvent):
        """Write event to realtime log in human-readable format.

        Args:
            event: The trace event
        """
        if not self._realtime_file:
            return

        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        task_id = event.task_id or "?"

        # Format based on event type
        if event.event_type.value.endswith("_start"):
            symbol = ">>>"
        elif event.event_type.value.endswith("_end"):
            symbol = "<<<"
        elif "error" in event.event_type.value.lower():
            symbol = "!!!"
        else:
            symbol = "---"

        # Build log line
        line_parts = [
            f"[{timestamp}]",
            f"[{task_id}]",
            symbol,
            f"{event.event_type.value}",
            f"({event.name})",
        ]

        # Add duration for end events
        if event.duration_ms is not None:
            line_parts.append(f"[{event.duration_ms:.0f}ms]")

        # Add error info
        if event.error:
            line_parts.append(f"ERROR: {event.error[:100]}")

        line = " ".join(line_parts)

        self._realtime_file.write(line + "\n")
        self._realtime_file.flush()  # Ensure immediate write for tail -f

    def finalize_session(self, task_id: str):
        """Finalize a session and write the complete JSON file.

        Args:
            task_id: The task identifier
        """
        with self._lock:
            if task_id not in self._sessions:
                return

            session = self._sessions[task_id]
            session.end_time = datetime.now()

            # Write complete session JSON
            json_path = self.config.traces_dir / f"{task_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    session.to_dict(),
                    f,
                    indent=2 if self.config.pretty_print else None,
                    ensure_ascii=False,
                    default=str,
                )

            logger.info(f"Finalized trace session: {task_id}")

    def get_session(self, task_id: str) -> Optional[TraceSession]:
        """Get a trace session.

        Args:
            task_id: The task identifier

        Returns:
            The trace session or None if not found
        """
        return self._sessions.get(task_id)

    def list_sessions(self) -> List[str]:
        """List all session task IDs.

        Returns:
            List of task IDs
        """
        return list(self._sessions.keys())

    @staticmethod
    def load_session(path: Path) -> TraceSession:
        """Load a trace session from a JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            The loaded trace session
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TraceSession.from_dict(data)

    @staticmethod
    def load_events(path: Path) -> Iterator[TraceEvent]:
        """Load trace events from a JSONL file.

        Args:
            path: Path to the JSONL file

        Yields:
            Trace events
        """
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    yield TraceEvent.from_dict(data)

    @staticmethod
    def load_traces_dir(traces_dir: Path) -> Dict[str, TraceSession]:
        """Load all sessions from a traces directory.

        Args:
            traces_dir: Path to the traces directory

        Returns:
            Dictionary mapping task_id to TraceSession
        """
        sessions = {}
        for json_file in traces_dir.glob("*.json"):
            if json_file.name == "realtime.log":
                continue
            try:
                session = TraceStore.load_session(json_file)
                sessions[session.task_id] = session
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        return sessions
