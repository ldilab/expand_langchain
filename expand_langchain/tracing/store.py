"""Storage backend for traces."""

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import TracingConfig
from .models import TraceEvent, TraceSession
from .yaml_utils import append_yaml_list_item, dump_yaml

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
        # Ensure directories exist
        config.ensure_directories()

    def __del__(self):
        """Cleanup resources."""
        self.close()

    def close(self):
        """Close open file handles."""
        return

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

            if self.config.write_yaml_trace:
                trace_path = _trace_yaml_path_for_task(self.config.traces_dir, task_id)
                payload = _build_trace_payload(event)
                append_yaml_list_item(trace_path, payload)
                if _should_write_parse_event(event):
                    for parse_payload in _build_parse_events(event):
                        append_yaml_list_item(trace_path, parse_payload)

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

            if self.config.write_json:
                json_path = self.config.traces_dir / f"{task_id}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        session.to_dict(),
                        f,
                        indent=2 if self.config.pretty_print else None,
                        ensure_ascii=False,
                        default=str,
                    )

            if self.config.write_full_run_histories:
                try:
                    history_path = (
                        self.config.traces_dir / f"{task_id}_full_run_histories.yaml"
                    )
                    history_payload = _build_full_run_history_payload(session)
                    if history_payload:
                        history_path.write_text(
                            dump_yaml(history_payload), encoding="utf-8"
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to write full run history for %s: %s", task_id, e
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


def _build_trace_payload(event: TraceEvent) -> Dict[str, Any]:
    payload = {
        "ts": event.timestamp.isoformat(),
        "event": event.event_type.value,
        "name": event.name,
        "run_id": event.run_id,
        "parent_run_id": event.parent_run_id,
        "task_id": event.task_id,
        "duration_ms": event.duration_ms,
        "inputs": event.inputs,
        "outputs": event.outputs,
        "metadata": event.metadata,
        "error": event.error,
        "tags": event.tags,
    }
    return {key: value for key, value in payload.items() if value is not None}


def _trace_yaml_path_for_task(traces_dir: Path, task_id: str) -> Path:
    safe_task_id = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("_")
    if not safe_task_id:
        safe_task_id = "unknown"
    return traces_dir / f"{safe_task_id}_trace.yaml"


def _should_write_parse_event(event: TraceEvent) -> bool:
    if not isinstance(event.outputs, dict):
        return False
    return "parsing_error" in event.outputs or "parsed" in event.outputs


def _build_parse_events(event: TraceEvent) -> List[Dict[str, Any]]:
    if not isinstance(event.outputs, dict):
        return []
    base = {
        "ts": event.timestamp.isoformat(),
        "name": event.name,
        "run_id": event.run_id,
        "parent_run_id": event.parent_run_id,
        "task_id": event.task_id,
    }
    items: List[Dict[str, Any]] = []
    if event.outputs.get("parsed") is not None:
        items.append(
            {
                **base,
                "event": "llm_output_parsed",
                "parsed": event.outputs.get("parsed"),
            }
        )
    if event.outputs.get("parsing_error") is not None:
        items.append(
            {
                **base,
                "event": "llm_output_parsing_error",
                "parsing_error": str(event.outputs.get("parsing_error")),
            }
        )
    return items


def _serialize_chat_history(chat_history: Any) -> List[Dict[str, Any]]:
    if not isinstance(chat_history, list):
        return []
    serialized: List[Dict[str, Any]] = []
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        msg_type = msg.get("type", "unknown")
        msg_data = msg.get("data") if isinstance(msg.get("data"), dict) else None
        if msg_data is not None:
            serialized.append(
                {
                    "type": msg_type,
                    "data": {
                        "content": msg_data.get("content", ""),
                        "additional_kwargs": msg_data.get("additional_kwargs", {}),
                        "response_metadata": msg_data.get("response_metadata", {}),
                        "name": msg_data.get("name"),
                        "id": msg_data.get("id"),
                    },
                }
            )
            continue
        serialized.append(
            {
                "type": msg_type,
                "data": {
                    "content": msg.get("content", ""),
                    "additional_kwargs": msg.get("additional_kwargs", {}),
                    "response_metadata": msg.get("response_metadata", {}),
                    "name": msg.get("name"),
                    "id": msg.get("id"),
                },
            }
        )
    return serialized


def _normalize_trace_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    if "chat_history" in normalized:
        normalized["chat_history"] = _serialize_chat_history(
            normalized.get("chat_history")
        )
    return normalized


def _update_agent_record(
    records: Dict[Tuple[int, str], Dict[str, Any]],
    payload: Dict[str, Any],
    order: List[Tuple[int, str]],
) -> None:
    agent_depth = payload.get("agent_depth")
    subtask = payload.get("subtask_description") or ""
    if agent_depth is None:
        has_terminal_state = any(
            key in payload
            for key in [
                "chat_history",
                "sql",
                "answer",
                "termination_reason",
            ]
        )
        if not has_terminal_state:
            return
        agent_depth_int = 0
        subtask = ""
    else:
        try:
            agent_depth_int = int(agent_depth)
        except (TypeError, ValueError):
            return
    key = (agent_depth_int, str(subtask))
    record = records.get(
        key,
        {
            "agent_depth": agent_depth_int,
            "subtask_description": subtask,
            "children": [],
            "node_trace": [],
        },
    )

    if "instance_id" in payload:
        record["instance_id"] = payload.get("instance_id")
    if "db_id" in payload:
        record["db_id"] = payload.get("db_id")
    if "global_memory" in payload:
        record["global_memory"] = payload.get("global_memory")
    if "local_memory" in payload:
        record["local_memory"] = payload.get("local_memory")
    if "termination_reason" in payload:
        record["termination_reason"] = payload.get("termination_reason")
    if "chat_history" in payload:
        record["chat_history"] = _serialize_chat_history(payload.get("chat_history"))

    records[key] = record
    if key not in order:
        order.append(key)


def _agent_key_from_payload(payload: Dict[str, Any]) -> Tuple[int, str] | None:
    agent_depth = payload.get("agent_depth")
    if agent_depth is None:
        return None
    try:
        agent_depth_int = int(agent_depth)
    except (TypeError, ValueError):
        return None
    subtask = payload.get("subtask_description") or ""
    return (agent_depth_int, str(subtask))


def _nest_agent_records(
    records: Dict[Tuple[int, str], Dict[str, Any]],
    order: List[Tuple[int, str]],
) -> Dict[str, Any] | None:
    if not records:
        return None

    root_key = None
    for key in order:
        if key[0] == 0:
            root_key = key
            break

    if root_key is None:
        return None

    last_by_depth: Dict[int, Tuple[int, str]] = {}
    for key in order:
        depth = key[0]
        last_by_depth[depth] = key
        if depth == 0:
            continue
        parent_key = last_by_depth.get(depth - 1)
        if parent_key is None:
            continue
        parent = records[parent_key]
        child = records[key]
        if child not in parent.get("children", []):
            parent.setdefault("children", []).append(child)

    return records[root_key]


def _nest_node_trace(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not nodes:
        return []

    nodes_by_run: Dict[str, Dict[str, Any]] = {}
    ordered = []
    for node in nodes:
        node.setdefault("children", [])
        ordered.append(node)
        run_id = node.get("run_id")
        if isinstance(run_id, str) and run_id:
            nodes_by_run[run_id] = node

    roots: List[Dict[str, Any]] = []
    last_tool_exec: Dict[str, Any] | None = None

    for node in ordered:
        parent_run_id = node.get("parent_run_id")
        parent = None
        if isinstance(parent_run_id, str) and parent_run_id:
            parent = nodes_by_run.get(parent_run_id)
        if parent is None and node.get("node_name") == "extract_tool_exec":
            parent = last_tool_exec
        if parent is None:
            roots.append(node)
        else:
            parent.setdefault("children", []).append(node)
        if node.get("node_name") == "tool_exec":
            last_tool_exec = node

    return roots


def _build_full_run_history_payload(session: TraceSession) -> Dict[str, Any] | None:
    records: Dict[Tuple[int, str], Dict[str, Any]] = {}
    order: List[Tuple[int, str]] = []
    outputs_by_run_id: Dict[str, Dict[str, Any]] = {}

    for event in session.events:
        if event.event_type.value == "chain_end" and isinstance(event.outputs, dict):
            outputs_by_run_id[event.run_id] = _normalize_trace_payload(event.outputs)

    for event in session.events:
        if (
            event.event_type.value == "graph_node_start"
            and event.name == "main_recursive_agent"
        ):
            if isinstance(event.inputs, dict):
                _update_agent_record(records, event.inputs, order)
        if isinstance(event.outputs, dict):
            _update_agent_record(records, event.outputs, order)

        if event.event_type.value == "graph_node_start" and isinstance(
            event.inputs, dict
        ):
            agent_key = _agent_key_from_payload(event.inputs)
            if agent_key and agent_key in records:
                node_trace = {
                    "node_name": event.name,
                    "run_id": event.run_id,
                    "parent_run_id": event.parent_run_id,
                    "inputs": _normalize_trace_payload(event.inputs),
                    "outputs": outputs_by_run_id.get(event.run_id),
                }
                records[agent_key].setdefault("node_trace", []).append(node_trace)

    if not records:
        return None

    root_record = _nest_agent_records(records, order)
    if root_record is None:
        return None

    def _compact_agent(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "instance_id": record.get("instance_id"),
            "db_id": record.get("db_id"),
            "agent_depth": record.get("agent_depth"),
            "subtask_description": record.get("subtask_description"),
            "termination_reason": record.get("termination_reason"),
            "chat_history": record.get("chat_history", []),
            "global_memory": record.get("global_memory"),
            "local_memory": record.get("local_memory"),
            "node_trace": _nest_node_trace(record.get("node_trace", [])),
            "children": [_compact_agent(child) for child in record.get("children", [])],
        }

    return {
        "root": _compact_agent(root_record),
    }
