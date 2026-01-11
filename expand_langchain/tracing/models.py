"""Data models for trace events."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TraceEventType(str, Enum):
    """Types of trace events."""

    # LLM events
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    LLM_ERROR = "llm_error"
    LLM_NEW_TOKEN = "llm_new_token"

    # Chat model events
    CHAT_MODEL_START = "chat_model_start"
    CHAT_MODEL_END = "chat_model_end"

    # Chain events
    CHAIN_START = "chain_start"
    CHAIN_END = "chain_end"
    CHAIN_ERROR = "chain_error"

    # Tool events
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"

    # Agent events
    AGENT_ACTION = "agent_action"
    AGENT_FINISH = "agent_finish"

    # Graph events (LangGraph)
    GRAPH_NODE_START = "graph_node_start"
    GRAPH_NODE_END = "graph_node_end"
    GRAPH_STATE_UPDATE = "graph_state_update"

    # Retriever events
    RETRIEVER_START = "retriever_start"
    RETRIEVER_END = "retriever_end"

    # Custom events
    CUSTOM = "custom"


@dataclass
class TraceEvent:
    """A single trace event.

    Attributes:
        event_type: Type of the event
        timestamp: When the event occurred
        run_id: Unique identifier for the run
        parent_run_id: Parent run ID for nested events
        name: Name of the component (e.g., model name, tool name)
        inputs: Input data for the event
        outputs: Output data for the event (None for start events)
        metadata: Additional metadata
        error: Error information if the event failed
        duration_ms: Duration in milliseconds (for end events)
        task_id: Identifier for the task being processed
        tags: List of tags associated with the event
    """

    event_type: TraceEventType
    timestamp: datetime
    run_id: str
    name: str
    parent_run_id: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    task_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "task_id": self.task_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceEvent":
        """Create from dictionary."""
        return cls(
            event_type=TraceEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            run_id=data["run_id"],
            parent_run_id=data.get("parent_run_id"),
            name=data["name"],
            inputs=data.get("inputs"),
            outputs=data.get("outputs"),
            metadata=data.get("metadata"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms"),
            task_id=data.get("task_id"),
            tags=data.get("tags", []),
        )

    def to_json(self, pretty: bool = False) -> str:
        """Convert to JSON string."""
        if pretty:
            return json.dumps(
                self.to_dict(), indent=2, ensure_ascii=False, default=str
            )
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


@dataclass
class TraceSession:
    """A collection of trace events for a single task.

    Attributes:
        task_id: Identifier for the task
        run_name: Name of the overall run
        start_time: When the session started
        end_time: When the session ended (None if still running)
        events: List of trace events
        metadata: Session-level metadata
    """

    task_id: str
    run_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    events: List[TraceEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: TraceEvent):
        """Add an event to the session."""
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "run_name": self.run_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceSession":
        """Create from dictionary."""
        session = cls(
            task_id=data["task_id"],
            run_name=data["run_name"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(
                datetime.fromisoformat(data["end_time"])
                if data.get("end_time")
                else None
            ),
            metadata=data.get("metadata", {}),
        )
        session.events = [
            TraceEvent.from_dict(e) for e in data.get("events", [])
        ]
        return session

    @property
    def duration_ms(self) -> Optional[float]:
        """Get session duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def event_count(self) -> int:
        """Get the number of events."""
        return len(self.events)

    @property
    def error_count(self) -> int:
        """Get the number of error events."""
        return sum(1 for e in self.events if e.error is not None)
