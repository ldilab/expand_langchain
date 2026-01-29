"""LangChain callback handler for local tracing."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

from .config import TracingConfig
from .models import TraceEvent, TraceEventType
from .store import TraceStore

logger = logging.getLogger(__name__)


def _truncate(value: Any, max_length: int) -> Any:
    """Truncate a value if it's too long.

    Args:
        value: Value to truncate
        max_length: Maximum length

    Returns:
        Truncated value
    """
    if isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length] + f"... [truncated, total {len(value)} chars]"
        return value
    elif isinstance(value, dict):
        return {k: _truncate(v, max_length) for k, v in value.items()}
    elif isinstance(value, list):
        return [_truncate(v, max_length) for v in value]
    return value


def _serialize_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    """Serialize messages to dicts.

    Args:
        messages: Messages to serialize

    Returns:
        List of serialized messages
    """
    result = []
    for msg in messages:
        result.append(
            {
                "type": msg.type,
                "content": (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                ),
                "additional_kwargs": getattr(msg, "additional_kwargs", {}),
            }
        )
    return result


def _serialize_value(value: Any) -> Any:
    """Safely serialize any value, including Pydantic models.

    Args:
        value: Value to serialize

    Returns:
        Serialized value safe for JSON
    """
    if value is None:
        return None

    # Check if it's a Pydantic model
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception as e:
            logger.debug(f"Failed to use model_dump: {e}")
            # Fallback to dict conversion
            try:
                return dict(value)
            except Exception:
                return str(value)

    # Check for old Pydantic v1 style
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception as e:
            logger.debug(f"Failed to use dict(): {e}")
            return str(value)

    # Handle dict
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}

    # Handle list/tuple
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]

    # Primitives
    if isinstance(value, (str, int, float, bool)):
        return value

    # Try to convert to dict if it has __dict__
    if hasattr(value, "__dict__"):
        try:
            return {
                k: _serialize_value(v)
                for k, v in value.__dict__.items()
                if not k.startswith("_")
            }
        except Exception:
            return str(value)

    # Last resort
    return str(value)


def _name_from_serialized(serialized: Any, fallback: str = "unknown") -> str:
    """Best-effort component name extraction.

    LangChain sometimes passes `serialized=None` or different shapes depending on
    the Runnable implementation.
    """
    if not isinstance(serialized, dict):
        return fallback

    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name

    ident = serialized.get("id")
    if isinstance(ident, (list, tuple)) and ident:
        last = ident[-1]
        return str(last)

    return fallback


class LocalTraceCallback(BaseCallbackHandler):
    """Callback handler for local tracing.

    This handler captures LLM calls, tool executions, chain runs, and agent actions,
    storing them in a local trace store.
    """

    def __init__(
        self,
        config: TracingConfig,
        store: Optional[TraceStore] = None,
    ):
        """Initialize the callback handler.

        Args:
            config: Tracing configuration
            store: Optional trace store (created if not provided)
        """
        self.config = config
        self.store = store or TraceStore(config)
        self._run_start_times: Dict[str, float] = {}
        self._current_task_id: Optional[str] = None
        self._llm_call_counter: Dict[str, int] = {}  # Track LLM call count per task
        self._pending_llm_calls: Dict[str, Dict[str, Any]] = (
            {}
        )  # Store start events by run_id

    def set_task_id(self, task_id: str):
        """Set the current task ID for tracing.

        Args:
            task_id: The task identifier
        """
        self._current_task_id = task_id
        # Initialize LLM call counter for this task
        if task_id not in self._llm_call_counter:
            self._llm_call_counter[task_id] = 0
        logger.debug(f"[LocalTraceCallback] set_task_id: {task_id}")

    def _write_llm_io_text(
        self,
        task_id: str,
        call_number: int,
        messages: List[Dict],
        response_text: str,
        timestamp: str,
    ):
        """Write LLM I/O to human-readable text file.

        Args:
            task_id: Task identifier
            call_number: Sequential call number
            messages: Input messages
            response_text: LLM response
            timestamp: Timestamp string
        """
        if not self.config.log_llm_io:
            return

        try:
            text_file = self.config.traces_dir / f"{task_id}_llm_io.txt"
            with open(text_file, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"LLM Call #{call_number}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write("=" * 80 + "\n\n")

                # Write input messages
                f.write("INPUT MESSAGES:\n")
                f.write("-" * 80 + "\n")
                for i, msg in enumerate(messages, 1):
                    msg_type = msg.get("type", "unknown")
                    msg_content = msg.get("content", "")
                    f.write(f"\n[Message {i} - {msg_type.upper()}]\n")
                    f.write(msg_content + "\n")

                # Write output
                f.write("\n" + "-" * 80 + "\n")
                f.write("OUTPUT RESPONSE:\n")
                f.write("-" * 80 + "\n")
                f.write(response_text + "\n\n")

        except Exception as e:
            logger.debug(f"Failed to write LLM I/O text file: {e}")

    def _get_task_id(
        self,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the current task ID.

        Args:
            tags: Optional tags that may contain task ID
            metadata: Optional metadata that may contain task ID

        Returns:
            The task ID
        """
        # Priority 1: Explicitly set task ID (most reliable)
        if self._current_task_id:
            return self._current_task_id

        # Priority 2: Metadata 'id' field (from config)
        if metadata and isinstance(metadata, dict):
            if "id" in metadata:
                return str(metadata["id"])

        # Priority 3: Tags (but skip internal tags like seq:, graph:, map:)
        if tags:
            for tag in tags:
                if tag and not any(
                    tag.startswith(prefix) for prefix in ["seq:", "graph:", "map:"]
                ):
                    return str(tag)

        # Fallback to unknown
        return "unknown"

    def _create_event(
        self,
        event_type: TraceEventType,
        run_id: UUID,
        name: str,
        parent_run_id: Optional[UUID] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> TraceEvent:
        """Create a trace event.

        Args:
            event_type: Type of event
            run_id: Run identifier
            name: Name of the component
            parent_run_id: Parent run ID
            inputs: Input data
            outputs: Output data
            metadata: Additional metadata
            error: Error message
            tags: Tags

        Returns:
            The trace event
        """
        # Calculate duration for end events
        duration_ms = None
        run_id_str = str(run_id)
        if event_type.value.endswith("_end") or error:
            if run_id_str in self._run_start_times:
                start_time = self._run_start_times.pop(run_id_str)
                duration_ms = (time.time() - start_time) * 1000

        # Truncate large inputs/outputs
        if inputs and self.config.max_content_length:
            inputs = _truncate(inputs, self.config.max_content_length)
        if outputs and self.config.max_content_length:
            outputs = _truncate(outputs, self.config.max_content_length)

        return TraceEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            run_id=run_id_str,
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            error=error,
            duration_ms=duration_ms,
            task_id=self._get_task_id(tags, metadata),
            tags=tags or [],
        )

    def _record_start(self, run_id: UUID):
        """Record the start time for a run.

        Args:
            run_id: Run identifier
        """
        self._run_start_times[str(run_id)] = time.time()

    # ========== LLM Callbacks ==========

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM start."""
        if not self.config.log_llm_io:
            return

        logger.debug(
            f"[LocalTraceCallback] on_llm_start: task_id={self._current_task_id}, run_id={run_id}"
        )

        self._record_start(run_id)
        name = _name_from_serialized(serialized, fallback="llm")

        event = self._create_event(
            event_type=TraceEventType.LLM_START,
            run_id=run_id,
            name=name,
            parent_run_id=parent_run_id,
            inputs={"prompts": prompts},
            metadata=metadata,
            tags=tags,
        )
        self.store.add_event(event)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM end."""
        if not self.config.log_llm_io:
            return

        # Extract generations
        outputs = {
            "generations": [
                [{"text": gen.text, "info": gen.generation_info} for gen in gens]
                for gens in response.generations
            ],
        }
        if response.llm_output:
            outputs["llm_output"] = response.llm_output

        # Write to human-readable text file if paired with chat_model_start
        run_id_str = str(run_id)
        if run_id_str in self._pending_llm_calls:
            pending = self._pending_llm_calls.pop(run_id_str)
            task_id = pending["task_id"]

            # Increment call counter
            self._llm_call_counter[task_id] = self._llm_call_counter.get(task_id, 0) + 1
            call_number = self._llm_call_counter[task_id]

            # Extract response text
            response_text = ""
            if response.generations and response.generations[0]:
                response_text = response.generations[0][0].text

            # Write to text file
            self._write_llm_io_text(
                task_id=task_id,
                call_number=call_number,
                messages=pending["messages"],
                response_text=response_text,
                timestamp=pending["timestamp"],
            )

        event = self._create_event(
            event_type=TraceEventType.LLM_END,
            run_id=run_id,
            name="llm",
            parent_run_id=parent_run_id,
            outputs=outputs,
            tags=tags,
        )
        self.store.add_event(event)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM error."""
        event = self._create_event(
            event_type=TraceEventType.LLM_ERROR,
            run_id=run_id,
            name="llm",
            parent_run_id=parent_run_id,
            error=str(error),
            tags=tags,
        )
        self.store.add_event(event)

    # ========== Chat Model Callbacks ==========

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chat model start."""
        if not self.config.log_llm_io:
            return

        self._record_start(run_id)
        name = _name_from_serialized(serialized, fallback="chat_model")

        # Serialize messages
        serialized_messages = [_serialize_messages(msgs) for msgs in messages]

        # Store for pairing with llm_end
        task_id = self._get_task_id(tags, metadata)
        self._pending_llm_calls[str(run_id)] = {
            "task_id": task_id,
            "messages": serialized_messages[0] if serialized_messages else [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        }

        event = self._create_event(
            event_type=TraceEventType.CHAT_MODEL_START,
            run_id=run_id,
            name=name,
            parent_run_id=parent_run_id,
            inputs={"messages": serialized_messages},
            metadata=metadata,
            tags=tags,
        )
        self.store.add_event(event)

    # ========== Chain Callbacks ==========

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain start."""
        self._record_start(run_id)
        name = _name_from_serialized(serialized, fallback="chain")

        # Safely serialize inputs
        try:
            serialized_inputs = _serialize_value(inputs)
        except Exception as e:
            logger.warning(f"Failed to serialize inputs in on_chain_start: {e}")
            serialized_inputs = {"error": f"Failed to serialize: {str(e)}"}

        # Check if this is a LangGraph node
        is_graph_node = metadata and metadata.get("langgraph_node")

        if is_graph_node and self.config.log_graph_state:
            event = self._create_event(
                event_type=TraceEventType.GRAPH_NODE_START,
                run_id=run_id,
                name=metadata.get("langgraph_node", name),
                parent_run_id=parent_run_id,
                inputs=serialized_inputs,
                metadata=metadata,
                tags=tags,
            )
        else:
            event = self._create_event(
                event_type=TraceEventType.CHAIN_START,
                run_id=run_id,
                name=name,
                parent_run_id=parent_run_id,
                inputs=serialized_inputs,
                metadata=metadata,
                tags=tags,
            )
        self.store.add_event(event)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain end."""
        # Safely serialize outputs
        try:
            serialized_outputs = _serialize_value(outputs)
        except Exception as e:
            logger.warning(f"Failed to serialize outputs in on_chain_end: {e}")
            serialized_outputs = {"error": f"Failed to serialize: {str(e)}"}

        event = self._create_event(
            event_type=TraceEventType.CHAIN_END,
            run_id=run_id,
            name="chain",
            parent_run_id=parent_run_id,
            outputs=serialized_outputs,
            tags=tags,
        )
        self.store.add_event(event)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain error."""
        # Extract detailed error information
        import traceback

        error_msg = str(error) if error else "Unknown error"
        error_type = type(error).__name__ if error else "UnknownError"
        error_traceback = (
            "".join(traceback.format_exception(type(error), error, error.__traceback__))
            if error
            else ""
        )

        # Construct detailed error message
        detailed_error = f"{error_type}: {error_msg}"
        if error_traceback:
            detailed_error += f"\n\nTraceback:\n{error_traceback}"

        # Log the error
        logger.error(f"Chain error in run {run_id}: {detailed_error}")

        # Also print to console for immediate visibility
        print(f"\n{'='*80}")
        print(f"CHAIN ERROR DETECTED")
        print(f"Run ID: {run_id}")
        print(f"Task ID: {self._get_task_id(tags)}")
        print(f"Error Type: {error_type}")
        print(f"Error Message: {error_msg}")
        if error_traceback:
            print(f"\nFull Traceback:\n{error_traceback}")
        print(f"{'='*80}\n")

        event = self._create_event(
            event_type=TraceEventType.CHAIN_ERROR,
            run_id=run_id,
            name="chain",
            parent_run_id=parent_run_id,
            error=detailed_error,
            tags=tags,
        )
        self.store.add_event(event)

    # ========== Tool Callbacks ==========

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool start."""
        if not self.config.log_tool_io:
            return

        self._record_start(run_id)
        name = _name_from_serialized(serialized, fallback="tool")

        event = self._create_event(
            event_type=TraceEventType.TOOL_START,
            run_id=run_id,
            name=name,
            parent_run_id=parent_run_id,
            inputs={"input": input_str, "parsed_inputs": inputs},
            metadata=metadata,
            tags=tags,
        )
        self.store.add_event(event)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool end."""
        if not self.config.log_tool_io:
            return

        event = self._create_event(
            event_type=TraceEventType.TOOL_END,
            run_id=run_id,
            name="tool",
            parent_run_id=parent_run_id,
            outputs={"output": str(output) if output else None},
            tags=tags,
        )
        self.store.add_event(event)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool error."""
        event = self._create_event(
            event_type=TraceEventType.TOOL_ERROR,
            run_id=run_id,
            name="tool",
            parent_run_id=parent_run_id,
            error=str(error),
            tags=tags,
        )
        self.store.add_event(event)

    # ========== Agent Callbacks ==========

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent action."""
        event = self._create_event(
            event_type=TraceEventType.AGENT_ACTION,
            run_id=run_id,
            name=action.tool,
            parent_run_id=parent_run_id,
            inputs={"tool_input": action.tool_input, "log": action.log},
            tags=tags,
        )
        self.store.add_event(event)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent finish."""
        event = self._create_event(
            event_type=TraceEventType.AGENT_FINISH,
            run_id=run_id,
            name="agent",
            parent_run_id=parent_run_id,
            outputs={"return_values": finish.return_values, "log": finish.log},
            tags=tags,
        )
        self.store.add_event(event)

    # ========== Retriever Callbacks ==========

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle retriever start."""
        self._record_start(run_id)
        name = _name_from_serialized(serialized, fallback="retriever")

        event = self._create_event(
            event_type=TraceEventType.RETRIEVER_START,
            run_id=run_id,
            name=name,
            parent_run_id=parent_run_id,
            inputs={"query": query},
            metadata=metadata,
            tags=tags,
        )
        self.store.add_event(event)

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle retriever end."""
        event = self._create_event(
            event_type=TraceEventType.RETRIEVER_END,
            run_id=run_id,
            name="retriever",
            parent_run_id=parent_run_id,
            outputs={
                "documents": [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in documents
                ]
            },
            tags=tags,
        )
        self.store.add_event(event)

    # ========== Async Versions ==========

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Async handle LLM start."""
        if not self.config.log_llm_io:
            return

        self._record_start(run_id)
        name = _name_from_serialized(serialized, fallback="llm")

        event = self._create_event(
            event_type=TraceEventType.LLM_START,
            run_id=run_id,
            name=name,
            parent_run_id=parent_run_id,
            inputs={"prompts": prompts},
            metadata=metadata,
            tags=tags,
        )
        self.store.add_event(event)

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Async handle chat model start."""
        if not self.config.log_llm_io:
            return

        self._record_start(run_id)
        name = _name_from_serialized(serialized, fallback="chat_model")

        # Serialize messages
        serialized_messages = [_serialize_messages(msgs) for msgs in messages]

        # Store for pairing with llm_end
        task_id = self._get_task_id(tags, metadata)
        self._pending_llm_calls[str(run_id)] = {
            "task_id": task_id,
            "messages": serialized_messages[0] if serialized_messages else [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        }

        event = self._create_event(
            event_type=TraceEventType.CHAT_MODEL_START,
            run_id=run_id,
            name=name,
            parent_run_id=parent_run_id,
            inputs={"messages": serialized_messages},
            metadata=metadata,
            tags=tags,
        )
        self.store.add_event(event)

    # ========== Custom Event Method ==========

    def log_custom_event(
        self,
        name: str,
        data: Dict[str, Any],
        task_id: Optional[str] = None,
    ):
        """Log a custom event.

        Args:
            name: Event name
            data: Event data
            task_id: Optional task ID (uses current if not provided)
        """
        from uuid import uuid4

        event = TraceEvent(
            event_type=TraceEventType.CUSTOM,
            timestamp=datetime.now(),
            run_id=str(uuid4()),
            name=name,
            inputs=data,
            task_id=task_id or self._current_task_id or "unknown",
        )
        self.store.add_event(event)

    def finalize(self, task_id: Optional[str] = None):
        """Finalize tracing for a task.

        Args:
            task_id: Optional task ID (uses current if not provided)
        """
        tid = task_id or self._current_task_id
        if tid:
            self.store.finalize_session(tid)

    def finalize_all(self):
        """Finalize all active sessions."""
        for task_id in self.store.list_sessions():
            self.store.finalize_session(task_id)
        self.store.close()
