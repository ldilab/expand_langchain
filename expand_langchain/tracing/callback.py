"""LangChain callback handler for local tracing."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
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
        result.append({
            "type": msg.type,
            "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            "additional_kwargs": getattr(msg, "additional_kwargs", {}),
        })
    return result


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

    def set_task_id(self, task_id: str):
        """Set the current task ID for tracing.

        Args:
            task_id: The task identifier
        """
        self._current_task_id = task_id

    def _get_task_id(self, tags: Optional[List[str]] = None) -> str:
        """Get the current task ID.

        Args:
            tags: Optional tags that may contain task ID

        Returns:
            The task ID
        """
        # Try to get from tags first (LangGraph passes task_id as a tag)
        if tags:
            for tag in tags:
                if tag and not tag.startswith("seq:"):
                    return str(tag)

        # Fall back to explicitly set task ID
        return self._current_task_id or "unknown"

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
            task_id=self._get_task_id(tags),
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

        self._record_start(run_id)
        name = serialized.get("name", serialized.get("id", ["unknown"])[-1])

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
        name = serialized.get("name", serialized.get("id", ["unknown"])[-1])

        # Serialize messages
        serialized_messages = [_serialize_messages(msgs) for msgs in messages]

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
        name = serialized.get("name", serialized.get("id", ["unknown"])[-1])

        # Check if this is a LangGraph node
        is_graph_node = metadata and metadata.get("langgraph_node")

        if is_graph_node and self.config.log_graph_state:
            event = self._create_event(
                event_type=TraceEventType.GRAPH_NODE_START,
                run_id=run_id,
                name=metadata.get("langgraph_node", name),
                parent_run_id=parent_run_id,
                inputs=inputs,
                metadata=metadata,
                tags=tags,
            )
        else:
            event = self._create_event(
                event_type=TraceEventType.CHAIN_START,
                run_id=run_id,
                name=name,
                parent_run_id=parent_run_id,
                inputs=inputs,
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
        event = self._create_event(
            event_type=TraceEventType.CHAIN_END,
            run_id=run_id,
            name="chain",
            parent_run_id=parent_run_id,
            outputs=outputs,
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
        event = self._create_event(
            event_type=TraceEventType.CHAIN_ERROR,
            run_id=run_id,
            name="chain",
            parent_run_id=parent_run_id,
            error=str(error),
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
        name = serialized.get("name", "unknown_tool")

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
        name = serialized.get("name", serialized.get("id", ["unknown"])[-1])

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
        # Delegate to sync version
        super().on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

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
        super().on_chat_model_start(
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

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
