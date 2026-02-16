"""Trace summarizer for generating AI-friendly debug reports."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import TraceEvent, TraceEventType, TraceSession


@dataclass
class ExecutionNode:
    """A node in the execution tree."""
    
    name: str
    node_type: str  # 'chain', 'graph_node', 'tool', 'llm', etc.
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "running"  # 'success', 'error', 'running'
    error: Optional[str] = None
    llm_call_ids: List[int] = field(default_factory=list)
    children: List["ExecutionNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None


@dataclass
class LLMCallSummary:
    """Summary of an LLM call."""
    
    call_id: int
    timestamp: datetime
    node_name: Optional[str]
    model_name: str
    input_messages: List[Dict[str, Any]]
    output_text: str
    duration_ms: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ErrorInfo:
    """Information about an error."""
    
    timestamp: datetime
    node_name: str
    error_type: str
    error_message: str
    context_before: Dict[str, Any]
    recovered: bool = False
    recovery_node: Optional[str] = None
    stack_trace: Optional[str] = None


class TraceSummarizer:
    """Generates AI-friendly summaries from trace sessions."""
    
    def __init__(self, session: TraceSession):
        """Initialize the summarizer.
        
        Args:
            session: The trace session to summarize
        """
        self.session = session
        self.llm_calls: List[LLMCallSummary] = []
        self.errors: List[ErrorInfo] = []
        self.execution_tree: Optional[ExecutionNode] = None
        self._event_by_run_id: Dict[str, List[TraceEvent]] = defaultdict(list)
        self._llm_call_counter = 0
        
    def analyze(self):
        """Analyze the trace session and build summaries."""
        # Index events by run_id
        for event in self.session.events:
            self._event_by_run_id[event.run_id].append(event)
        
        # Build execution tree
        self.execution_tree = self._build_execution_tree()
        
        # Extract LLM calls
        self.llm_calls = self._extract_llm_calls()
        
        # Analyze errors
        self.errors = self._analyze_errors()
        
    def _build_execution_tree(self) -> Optional[ExecutionNode]:
        """Build a hierarchical execution tree from flat events."""
        # Find root event (no parent_run_id)
        root_events = [e for e in self.session.events if e.parent_run_id is None]
        if not root_events:
            return None
        
        root_event = root_events[0]
        return self._build_node_recursive(root_event)
    
    def _build_node_recursive(self, start_event: TraceEvent) -> ExecutionNode:
        """Recursively build execution node from events."""
        run_id = start_event.run_id
        events = self._event_by_run_id[run_id]
        
        # Find end event
        end_event = None
        for event in events:
            if event.event_type.value.endswith("_end") or event.event_type.value.endswith("_finish"):
                end_event = event
                break
        
        # Determine status
        status = "running"
        error = None
        if end_event:
            status = "success"
        
        # Check for errors
        error_events = [e for e in events if e.error is not None]
        if error_events:
            status = "error"
            error = error_events[0].error
        
        # Determine node type
        node_type = self._get_node_type(start_event)
        
        # Create node
        node = ExecutionNode(
            name=start_event.name,
            node_type=node_type,
            start_time=start_event.timestamp,
            end_time=end_event.timestamp if end_event else None,
            duration_ms=end_event.duration_ms if end_event and end_event.duration_ms else None,
            status=status,
            error=error,
            metadata=start_event.metadata or {},
            inputs=start_event.inputs,
            outputs=end_event.outputs if end_event else None,
        )
        
        # Find child events (events with this run_id as parent)
        child_starts = [
            e for e in self.session.events
            if e.parent_run_id == run_id and e.event_type.value.endswith("_start")
        ]
        
        for child_start in child_starts:
            child_node = self._build_node_recursive(child_start)
            node.children.append(child_node)
        
        return node
    
    def _get_node_type(self, event: TraceEvent) -> str:
        """Determine node type from event."""
        event_type = event.event_type.value
        if "llm" in event_type or "chat" in event_type:
            return "llm"
        elif "tool" in event_type:
            return "tool"
        elif "graph" in event_type:
            return "graph_node"
        elif "chain" in event_type:
            return "chain"
        elif "agent" in event_type:
            return "agent"
        return "unknown"
    
    def _extract_llm_calls(self) -> List[LLMCallSummary]:
        """Extract LLM call information."""
        llm_calls = []
        llm_starts = {}
        
        for event in self.session.events:
            if event.event_type in (TraceEventType.LLM_START, TraceEventType.CHAT_MODEL_START):
                llm_starts[event.run_id] = event
            
            elif event.event_type in (TraceEventType.LLM_END, TraceEventType.CHAT_MODEL_END):
                start_event = llm_starts.get(event.run_id)
                if not start_event:
                    continue
                
                self._llm_call_counter += 1
                
                # Extract input messages
                input_messages = []
                if start_event.inputs:
                    if "messages" in start_event.inputs:
                        messages = start_event.inputs["messages"]
                        # Handle nested list structure
                        if isinstance(messages, list) and messages:
                            if isinstance(messages[0], list):
                                # Flatten if nested
                                input_messages = messages[0]
                            else:
                                input_messages = messages
                    elif "prompts" in start_event.inputs:
                        prompts = start_event.inputs["prompts"]
                        if prompts:
                            input_messages = [{"role": "user", "content": prompts[0]}]
                
                # Extract output text
                output_text = ""
                if event.outputs:
                    if "generations" in event.outputs:
                        generations = event.outputs["generations"]
                        if generations and generations[0]:
                            output_text = generations[0][0].get("text", "")
                
                # Get node name from metadata
                node_name = None
                if start_event.metadata:
                    node_name = start_event.metadata.get("langgraph_node")
                
                # Get model name
                model_name = start_event.name
                if start_event.metadata:
                    model_name = start_event.metadata.get("ls_model_name", model_name)
                
                llm_call = LLMCallSummary(
                    call_id=self._llm_call_counter,
                    timestamp=start_event.timestamp,
                    node_name=node_name,
                    model_name=model_name,
                    input_messages=input_messages,
                    output_text=output_text,
                    duration_ms=event.duration_ms,
                    metadata=start_event.metadata or {},
                )
                llm_calls.append(llm_call)
        
        return llm_calls
    
    def _analyze_errors(self) -> List[ErrorInfo]:
        """Analyze error events and their context."""
        errors = []
        
        error_events = [
            e for e in self.session.events
            if e.error is not None or e.event_type.value.endswith("_error")
        ]
        
        for error_event in error_events:
            # Find context before error
            error_time = error_event.timestamp
            events_before = [
                e for e in self.session.events
                if e.timestamp < error_time
            ]
            
            # Get the most recent state
            context_before = {}
            for event in reversed(events_before):
                if event.outputs:
                    context_before = event.outputs
                    break
            
            # Check if error was recovered
            recovered = False
            recovery_node = None
            events_after = [
                e for e in self.session.events
                if e.timestamp > error_time
            ]
            
            # If there are successful events after the error, it was likely recovered
            for event in events_after:
                if (event.event_type.value.endswith("_end") and 
                    event.error is None and
                    event.parent_run_id == error_event.parent_run_id):
                    recovered = True
                    recovery_node = event.name
                    break
            
            error_info = ErrorInfo(
                timestamp=error_event.timestamp,
                node_name=error_event.name,
                error_type=type(error_event.error).__name__ if error_event.error else "Unknown",
                error_message=str(error_event.error) if error_event.error else "Unknown error",
                context_before=context_before,
                recovered=recovered,
                recovery_node=recovery_node,
            )
            errors.append(error_info)
        
        return errors
    
    def generate_summary_md(self) -> str:
        """Generate SUMMARY.md content."""
        lines = ["# Task Execution Summary\n"]
        
        # Overview section
        lines.append("## Overview\n")
        lines.append(f"- **Task ID**: {self.session.task_id}")
        lines.append(f"- **Run Name**: {self.session.run_name}")
        lines.append(f"- **Status**: {'❌ Failed' if self.errors and not all(e.recovered for e in self.errors) else '✅ Success'}")
        
        if self.session.duration_ms:
            duration_s = self.session.duration_ms / 1000
            lines.append(f"- **Total Duration**: {duration_s:.1f}s")
        
        lines.append(f"- **LLM Calls**: {len(self.llm_calls)}")
        lines.append(f"- **Total Events**: {self.session.event_count}")
        
        if self.errors:
            recovered_count = sum(1 for e in self.errors if e.recovered)
            lines.append(f"- **Errors**: {len(self.errors)} ({recovered_count} recovered)")
        
        lines.append("")
        
        # Execution flow
        if self.execution_tree:
            lines.append("## Execution Flow\n")
            self._append_execution_tree(lines, self.execution_tree, indent=0)
            lines.append("")
        
        # Key LLM Calls
        if self.llm_calls:
            lines.append("## LLM Calls Overview\n")
            for call in self.llm_calls:
                status = "✅" if not call.error else "❌"
                duration = f"{call.duration_ms/1000:.1f}s" if call.duration_ms else "?"
                node_info = f" (in {call.node_name})" if call.node_name else ""
                lines.append(f"{status} **Call #{call.call_id}**{node_info} - {call.model_name} - {duration}")
                
                # Brief input summary
                if call.input_messages:
                    msg_count = len(call.input_messages)
                    lines.append(f"   - Input: {msg_count} message(s)")
                
                # Brief output summary
                if call.output_text:
                    preview = call.output_text[:100].replace('\n', ' ')
                    if len(call.output_text) > 100:
                        preview += "..."
                    lines.append(f"   - Output: {preview}")
                
                lines.append("")
        
        # Errors section
        if self.errors:
            lines.append("## Errors & Recovery\n")
            for i, error in enumerate(self.errors, 1):
                status = "✅ Recovered" if error.recovered else "❌ Not Recovered"
                lines.append(f"### Error #{i}: {error.error_type} ({status})\n")
                lines.append(f"- **Location**: {error.node_name}")
                lines.append(f"- **Time**: {error.timestamp.strftime('%H:%M:%S')}")
                lines.append(f"- **Message**: {error.error_message}")
                if error.recovered:
                    lines.append(f"- **Fixed By**: {error.recovery_node}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _append_execution_tree(self, lines: List[str], node: ExecutionNode, indent: int):
        """Recursively append execution tree to lines."""
        prefix = "  " * indent
        
        # Status icon
        if node.status == "success":
            icon = "✅"
        elif node.status == "error":
            icon = "❌"
        else:
            icon = "⏳"
        
        # Duration
        duration = ""
        if node.duration_ms:
            duration = f" ({node.duration_ms/1000:.1f}s)"
        
        # Node info
        node_info = f"{prefix}{icon} **{node.name}**"
        if node.node_type != "chain":
            node_info += f" `[{node.node_type}]`"
        node_info += duration
        
        lines.append(node_info)
        
        # Show LLM calls in this node
        if node.llm_call_ids:
            for call_id in node.llm_call_ids:
                lines.append(f"{prefix}  - 🤖 LLM Call #{call_id}")
        
        # Show error if present
        if node.error:
            error_preview = node.error[:100].replace('\n', ' ')
            if len(node.error) > 100:
                error_preview += "..."
            lines.append(f"{prefix}  ⚠️ *{error_preview}*")
        
        # Recurse to children
        for child in node.children:
            self._append_execution_tree(lines, child, indent + 1)
    
    def generate_llm_calls_md(self, max_content_length: int = 5000) -> str:
        """Generate LLM_CALLS.md content."""
        lines = ["# LLM Call Details\n"]
        lines.append(f"Total LLM calls: {len(self.llm_calls)}\n")
        
        for call in self.llm_calls:
            lines.append(f"## Call #{call.call_id} - {call.node_name or 'Unknown Node'}\n")
            lines.append(f"- **Timestamp**: {call.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"- **Model**: {call.model_name}")
            
            if call.duration_ms:
                lines.append(f"- **Duration**: {call.duration_ms/1000:.1f}s")
            
            lines.append("")
            
            # Input messages
            lines.append("### Input Messages\n")
            for i, msg in enumerate(call.input_messages, 1):
                # Handle both dict and other formats
                if isinstance(msg, dict):
                    role = msg.get("role", msg.get("type", "unknown")).upper()
                    content = str(msg.get("content", ""))
                else:
                    role = "UNKNOWN"
                    content = str(msg)
                
                # Truncate if too long
                if len(content) > max_content_length:
                    content = content[:max_content_length] + f"\n... [truncated, total {len(content)} chars]"
                
                lines.append(f"#### Message {i} - [{role}]\n")
                lines.append("```")
                lines.append(content)
                lines.append("```\n")
            
            # Output
            lines.append("### Output\n")
            output = call.output_text
            if len(output) > max_content_length:
                output = output[:max_content_length] + f"\n... [truncated, total {len(output)} chars]"
            
            lines.append("```")
            lines.append(output)
            lines.append("```\n")
            
            lines.append("---\n")
        
        return "\n".join(lines)
    
    def generate_error_analysis_md(self) -> str:
        """Generate ERROR_ANALYSIS.md content."""
        if not self.errors:
            return "# Error Analysis\n\nNo errors occurred during execution. ✅\n"
        
        lines = ["# Error Analysis\n"]
        lines.append(f"Total errors: {len(self.errors)}")
        lines.append(f"Recovered: {sum(1 for e in self.errors if e.recovered)}\n")
        
        for i, error in enumerate(self.errors, 1):
            lines.append(f"## Error #{i}: {error.error_type}\n")
            lines.append(f"### When & Where\n")
            lines.append(f"- **Timestamp**: {error.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"- **Node**: {error.node_name}")
            lines.append(f"- **Status**: {'✅ Recovered' if error.recovered else '❌ Not Recovered'}")
            
            if error.recovered and error.recovery_node:
                lines.append(f"- **Recovered By**: {error.recovery_node}")
            
            lines.append("")
            
            lines.append("### Error Details\n")
            lines.append("```")
            lines.append(error.error_message)
            lines.append("```\n")
            
            if error.context_before:
                lines.append("### Context Before Error\n")
                lines.append("```json")
                lines.append(json.dumps(error.context_before, indent=2, ensure_ascii=False))
                lines.append("```\n")
            
            lines.append("---\n")
        
        return "\n".join(lines)
    
    def generate_debug_guide_md(self) -> str:
        """Generate DEBUG_GUIDE.md content."""
        lines = ["# Debug Guide for AI Agents\n"]
        
        # Quick status
        lines.append("## Quick Status Check\n")
        
        if not self.errors:
            lines.append("✅ Task completed successfully with no errors.\n")
        else:
            unrecovered = [e for e in self.errors if not e.recovered]
            if unrecovered:
                lines.append(f"❌ Task failed with {len(unrecovered)} unrecovered error(s).\n")
            else:
                lines.append(f"✅ Task completed successfully (recovered from {len(self.errors)} error(s)).\n")
        
        # How to investigate
        lines.append("## How to Investigate\n")
        lines.append("1. **Start with** `SUMMARY.md` to understand the overall execution flow")
        lines.append("2. **Check** `ERROR_ANALYSIS.md` if there were any errors")
        lines.append("3. **Review** `LLM_CALLS.md` to see what the LLM generated at each step")
        lines.append("4. **Examine** the original trace files (`*.jsonl`) for raw event data\n")
        
        # Common issues
        if self.errors:
            lines.append("## Issues Found in This Run\n")
            for i, error in enumerate(self.errors, 1):
                status = "✅ Fixed" if error.recovered else "❌ Still broken"
                lines.append(f"{i}. **{error.error_type}** at {error.node_name} - {status}")
                lines.append(f"   - See ERROR_ANALYSIS.md for details")
                
                if error.recovered:
                    lines.append(f"   - Recovery strategy: {error.recovery_node} node handled it")
                else:
                    lines.append(f"   - **Action needed**: This error blocked execution")
                
                lines.append("")
        
        # Debugging tips
        lines.append("## Debugging Tips\n")
        lines.append("- If an LLM generated incorrect output, check the corresponding call in `LLM_CALLS.md`")
        lines.append("- Look for patterns in error recovery to understand the agent's self-correction logic")
        lines.append("- Check node execution order in `SUMMARY.md` to understand the workflow")
        lines.append("- Duration metrics can help identify performance bottlenecks\n")
        
        return "\n".join(lines)
    
    def generate_summary_json(self) -> Dict[str, Any]:
        """Generate structured summary as JSON."""
        return {
            "task_id": self.session.task_id,
            "run_name": self.session.run_name,
            "status": "success" if not any(not e.recovered for e in self.errors) else "failed",
            "start_time": self.session.start_time.isoformat(),
            "end_time": self.session.end_time.isoformat() if self.session.end_time else None,
            "duration_ms": self.session.duration_ms,
            "event_count": self.session.event_count,
            "llm_calls": [
                {
                    "call_id": call.call_id,
                    "timestamp": call.timestamp.isoformat(),
                    "node": call.node_name,
                    "model": call.model_name,
                    "duration_ms": call.duration_ms,
                    "input_message_count": len(call.input_messages),
                    "output_length": len(call.output_text),
                    "error": call.error,
                }
                for call in self.llm_calls
            ],
            "errors": [
                {
                    "timestamp": error.timestamp.isoformat(),
                    "node": error.node_name,
                    "type": error.error_type,
                    "message": error.error_message,
                    "recovered": error.recovered,
                    "recovery_node": error.recovery_node,
                }
                for error in self.errors
            ],
            "execution_tree": self._node_to_dict(self.execution_tree) if self.execution_tree else None,
        }
    
    def _node_to_dict(self, node: ExecutionNode) -> Dict[str, Any]:
        """Convert execution node to dictionary."""
        return {
            "name": node.name,
            "type": node.node_type,
            "status": node.status,
            "duration_ms": node.duration_ms,
            "error": node.error,
            "llm_calls": node.llm_call_ids,
            "children": [self._node_to_dict(child) for child in node.children],
        }


def generate_ai_summary(session: TraceSession, output_dir: Path, max_content_length: int = 5000):
    """Generate all AI-friendly summary files for a trace session.
    
    Args:
        session: The trace session to summarize
        output_dir: Directory to write summary files
        max_content_length: Maximum content length for LLM I/O
    """
    summarizer = TraceSummarizer(session)
    summarizer.analyze()
    
    task_id = session.task_id
    
    # Generate SUMMARY.md
    summary_md = summarizer.generate_summary_md()
    (output_dir / f"{task_id}_SUMMARY.md").write_text(summary_md, encoding="utf-8")
    
    # Generate LLM_CALLS.md
    llm_calls_md = summarizer.generate_llm_calls_md(max_content_length)
    (output_dir / f"{task_id}_LLM_CALLS.md").write_text(llm_calls_md, encoding="utf-8")
    
    # Generate ERROR_ANALYSIS.md
    error_md = summarizer.generate_error_analysis_md()
    (output_dir / f"{task_id}_ERROR_ANALYSIS.md").write_text(error_md, encoding="utf-8")
    
    # Generate DEBUG_GUIDE.md
    debug_md = summarizer.generate_debug_guide_md()
    (output_dir / f"{task_id}_DEBUG_GUIDE.md").write_text(debug_md, encoding="utf-8")
    
    # Generate summary.json
    summary_json = summarizer.generate_summary_json()
    (output_dir / f"{task_id}_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
