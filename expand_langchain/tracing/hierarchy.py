"""Enhanced trace structure builder for AI-friendly debugging."""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .models import TraceEvent, TraceEventType


class TraceNode:
    """A node in the hierarchical trace structure."""
    
    def __init__(self, event: TraceEvent):
        self.run_id = event.run_id
        self.event_type = event.event_type
        self.name = event.name
        self.start_time = event.timestamp
        self.end_time: Optional[datetime] = None
        self.duration_ms: Optional[float] = None
        self.inputs = event.inputs
        self.outputs: Optional[Dict[str, Any]] = None
        self.metadata = event.metadata or {}
        self.error: Optional[str] = None
        self.tags = event.tags
        self.children: List[TraceNode] = []
        
        # LLM-specific fields
        self.llm_input_messages: Optional[List[Dict[str, Any]]] = None
        self.llm_output_text: Optional[str] = None
        self.llm_model: Optional[str] = None
        
        # Node context
        self.node_name: Optional[str] = None
        if self.metadata:
            self.node_name = self.metadata.get("langgraph_node")
            self.llm_model = self.metadata.get("ls_model_name")
    
    def finalize(self, end_event: TraceEvent):
        """Finalize the node with end event data."""
        self.end_time = end_event.timestamp
        self.duration_ms = end_event.duration_ms
        self.outputs = end_event.outputs
        if end_event.error:
            self.error = end_event.error
    
    def add_child(self, child: "TraceNode"):
        """Add a child node."""
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result = {
            "type": self._get_node_type(),
            "name": self.name,
            "start": self.start_time.isoformat(),
        }
        
        if self.node_name:
            result["graph_node"] = self.node_name
        
        if self.end_time:
            result["end"] = self.end_time.isoformat()
            result["duration_ms"] = self.duration_ms
        
        # Status
        if self.error:
            result["status"] = "error"
            result["error"] = self.error
        elif self.end_time:
            result["status"] = "success"
        else:
            result["status"] = "running"
        
        # LLM-specific info (grouped together)
        if self.llm_input_messages or self.llm_output_text:
            llm_info = {}
            if self.llm_model:
                llm_info["model"] = self.llm_model
            if self.llm_input_messages:
                llm_info["input_messages"] = self.llm_input_messages
            if self.llm_output_text:
                llm_info["output"] = self.llm_output_text
            result["llm"] = llm_info
        
        # Inputs/outputs for non-LLM nodes
        elif self.inputs:
            result["inputs"] = self.inputs
        
        if self.outputs and not self.llm_output_text:
            result["outputs"] = self.outputs
        
        # Add children
        if self.children:
            result["steps"] = [child.to_dict() for child in self.children]
        
        return result
    
    def _get_node_type(self) -> str:
        """Get simplified node type."""
        event_val = self.event_type.value
        if "llm" in event_val or "chat" in event_val:
            return "llm_call"
        elif "tool" in event_val:
            return "tool"
        elif "graph" in event_val:
            return "graph_node"
        elif "chain" in event_val:
            return "chain"
        elif "agent" in event_val:
            return "agent"
        return "unknown"


class HierarchicalTraceBuilder:
    """Builds hierarchical trace structure from flat events."""
    
    def __init__(self):
        self.nodes: Dict[str, TraceNode] = {}
        self.root_nodes: List[TraceNode] = []
        self.llm_starts: Dict[str, TraceNode] = {}
    
    def add_event(self, event: TraceEvent):
        """Add an event to the trace structure."""
        run_id = event.run_id
        
        # Handle start events
        if event.event_type.value.endswith("_start"):
            node = TraceNode(event)
            self.nodes[run_id] = node
            
            # Track LLM starts separately
            if event.event_type in (TraceEventType.LLM_START, TraceEventType.CHAT_MODEL_START):
                self.llm_starts[run_id] = node
                # Extract input messages
                if event.inputs:
                    if "messages" in event.inputs:
                        messages = event.inputs["messages"]
                        # Handle nested list
                        if isinstance(messages, list) and messages and isinstance(messages[0], list):
                            node.llm_input_messages = messages[0]
                        else:
                            node.llm_input_messages = messages
            
            # Add to parent or root
            if event.parent_run_id:
                parent = self.nodes.get(event.parent_run_id)
                if parent:
                    parent.add_child(node)
            else:
                self.root_nodes.append(node)
        
        # Handle end events
        elif event.event_type.value.endswith("_end") or event.event_type.value.endswith("_finish"):
            node = self.nodes.get(run_id)
            if node:
                node.finalize(event)
                
                # Extract LLM output
                if event.event_type in (TraceEventType.LLM_END, TraceEventType.CHAT_MODEL_END):
                    if event.outputs and "generations" in event.outputs:
                        generations = event.outputs["generations"]
                        if generations and generations[0]:
                            node.llm_output_text = generations[0][0].get("text", "")
        
        # Handle error events
        elif event.event_type.value.endswith("_error"):
            node = self.nodes.get(run_id)
            if node:
                node.error = event.error
        
        # Handle agent actions
        elif event.event_type == TraceEventType.AGENT_ACTION:
            node = TraceNode(event)
            self.nodes[run_id] = node
            
            if event.parent_run_id:
                parent = self.nodes.get(event.parent_run_id)
                if parent:
                    parent.add_child(node)
            else:
                self.root_nodes.append(node)
    
    def build(self) -> List[Dict[str, Any]]:
        """Build the hierarchical structure."""
        return [node.to_dict() for node in self.root_nodes]


def build_hierarchical_trace(events: List[TraceEvent]) -> Dict[str, Any]:
    """Build a hierarchical, AI-friendly trace structure from events.
    
    Args:
        events: List of trace events
        
    Returns:
        Hierarchical trace structure as a dictionary
    """
    if not events:
        return {"execution": []}
    
    builder = HierarchicalTraceBuilder()
    
    # Process all events
    for event in events:
        builder.add_event(event)
    
    # Build final structure
    execution_tree = builder.build()
    
    # Add metadata
    first_event = events[0]
    last_event = events[-1]
    
    # Count stats
    llm_calls = sum(1 for e in events if e.event_type in (TraceEventType.LLM_END, TraceEventType.CHAT_MODEL_END))
    errors = sum(1 for e in events if e.error is not None)
    
    result = {
        "task_id": first_event.task_id or "unknown",
        "start_time": first_event.timestamp.isoformat(),
        "end_time": last_event.timestamp.isoformat(),
        "total_events": len(events),
        "llm_calls": llm_calls,
        "errors": errors,
        "execution": execution_tree,
    }
    
    return result
