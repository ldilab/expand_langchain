"""Test hierarchical trace builder with existing trace data."""

import json
from pathlib import Path

from expand_langchain.tracing.hierarchy import build_hierarchical_trace
from expand_langchain.tracing.models import TraceSession
from expand_langchain.tracing.yaml_utils import dump_yaml


def test_with_existing_trace():
    """Test hierarchical trace generation with existing trace file."""
    # Use existing trace file
    trace_file = Path("/root/Projects/PAPERS/MedSQL-paper/source_code/MedSQL/results/20260127_015001-reforce-spider2-claude-spider2-snow/traces/0.json")
    
    if not trace_file.exists():
        print(f"Trace file not found: {trace_file}")
        return False
    
    print("=" * 80)
    print("Testing Hierarchical Trace Builder")
    print("=" * 80)
    print(f"\nLoading trace from: {trace_file}")
    
    try:
        # Load session
        with open(trace_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session = TraceSession.from_dict(data)
        print(f"Loaded session: {session.task_id}")
        print(f"  Events: {session.event_count}")
        print(f"  Errors: {session.error_count}")
        
        # Build hierarchical structure
        print("\nBuilding hierarchical trace...")
        hierarchical = build_hierarchical_trace(session.events)
        
        # Convert to YAML
        yaml_output = dump_yaml(hierarchical)
        
        # Save output
        output_file = Path("/root/Projects/PAPERS/MedSQL-paper/source_code/MedSQL/third_party/expand_langchain.worktrees/copilot-worktree-2026-02-16T11-08-35/test_debug_trace.yaml")
        output_file.write_text(yaml_output, encoding='utf-8')
        
        print(f"\n✅ Hierarchical trace generated")
        print(f"   Output: {output_file}")
        print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
        
        # Show preview
        print("\n" + "=" * 80)
        print("Preview (first 100 lines):")
        print("=" * 80)
        lines = yaml_output.split('\n')[:100]
        print('\n'.join(lines))
        if len(yaml_output.split('\n')) > 100:
            print("\n... (truncated)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_with_existing_trace()
    sys.exit(0 if success else 1)
