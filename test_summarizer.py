"""Test script for trace summarizer."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from expand_langchain.tracing.models import TraceSession
from expand_langchain.tracing.summarizer import generate_ai_summary


def load_yaml_trace_to_session(yaml_path: Path) -> TraceSession:
    """Load YAML trace and convert to TraceSession.
    
    This function is kept for potential future use but not currently needed.
    """
    pass


def test_with_real_trace():
    """Test summarizer with a real trace file."""
    # Find a real trace file
    trace_dir = Path("/root/Projects/PAPERS/MedSQL-paper/source_code/MedSQL/results/20260127_015001-reforce-spider2-claude-spider2-snow/traces")
    
    if not trace_dir.exists():
        print(f"Trace directory not found: {trace_dir}")
        return
    
    # Try JSON files first (more common)
    json_files = list(trace_dir.glob("*.json"))
    # Filter out full_run_histories files
    json_files = [f for f in json_files if "_full_run_histories" not in f.name]
    
    if not json_files:
        print(f"No JSON trace files found in {trace_dir}")
        return
    
    json_file = json_files[0]
    print(f"\nLoading trace from: {json_file}")
    
    try:
        # Load session from JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session = TraceSession.from_dict(data)
        print(f"Loaded session: {session.task_id}")
        print(f"  Events: {session.event_count}")
        print(f"  Errors: {session.error_count}")
        
        # Create output directory for test
        test_output = Path("/tmp/trace_summary_test")
        test_output.mkdir(exist_ok=True)
        
        print(f"\nGenerating AI summary to: {test_output}")
        
        # Generate summary
        generate_ai_summary(
            session=session,
            output_dir=test_output,
            max_content_length=3000,
        )
        
        print("\n✅ Summary files generated:")
        for file in sorted(test_output.glob(f"{session.task_id}_*")):
            size_kb = file.stat().st_size / 1024
            print(f"  - {file.name} ({size_kb:.1f} KB)")
        
        # Show preview of SUMMARY.md
        summary_file = test_output / f"{session.task_id}_SUMMARY.md"
        if summary_file.exists():
            print("\n" + "="*80)
            print("Preview of SUMMARY.md:")
            print("="*80)
            content = summary_file.read_text(encoding='utf-8')
            lines = content.split('\n')[:50]  # First 50 lines
            print('\n'.join(lines))
            if len(content.split('\n')) > 50:
                print("\n... (truncated)")
        
        print("\n" + "="*80)
        print(f"\nAll files available at: {test_output}")
        print("You can inspect them to verify the output.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing Trace Summarizer")
    print("="*80)
    test_with_real_trace()
