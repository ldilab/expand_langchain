"""Test hierarchical trace generation with real generator run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from expand_langchain.generator import Generator


def test_trace_generation():
    """Run a small test to generate hierarchical trace."""
    print("=" * 80)
    print("Testing Hierarchical Trace Generation")
    print("=" * 80)
    
    # Create generator with tracing enabled
    generator = Generator(
        config_name="ours",
        dataset_name="spider2",
        tracing_on=True,
        tracing_ai_summary=True,  # Enable hierarchical trace
    )
    
    # Get trace directory from trace callback if available
    trace_dir = Path("results") / generator.run_name / "traces"
    print(f"\nTrace directory: {trace_dir}")
    print("\nRunning 1 test instance...")
    
    try:
        # Run just 1 instance for quick test
        generator.run(n=1)
        
        print("\n" + "=" * 80)
        print("Test completed!")
        print("=" * 80)
        
        # Check generated files
        print(f"\nGenerated trace files in: {trace_dir}")
        print("\nFiles:")
        for file in sorted(trace_dir.glob("*")):
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"  - {file.name} ({size_kb:.1f} KB)")
        
        # Show debug trace if exists
        debug_files = list(trace_dir.glob("*_debug.yaml"))
        if debug_files:
            debug_file = debug_files[0]
            print(f"\n" + "=" * 80)
            print(f"Preview of {debug_file.name}:")
            print("=" * 80)
            content = debug_file.read_text(encoding='utf-8')
            lines = content.split('\n')[:80]  # First 80 lines
            print('\n'.join(lines))
            if len(content.split('\n')) > 80:
                print("\n... (truncated)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_trace_generation()
    sys.exit(0 if success else 1)
