"""CLI tool for viewing traces.

Usage:
    uv run python -m expand_langchain.tracing.viewer results/run-name/traces/
    uv run python -m expand_langchain.tracing.viewer results/run-name/traces/ --task-id task_0
    uv run python -m expand_langchain.tracing.viewer results/run-name/traces/ --summary
    uv run python -m expand_langchain.tracing.viewer results/run-name/traces/ --watch
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import TraceEvent, TraceEventType, TraceSession
from .store import TraceStore


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"


def colorize(text: str, color: str) -> str:
    """Add color to text if terminal supports it."""
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{Colors.RESET}"


def get_event_color(event_type: TraceEventType) -> str:
    """Get color for event type."""
    if "error" in event_type.value.lower():
        return Colors.RED
    elif event_type.value.endswith("_start"):
        return Colors.GREEN
    elif event_type.value.endswith("_end"):
        return Colors.BLUE
    elif event_type == TraceEventType.AGENT_ACTION:
        return Colors.YELLOW
    elif event_type == TraceEventType.AGENT_FINISH:
        return Colors.MAGENTA
    elif "llm" in event_type.value.lower() or "chat" in event_type.value.lower():
        return Colors.CYAN
    return Colors.WHITE


def format_duration(duration_ms: Optional[float]) -> str:
    """Format duration in a human-readable way."""
    if duration_ms is None:
        return ""
    if duration_ms < 1000:
        return f"{duration_ms:.0f}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.1f}s"
    else:
        return f"{duration_ms / 60000:.1f}m"


def format_timestamp(dt: datetime) -> str:
    """Format timestamp."""
    return dt.strftime("%H:%M:%S.%f")[:-3]


def truncate_string(s: str, max_length: int = 100) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_length:
        return s
    return s[: max_length - 3] + "..."


def print_event(event: TraceEvent, indent: int = 0, verbose: bool = False):
    """Print a single trace event."""
    prefix = "  " * indent
    color = get_event_color(event.event_type)
    timestamp = format_timestamp(event.timestamp)
    duration = format_duration(event.duration_ms)

    # Event header
    header_parts = [
        colorize(f"[{timestamp}]", Colors.DIM),
        colorize(event.event_type.value, color),
        colorize(f"({event.name})", Colors.BOLD),
    ]
    if duration:
        header_parts.append(colorize(f"[{duration}]", Colors.YELLOW))
    if event.error:
        header_parts.append(colorize("ERROR", Colors.BG_RED))

    print(f"{prefix}{' '.join(header_parts)}")

    # Show inputs/outputs in verbose mode
    if verbose:
        if event.inputs:
            input_str = json.dumps(event.inputs, ensure_ascii=False, default=str)
            print(f"{prefix}  Input: {truncate_string(input_str, 200)}")
        if event.outputs:
            output_str = json.dumps(event.outputs, ensure_ascii=False, default=str)
            print(f"{prefix}  Output: {truncate_string(output_str, 200)}")
        if event.error:
            print(f"{prefix}  {colorize('Error:', Colors.RED)} {event.error[:200]}")


def print_session_summary(session: TraceSession):
    """Print a summary of a trace session."""
    print(colorize(f"\n{'=' * 60}", Colors.DIM))
    print(colorize(f"Task: {session.task_id}", Colors.BOLD))
    print(colorize(f"Run: {session.run_name}", Colors.DIM))
    print(f"Start: {format_timestamp(session.start_time)}")
    if session.end_time:
        print(f"End: {format_timestamp(session.end_time)}")
        duration = format_duration(session.duration_ms)
        print(f"Duration: {duration}")
    print(f"Events: {session.event_count}")
    if session.error_count > 0:
        print(colorize(f"Errors: {session.error_count}", Colors.RED))

    # Event type breakdown
    type_counts: Dict[str, int] = {}
    for event in session.events:
        type_counts[event.event_type.value] = type_counts.get(event.event_type.value, 0) + 1

    print("\nEvent breakdown:")
    for event_type, count in sorted(type_counts.items()):
        print(f"  {event_type}: {count}")


def print_session_events(session: TraceSession, verbose: bool = False):
    """Print all events in a session."""
    print_session_summary(session)
    print(colorize(f"\n{'=' * 60}", Colors.DIM))
    print(colorize("Events:", Colors.BOLD))
    print()

    for event in session.events:
        print_event(event, indent=0, verbose=verbose)

    print()


def print_all_sessions_summary(sessions: Dict[str, TraceSession]):
    """Print a summary of all sessions."""
    print(colorize(f"\n{'=' * 60}", Colors.DIM))
    print(colorize(f"Trace Summary", Colors.BOLD))
    print(f"Total sessions: {len(sessions)}")

    total_events = sum(s.event_count for s in sessions.values())
    total_errors = sum(s.error_count for s in sessions.values())
    print(f"Total events: {total_events}")
    if total_errors > 0:
        print(colorize(f"Total errors: {total_errors}", Colors.RED))

    print(colorize(f"\n{'=' * 60}", Colors.DIM))
    print(colorize("Sessions:", Colors.BOLD))

    for task_id, session in sorted(sessions.items()):
        status = colorize("DONE", Colors.GREEN) if session.end_time else colorize("RUNNING", Colors.YELLOW)
        duration = format_duration(session.duration_ms) if session.duration_ms else "..."
        error_indicator = colorize(f" ({session.error_count} errors)", Colors.RED) if session.error_count else ""
        print(f"  [{status}] {task_id}: {session.event_count} events, {duration}{error_indicator}")

    print()


def watch_realtime_log(log_path: Path):
    """Watch the realtime log file (like tail -f)."""
    print(colorize(f"Watching {log_path}...", Colors.DIM))
    print(colorize("Press Ctrl+C to stop\n", Colors.DIM))

    # Read existing content
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
            if content:
                print(content, end="")

    # Watch for new content
    last_pos = log_path.stat().st_size if log_path.exists() else 0
    try:
        while True:
            if log_path.exists():
                current_size = log_path.stat().st_size
                if current_size > last_pos:
                    with open(log_path, "r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        new_content = f.read()
                        print(new_content, end="", flush=True)
                    last_pos = current_size
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(colorize("\nStopped watching.", Colors.DIM))


def main():
    """Main entry point for the viewer CLI."""
    parser = argparse.ArgumentParser(
        description="View local traces from MedSQL runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View summary of all traces
  uv run python -m expand_langchain.tracing.viewer results/run-name/traces/ --summary

  # View all events for a specific task
  uv run python -m expand_langchain.tracing.viewer results/run-name/traces/ --task-id task_0

  # View with verbose output (shows inputs/outputs)
  uv run python -m expand_langchain.tracing.viewer results/run-name/traces/ --task-id task_0 -v

  # Watch realtime log
  uv run python -m expand_langchain.tracing.viewer results/run-name/traces/ --watch
        """,
    )
    parser.add_argument(
        "traces_dir",
        type=str,
        help="Path to the traces directory",
    )
    parser.add_argument(
        "--task-id",
        "-t",
        type=str,
        help="Show events for a specific task ID",
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Show summary only (no individual events)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including inputs/outputs",
    )
    parser.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help="Watch the realtime log file",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output raw JSON instead of formatted text",
    )

    args = parser.parse_args()
    traces_dir = Path(args.traces_dir)

    if not traces_dir.exists():
        print(colorize(f"Error: Directory not found: {traces_dir}", Colors.RED), file=sys.stderr)
        sys.exit(1)

    # Watch mode
    if args.watch:
        log_path = traces_dir / "realtime.log"
        watch_realtime_log(log_path)
        return

    # Load sessions
    sessions = TraceStore.load_traces_dir(traces_dir)

    if not sessions:
        # Try loading from JSONL files
        for jsonl_file in traces_dir.glob("*.jsonl"):
            task_id = jsonl_file.stem
            events = list(TraceStore.load_events(jsonl_file))
            if events:
                session = TraceSession(
                    task_id=task_id,
                    run_name="unknown",
                    start_time=events[0].timestamp,
                    end_time=events[-1].timestamp if events else None,
                )
                session.events = events
                sessions[task_id] = session

    if not sessions:
        print(colorize("No traces found in directory.", Colors.YELLOW))
        sys.exit(0)

    # JSON output
    if args.json:
        if args.task_id:
            if args.task_id in sessions:
                print(json.dumps(sessions[args.task_id].to_dict(), indent=2, ensure_ascii=False))
            else:
                print(colorize(f"Task not found: {args.task_id}", Colors.RED), file=sys.stderr)
                sys.exit(1)
        else:
            print(json.dumps({k: v.to_dict() for k, v in sessions.items()}, indent=2, ensure_ascii=False))
        return

    # Specific task
    if args.task_id:
        if args.task_id in sessions:
            session = sessions[args.task_id]
            if args.summary:
                print_session_summary(session)
            else:
                print_session_events(session, verbose=args.verbose)
        else:
            print(colorize(f"Task not found: {args.task_id}", Colors.RED), file=sys.stderr)
            print(f"Available tasks: {', '.join(sorted(sessions.keys()))}")
            sys.exit(1)
    else:
        # All sessions
        if args.summary:
            print_all_sessions_summary(sessions)
        else:
            # Show summary and ask for task selection
            print_all_sessions_summary(sessions)
            print(colorize("Use --task-id <id> to view specific task events", Colors.DIM))


if __name__ == "__main__":
    main()
