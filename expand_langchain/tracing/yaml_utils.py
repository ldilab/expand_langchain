"""YAML helpers for trace and snapshot serialization."""

from __future__ import annotations

from io import StringIO
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


def _create_yaml() -> YAML:
    """Create a YAML instance with consistent settings.

    Returns:
        Configured YAML instance.
    """
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.allow_unicode = True
    yaml.width = 120
    return yaml


def sanitize_for_yaml(
    value: Any,
    *,
    seen: set[int] | None = None,
    depth: int = 0,
    max_depth: int = 50,
) -> Any:
    """Sanitize values for YAML serialization.

    Args:
        value: Value to sanitize.
        seen: Set of visited object IDs (for recursion detection).
        depth: Current recursion depth.
        max_depth: Maximum recursion depth.

    Returns:
        YAML-safe value.
    """
    if seen is None:
        seen = set()
    if depth > max_depth:
        return "<max_depth_reached>"

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    value_id = id(value)
    if value_id in seen:
        return "<recursion_detected>"
    seen.add(value_id)

    if isinstance(value, list):
        return [
            sanitize_for_yaml(item, seen=seen, depth=depth + 1, max_depth=max_depth)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: sanitize_for_yaml(val, seen=seen, depth=depth + 1, max_depth=max_depth)
            for key, val in value.items()
        }

    return str(value)


def stringify_multiline(value: Any) -> Any:
    """Convert multiline or long strings to YAML literal blocks.

    Args:
        value: Value to process.

    Returns:
        YAML-friendly value with LiteralScalarString for long text.
    """
    if isinstance(value, str):
        if "\n" in value or len(value) > 160:
            return LiteralScalarString(value)
        return value
    if isinstance(value, list):
        return [stringify_multiline(item) for item in value]
    if isinstance(value, dict):
        return {key: stringify_multiline(val) for key, val in value.items()}
    return value


def dump_yaml(payload: Any) -> str:
    """Dump a payload to YAML string.

    Args:
        payload: Payload to serialize.

    Returns:
        YAML string.
    """
    buffer = StringIO()
    sanitized = sanitize_for_yaml(payload)
    yaml = _create_yaml()
    yaml.dump(stringify_multiline(sanitized), buffer)
    return buffer.getvalue()


def append_yaml_list_item(path, item: Any) -> None:
    """Append a YAML list item to a file.

    Args:
        path: Path to the YAML file.
        item: Item to append as a list element.
    """
    text = dump_yaml([item]).rstrip()
    if not text:
        return

    if not path.exists() or path.stat().st_size == 0:
        path.write_text(text + "\n", encoding="utf-8")
        return

    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + text + "\n")
