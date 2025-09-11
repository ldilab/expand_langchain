"""Config registry for automatic root_node loading."""

import logging
from typing import Any, Dict

# Global registry to store root_nodes
_CONFIG_REGISTRY: Dict[str, Any] = {}


def register_config(name: str, root_node: Any):
    """Register a root_node with a config name."""
    _CONFIG_REGISTRY[name] = root_node
    logging.info(f"Registered config '{name}' with root_node")


def get_config(name: str) -> Any:
    """Get root_node by config name."""
    if name in _CONFIG_REGISTRY:
        return _CONFIG_REGISTRY[name]
    else:
        raise ValueError(f"Config '{name}' not found in registry")


def list_configs() -> list:
    """List all registered config names."""
    return list(_CONFIG_REGISTRY.keys())


def clear_registry():
    """Clear the registry (for testing)."""
    global _CONFIG_REGISTRY
    _CONFIG_REGISTRY = {}
