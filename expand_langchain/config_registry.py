"""Config registry for automatic root_node loading."""

import logging
from typing import Any, Dict, Generic, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RegistryError(Exception):
    """Raised when registry operations fail."""

    pass


class Registry(Generic[T]):
    """Generic registry for storing and retrieving items by name."""

    def __init__(self, registry_type: str):
        self._registry: Dict[str, T] = {}
        self._type = registry_type

    def register(self, name: str, item: T) -> None:
        """Register an item with a name."""
        if name in self._registry:
            logger.warning(f"{self._type} '{name}' already exists, overwriting")

        self._registry[name] = item
        logger.info(f"Registered {self._type} '{name}'")

    def get(self, name: str) -> T:
        """Retrieve an item by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"{self._type} '{name}' not found in registry. "
                f"Available items: {available}"
            )

        return self._registry[name]

    def list_all(self) -> List[str]:
        """List all registered item names."""
        return list(self._registry.keys())

    def clear(self) -> None:
        """Clear all registered items."""
        self._registry.clear()
        logger.info(f"Cleared {self._type} registry")


# Global registry instance using the new unified registry
_config_registry = Registry[Any]("config")


def register_config(name: str, root_node: Any):
    """Register a root_node with a config name."""
    _config_registry.register(name, root_node)


def get_config(name: str) -> Any:
    """Get root_node by config name."""
    return _config_registry.get(name)


def list_configs() -> list:
    """List all registered config names."""
    return _config_registry.list_all()


def clear_registry():
    """Clear the registry (for testing)."""
    _config_registry.clear()
