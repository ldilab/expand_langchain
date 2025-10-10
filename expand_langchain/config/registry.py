"""
Generic registry pattern implementation.
Replaces the duplicate registry code in config_registry and dataset_registry.
"""

import logging
from typing import Dict, Generic, List, TypeVar


class RegistryError(Exception):
    """Raised when registry operations fail."""

    pass


logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Generic registry for storing and retrieving items by name.

    This class provides a type-safe way to register and retrieve items,
    eliminating the need for duplicate registry implementations.
    """

    def __init__(self, registry_type: str):
        """
        Initialize the registry.

        Args:
            registry_type: Human-readable name for the registry type (e.g., 'config', 'dataset')
        """
        self._registry: Dict[str, T] = {}
        self._type = registry_type
        logger.debug(f"Initialized {registry_type} registry")

    def register(self, name: str, item: T) -> None:
        """
        Register an item with a name.

        Args:
            name: Unique identifier for the item
            item: Item to register

        Raises:
            RegistryError: If an item with the same name already exists
        """
        if name in self._registry:
            logger.warning(f"{self._type} '{name}' already exists, overwriting")

        self._registry[name] = item
        logger.info(f"Registered {self._type} '{name}'")

    def get(self, name: str) -> T:
        """
        Retrieve an item by name.

        Args:
            name: Name of the item to retrieve

        Returns:
            The registered item

        Raises:
            RegistryError: If the item is not found
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise RegistryError(
                f"{self._type} '{name}' not found in registry. "
                f"Available items: {available}"
            )

        logger.debug(f"Retrieved {self._type} '{name}'")
        return self._registry[name]

    def list_all(self) -> List[str]:
        """
        List all registered item names.

        Returns:
            List of all registered names
        """
        return list(self._registry.keys())

    def clear(self) -> None:
        """
        Clear all registered items.

        This is primarily useful for testing.
        """
        count = len(self._registry)
        self._registry.clear()
        logger.info(f"Cleared {count} items from {self._type} registry")

    def contains(self, name: str) -> bool:
        """
        Check if an item is registered.

        Args:
            name: Name to check

        Returns:
            True if the item exists, False otherwise
        """
        return name in self._registry

    def remove(self, name: str) -> T:
        """
        Remove and return an item from the registry.

        Args:
            name: Name of the item to remove

        Returns:
            The removed item

        Raises:
            RegistryError: If the item is not found
        """
        if name not in self._registry:
            raise RegistryError(f"{self._type} '{name}' not found in registry")

        item = self._registry.pop(name)
        logger.info(f"Removed {self._type} '{name}' from registry")
        return item

    def update(self, name: str, item: T) -> None:
        """
        Update an existing item in the registry.

        Args:
            name: Name of the item to update
            item: New item value

        Raises:
            RegistryError: If the item is not found
        """
        if name not in self._registry:
            raise RegistryError(f"{self._type} '{name}' not found in registry")

        self._registry[name] = item
        logger.info(f"Updated {self._type} '{name}' in registry")

    def size(self) -> int:
        """
        Get the number of registered items.

        Returns:
            Number of items in the registry
        """
        return len(self._registry)

    def __len__(self) -> int:
        """Support for len() function."""
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        """Support for 'in' operator."""
        return name in self._registry

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"Registry<{self._type}>({list(self._registry.keys())})"
