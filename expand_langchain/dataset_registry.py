"""Dataset registry for automatic dataset loading."""

import logging
from typing import Dict, Generic, List, TypeVar

from .loader import MultiSourceDatasetMerger

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
_dataset_registry = Registry[MultiSourceDatasetMerger]("dataset")


def register_dataset(name: str, dataset: MultiSourceDatasetMerger):
    """Register a dataset with a name."""
    _dataset_registry.register(name, dataset)


def get_dataset(name: str) -> MultiSourceDatasetMerger:
    """Get dataset by name."""
    return _dataset_registry.get(name)


def list_datasets() -> list:
    """List all registered dataset names."""
    return _dataset_registry.list_all()


def clear_dataset_registry():
    """Clear the dataset registry (for testing)."""
    _dataset_registry.clear()
