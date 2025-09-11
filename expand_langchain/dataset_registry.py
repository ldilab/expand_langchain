"""Dataset registry for automatic dataset loading."""

import logging
from typing import Any, Dict

from .loader import MultiSourceDatasetMerger

# Global registry to store datasets
_DATASET_REGISTRY: Dict[str, MultiSourceDatasetMerger] = {}


def register_dataset(name: str, dataset: MultiSourceDatasetMerger):
    """Register a dataset with a name."""
    _DATASET_REGISTRY[name] = dataset
    logging.info(f"Registered dataset '{name}'")


def get_dataset(name: str) -> MultiSourceDatasetMerger:
    """Get dataset by name."""
    if name in _DATASET_REGISTRY:
        return _DATASET_REGISTRY[name]
    else:
        raise ValueError(f"Dataset '{name}' not found in registry")


def list_datasets() -> list:
    """List all registered dataset names."""
    return list(_DATASET_REGISTRY.keys())


def clear_dataset_registry():
    """Clear the dataset registry (for testing)."""
    global _DATASET_REGISTRY
    _DATASET_REGISTRY = {}
