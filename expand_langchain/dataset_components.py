"""
Components for dataset merging functionality.
This module breaks down the large MultiSourceDatasetMerger into smaller, focused classes.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Handles caching operations for dataset merging."""

    def __init__(self, base_cache_dir: str = "./data/merged_datasets"):
        self.base_cache_dir = base_cache_dir
        self.cache_dir: Optional[str] = None
        self.data_file_path: Optional[str] = None

    def setup_cache_directory(self, config_hash: str) -> None:
        """Setup cache directory based on configuration hash."""
        self.cache_dir = os.path.join(self.base_cache_dir, config_hash)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.data_file_path = os.path.join(self.cache_dir, "data.jsonl")
        logger.info(f"Using cache directory: {self.cache_dir}")

    def generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash from configuration for cache directory naming."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def cache_exists(self) -> bool:
        """Check if cache file exists."""
        return self.data_file_path and os.path.exists(self.data_file_path)

    def get_cache_path(self) -> Optional[str]:
        """Get the cache file path."""
        return self.data_file_path


class DatasetValidator:
    """Validates dataset configurations and data integrity."""

    @staticmethod
    def validate_field_sources(fields: Dict[str, Any], sources: Dict[str, Any]) -> None:
        """Validate that field sources exist in the available sources."""
        source_names = set(sources.keys())

        for field_name, field_config in fields.items():
            if (
                hasattr(field_config, "source")
                and field_config.source not in source_names
            ):
                raise ValueError(
                    f"Field '{field_name}' references unknown source '{field_config.source}'. "
                    f"Available sources: {list(source_names)}"
                )

    @staticmethod
    def validate_primary_key_exists(
        data: List[Dict[str, Any]], primary_key: str, source_name: str
    ) -> bool:
        """Validate that primary key exists in the data."""
        if not data:
            logger.warning(f"No data found in source '{source_name}'")
            return False

        if primary_key not in data[0]:
            logger.warning(
                f"Primary key '{primary_key}' not found in source '{source_name}'"
            )
            return False

        return True


class LookupTableBuilder:
    """Builds and manages lookup tables for fast data access."""

    def __init__(self):
        self.lookup_tables: Dict[str, Dict[Any, Dict[str, Any]]] = {}

    def build_lookup_tables(self, sources: Dict[str, Any], primary_key: str) -> None:
        """Build lookup tables for all sources."""
        self.lookup_tables = {}

        for source_name, source_loader in sources.items():
            if not source_loader.data:
                logger.warning(f"No data found in source '{source_name}'")
                continue

            if not DatasetValidator.validate_primary_key_exists(
                source_loader.data, primary_key, source_name
            ):
                continue

            lookup_table = {}
            for row in source_loader.data:
                primary_key_value = row[primary_key]
                lookup_table[primary_key_value] = row

            self.lookup_tables[source_name] = lookup_table
            logger.debug(
                f"Built lookup table for source '{source_name}' with {len(lookup_table)} entries"
            )

    def find_common_primary_keys(self) -> List[Any]:
        """Find primary keys that exist in all sources."""
        if not self.lookup_tables:
            return []

        source_names = list(self.lookup_tables.keys())
        if not source_names:
            return []

        # Start with keys from first source
        common_keys = set(self.lookup_tables[source_names[0]].keys())

        # Find intersection with other sources
        for source_name in source_names[1:]:
            source_keys = set(self.lookup_tables[source_name].keys())
            common_keys = common_keys.intersection(source_keys)

        return sorted(list(common_keys))

    def get_lookup_table(self, source_name: str) -> Optional[Dict[Any, Dict[str, Any]]]:
        """Get lookup table for a specific source."""
        return self.lookup_tables.get(source_name)


class FieldExtractor:
    """Handles field extraction and transformation logic."""

    def __init__(self, lookup_table_builder: LookupTableBuilder):
        self.lookup_builder = lookup_table_builder

    def extract_field_value(
        self, primary_key_value: Any, field_config: Any, sources: Dict[str, Any]
    ) -> Any:
        """Extract field value based on field configuration."""
        source_name = field_config.source

        if source_name not in sources:
            raise ValueError(f"Source '{source_name}' not found")

        lookup_table = self.lookup_builder.get_lookup_table(source_name)
        if lookup_table is None:
            raise ValueError(f"Lookup table for source '{source_name}' not found")

        if primary_key_value not in lookup_table:
            raise ValueError(
                f"No matching row found for primary key {primary_key_value}"
            )

        row = lookup_table[primary_key_value]

        if field_config.key not in row:
            raise ValueError(
                f"Key '{field_config.key}' not found in source '{source_name}'"
            )

        value = row[field_config.key]

        # Apply transformation if provided
        if hasattr(field_config, "transform") and field_config.transform:
            value = field_config.transform(value)

        return value

    def build_sample(
        self,
        primary_key_value: Any,
        primary_key: str,
        fields: Dict[str, Any],
        sources: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build a complete sample for the given primary key."""
        sample = {primary_key: primary_key_value}

        for field_name, field_config in fields.items():
            try:
                value = self.extract_field_value(
                    primary_key_value, field_config, sources
                )
                sample[field_name] = value
            except Exception as e:
                logger.debug(
                    f"Failed to extract field '{field_name}' for key {primary_key_value}: {e}"
                )

                # Use default value if extraction fails
                if (
                    hasattr(field_config, "default_value")
                    and field_config.default_value is not None
                ):
                    sample[field_name] = field_config.default_value
                else:
                    sample[field_name] = None

        return sample


class DatasetWriter:
    """Handles writing dataset to disk."""

    @staticmethod
    def write_dataset(
        cache_manager: CacheManager,
        primary_keys: List[Any],
        field_extractor: FieldExtractor,
        primary_key: str,
        fields: Dict[str, Any],
        sources: Dict[str, Any],
        filter_func: Optional[Any] = None,
        max_samples: Optional[int] = None,
    ) -> int:
        """Write dataset to disk and return number of saved samples."""
        data_file_path = cache_manager.get_cache_path()
        if not data_file_path:
            raise ValueError("Cache manager not properly initialized")

        processed_count = 0
        saved_count = 0

        with open(data_file_path, "w", encoding="utf-8") as f:
            for pk in primary_keys:
                try:
                    sample = field_extractor.build_sample(
                        pk, primary_key, fields, sources
                    )
                    if sample is not None:
                        # Apply filter if provided
                        if filter_func and not filter_func(sample):
                            continue

                        # Check max samples limit
                        if max_samples and saved_count >= max_samples:
                            break

                        # Save to disk
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        saved_count += 1

                    processed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to build sample for primary key {pk}: {e}")
                    continue

        logger.info(
            f"Processed {processed_count} samples, saved {saved_count} samples to disk"
        )
        return saved_count
