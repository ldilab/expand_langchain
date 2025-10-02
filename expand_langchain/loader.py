import hashlib
import json
import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from pydantic import field_validator, model_validator
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SourceLoader(BaseModel):
    sort_key: str
    path: str

    # private
    data: Optional[List[Dict[str, Any]]] = None

    # pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def run(self) -> "SourceLoader":
        pass


class HuggingFaceSourceLoader(SourceLoader):
    split: Optional[str] = None
    config_name: Optional[str] = None
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    streaming: bool = False
    num_proc: Optional[int] = None

    def run(self):
        """Load data from Hugging Face dataset"""
        try:
            from datasets import Dataset, load_dataset  # type: ignore
        except ImportError:
            raise ImportError(
                "datasets library is required for HuggingFaceSourceLoader. Install with: pip install datasets"
            )

        if self.config_name:
            dataset = load_dataset(
                self.path,
                name=self.config_name,
                split=self.split,
            )
        else:
            dataset = load_dataset(
                self.path,
                split=self.split,
            )

        # Convert to generator if needed
        if hasattr(dataset, "to_iterable_dataset"):
            if self.streaming:
                dataset = Dataset.from_generator(lambda: dataset)

        self.data = list(dataset)
        logger.info(
            f"Loaded {len(self.data)} samples from Hugging Face dataset {self.path}"
        )
        return self


class LocalSourceLoader(SourceLoader):
    format: str = "json"  # json, jsonl, csv, parquet, arrow, yaml

    def run(self):
        """Load dataset from local file"""
        try:
            if self.format == "json":
                import json

                with open(self.path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    data_list = json.loads(content)
                    if not isinstance(data_list, list):
                        # Single object case
                        data_list = [data_list]

            elif self.format == "jsonl":
                # Load JSONL file (JSON Lines format)
                import json

                data_list = []
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            data_list.append(json.loads(line))
            elif self.format == "csv":
                import pandas as pd

                df = pd.read_csv(self.path)
                data_list = df.to_dict("records")
            elif self.format == "parquet":
                import pandas as pd

                df = pd.read_parquet(self.path)
                data_list = df.to_dict("records")
            elif self.format == "arrow":
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    table = pq.read_table(self.path)
                    data_list = table.to_pylist()
                except ImportError:
                    raise ImportError(
                        "pyarrow is required for arrow format. Install with: pip install pyarrow"
                    )
            elif self.format == "yaml" or self.format == "yml":
                # Load YAML file and convert to dataset
                data_yaml = yaml.load(
                    Path(self.path).read_text(), Loader=yaml.FullLoader
                )
                data_list = data_yaml if isinstance(data_yaml, list) else [data_yaml]
            else:
                raise ValueError(f"Unsupported format: {self.format}")

            # Add sort key if not present and sort
            column_names = set()
            if data_list:
                column_names = set(data_list[0].keys())

            if self.sort_key not in column_names:
                for i, item in enumerate(data_list):
                    item[self.sort_key] = i

            # Sort by sort_key
            try:
                data_list.sort(key=lambda x: x.get(self.sort_key, 0))
            except Exception as e:
                logger.warning(f"Failed to sort by '{self.sort_key}': {e}")

            self.data = cast(List[Dict[str, Any]], data_list)
            logger.info(f"Loaded {len(data_list)} examples from {self.path}")

        except Exception as e:
            logger.error(f"Failed to load dataset from {self.path}: {e}")
            raise

        return self


class MultiSourceDatasetMerger(BaseModel):
    sources: Dict[str, SourceLoader]
    primary_key: str = PydanticField(
        ..., description="모든 소스에서 공통으로 사용할 기본 키"
    )

    class MergerField(BaseModel):
        """데이터셋 필드 정의 클래스"""

        source: str = PydanticField(..., description="소스 데이터셋 이름")
        key: str = PydanticField(..., description="소스에서 추출할 키")
        transform: Optional[Callable] = PydanticField(None, description="값 변환 함수")
        default_value: Any = PydanticField(None, description="기본값")

        # pydantic
        model_config = ConfigDict(arbitrary_types_allowed=True)

        @classmethod
        def from_tuple(
            cls, field_tuple: Tuple
        ) -> "MultiSourceDatasetMerger.MergerField":
            """튜플로부터 Field 객체 생성"""
            if not isinstance(field_tuple, tuple):
                raise ValueError(f"Expected tuple, got {type(field_tuple)}")

            if len(field_tuple) == 2:
                # (source, key)
                source, key = field_tuple
                return cls(source=source, key=key, transform=None, default_value=None)
            elif len(field_tuple) == 3:
                # (source, key, default_value)
                source, key, default_value = field_tuple
                return cls(
                    source=source, key=key, transform=None, default_value=default_value
                )
            elif len(field_tuple) == 4:
                # (source, key, default_value, transform)
                source, key, default_value, transform = field_tuple
                return cls(
                    source=source,
                    key=key,
                    default_value=default_value,
                    transform=transform,
                )
            else:
                raise ValueError(
                    f"Tuple must have 2-4 elements, got {len(field_tuple)}"
                )

        def __str__(self):
            return f"Field(source='{self.source}', key='{self.key}', default={self.default_value})"

    fields: Dict[str, MergerField] = PydanticField(
        ..., description="필드 정의 딕셔너리"
    )
    filter_func: Optional[Callable] = PydanticField(
        default=None, description="전역 필터 함수"
    )
    max_samples: Optional[int] = None

    # private
    data_cache_dir: Optional[str] = None
    data_file_path: Optional[str] = None
    total_samples: int = 0
    lookup_tables: Dict[str, Dict[Any, Dict[str, Any]]] = PydanticField(
        default_factory=dict, exclude=True
    )

    # pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create_with_tuple_fields(
        cls,
        sources: Dict[str, SourceLoader],
        primary_key: str,
        fields: Dict[
            str,
            Union[
                "MultiSourceDatasetMerger.MergerField",
                Tuple[str, str],
                Tuple[str, str, Any],
                Tuple[str, str, Any, Callable],
            ],
        ],
        **kwargs,
    ):
        """tuple 필드를 지원하는 생성자"""
        converted_fields = {}

        for field_name, field_config in fields.items():
            if isinstance(field_config, tuple):
                try:
                    converted_fields[field_name] = cls.MergerField.from_tuple(
                        field_config
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert field '{field_name}' from tuple: {e}"
                    )
            elif isinstance(field_config, cls.MergerField):
                converted_fields[field_name] = field_config
            else:
                raise ValueError(
                    f"Field '{field_name}' must be MergerField object or tuple"
                )

        return cls(
            sources=sources, primary_key=primary_key, fields=converted_fields, **kwargs
        )

    @model_validator(mode="after")
    def validate_field_sources(self):
        """필드의 소스가 실제 소스에 존재하는지 검증"""
        source_names = set(self.sources.keys())

        for field_name, field_config in self.fields.items():
            if field_config.source not in source_names:
                raise ValueError(
                    f"Field '{field_name}' references unknown source '{field_config.source}'. "
                    f"Available sources: {list(source_names)}"
                )

        return self

    def run(self):
        """
        여러 소스의 데이터셋을 사용자 정의 필드로 재구축하는 메인 메서드
        """
        logger.info("Starting dataset reconstruction...")

        # Setup cache directory
        self._setup_cache_directory()

        # 1. Load all source datasets
        logger.info("Loading source datasets...")
        for source_name, source_loader in tqdm(
            self.sources.items(), desc="Loading sources"
        ):
            source_loader.run()
            logger.info(
                f"Loaded source '{source_name}' with {len(source_loader.data or [])} samples"
            )

        # 2. Build lookup tables for fast access
        logger.info("Building lookup tables...")
        self._buildlookup_tables()

        # 3. Find common primary keys across all sources
        logger.info("Finding common primary keys...")
        primary_keys = self._find_common_primary_keys()
        logger.info(f"Found {len(primary_keys)} common primary keys")

        # 4. Build reconstructed dataset and save to disk
        logger.info("Reconstructing dataset with custom fields...")
        self._build_and_save_dataset(primary_keys)

        logger.info(
            f"Dataset reconstruction complete. Final dataset has {self.total_samples} samples"
        )
        return self

    def _setup_cache_directory(self):
        """캐시 디렉토리 설정"""
        # Create cache directory based on configuration hash
        config_hash = self._get_config_hash()
        self.data_cache_dir = f"./data/merged_datasets/{config_hash}"
        os.makedirs(self.data_cache_dir, exist_ok=True)
        self.data_file_path = os.path.join(self.data_cache_dir, "data.jsonl")

        logger.info(f"Using cache directory: {self.data_cache_dir}")

    def _get_config_hash(self) -> str:
        """설정 기반 해시 생성"""
        config_str = json.dumps(
            {
                "sources": {
                    name: {
                        "path": loader.path,
                        "format": getattr(loader, "format", "unknown"),
                    }
                    for name, loader in self.sources.items()
                },
                "primary_key": self.primary_key,
                "fields": {name: str(field) for name, field in self.fields.items()},
                "max_samples": self.max_samples,
            },
            sort_keys=True,
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def _build_and_save_dataset(self, primary_keys: List[Any]):
        """데이터셋을 재구축하고 디스크에 저장"""
        if not self.data_file_path:
            raise ValueError(
                "Data file path not set. Call _setup_cache_directory first."
            )

        processed_count = 0
        saved_count = 0

        with open(self.data_file_path, "w", encoding="utf-8") as f:
            for pk in tqdm(primary_keys, desc="Reconstructing samples"):
                try:
                    sample = self._build_sample(pk)
                    if sample is not None:
                        # Apply filter if provided
                        if self.filter_func and not self.filter_func(sample):
                            continue

                        # Check max samples limit
                        if self.max_samples and saved_count >= self.max_samples:
                            break

                        # Save to disk
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        saved_count += 1

                    processed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to build sample for primary key {pk}: {e}")
                    continue

        self.total_samples = saved_count
        logger.info(
            f"Processed {processed_count} samples, saved {saved_count} samples to disk"
        )

    def _buildlookup_tables(self):
        """각 소스별로 primary key를 인덱스로 하는 lookup table을 생성합니다"""
        self.lookup_tables = {}

        for source_name, source_loader in self.sources.items():
            if not source_loader.data:
                logger.warning(f"No data found in source '{source_name}'")
                continue

            # Check if primary key exists in the data
            if source_loader.data and self.primary_key not in source_loader.data[0]:
                logger.warning(
                    f"Primary key '{self.primary_key}' not found in source '{source_name}'"
                )
                continue

            lookup_table = {}
            for row in source_loader.data:
                primary_key_value = row[self.primary_key]
                lookup_table[primary_key_value] = row

            self.lookup_tables[source_name] = lookup_table
            logger.debug(
                f"Built lookup table for source '{source_name}' with {len(lookup_table)} entries"
            )

    def _find_common_primary_keys(self) -> List[Any]:
        """모든 소스에서 공통된 primary key를 찾습니다"""
        if not self.lookup_tables:
            return []

        # Get primary keys from lookup tables (much faster than accessing datasets)
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

    def _build_sample(self, primary_key_value: Any) -> Optional[Dict[str, Any]]:
        """특정 primary key 값에 대해 모든 필드를 재구축합니다"""
        sample = {self.primary_key: primary_key_value}

        for field_name, field_config in self.fields.items():
            try:
                value = self._extract_field_value(primary_key_value, field_config)
                sample[field_name] = value
            except Exception as e:
                logger.debug(
                    f"Failed to extract field '{field_name}' for key {primary_key_value}: {e}"
                )
                # Use default value if extraction fails
                if field_config.default_value is not None:
                    sample[field_name] = field_config.default_value
                else:
                    sample[field_name] = None

        return sample

    def _extract_field_value(
        self, primary_key_value: Any, field_config: MergerField
    ) -> Any:
        """특정 필드 설정에 따라 값을 추출합니다"""
        source_name = field_config.source

        if source_name not in self.sources:
            raise ValueError(f"Source '{source_name}' not found")

        # Use lookup table for fast access
        if source_name not in self.lookup_tables:
            raise ValueError(f"Lookup table for source '{source_name}' not found")

        lookup_table = self.lookup_tables[source_name]

        if primary_key_value not in lookup_table:
            raise ValueError(
                f"No matching row found for primary key {primary_key_value}"
            )

        # Get the row directly from lookup table
        row = lookup_table[primary_key_value]

        # Extract the specified field
        if field_config.key not in row:
            raise ValueError(
                f"Key '{field_config.key}' not found in source '{source_name}'"
            )

        value = row[field_config.key]

        # Apply transformation if provided
        if field_config.transform:
            value = field_config.transform(value)

        return value

    def get_data(self) -> List[Dict[str, Any]]:
        """재구축된 데이터셋을 반환합니다 (메모리에 로드)"""
        if not self.data_file_path or not os.path.exists(self.data_file_path):
            raise ValueError("Dataset not built yet. Call run() first.")

        data = []
        with open(self.data_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def iter_data(self):
        """데이터를 하나씩 반복하여 반환 (메모리 효율적)"""
        if not self.data_file_path or not os.path.exists(self.data_file_path):
            raise ValueError("Dataset not built yet. Call run() first.")

        with open(self.data_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def get_sample(self, index: int) -> Dict[str, Any]:
        """특정 인덱스의 샘플을 반환"""
        if not self.data_file_path or not os.path.exists(self.data_file_path):
            raise ValueError("Dataset not built yet. Call run() first.")

        if index < 0 or index >= self.total_samples:
            raise IndexError(f"Index {index} out of range [0, {self.total_samples})")

        with open(self.data_file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == index:
                    return json.loads(line.strip())

        raise IndexError(f"Could not find sample at index {index}")

    def get_batch(self, start: int, size: int) -> List[Dict[str, Any]]:
        """배치 단위로 데이터를 반환"""
        if not self.data_file_path or not os.path.exists(self.data_file_path):
            raise ValueError("Dataset not built yet. Call run() first.")

        if start < 0 or start >= self.total_samples:
            raise IndexError(
                f"Start index {start} out of range [0, {self.total_samples})"
            )

        batch = []
        with open(self.data_file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= start and len(batch) < size:
                    batch.append(json.loads(line.strip()))
                elif len(batch) >= size:
                    break

        return batch

    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return self.total_samples

    def save(self, path: str, format: str = "json"):
        """재구축된 데이터셋을 파일로 저장합니다"""
        if not self.data_file_path or not os.path.exists(self.data_file_path):
            raise ValueError("Dataset not built yet. Call run() first.")

        if format == "json":
            # Save as JSON array
            data = self.get_data()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == "jsonl":
            # Copy JSONL file
            import shutil

            shutil.copy2(self.data_file_path, path)
        elif format == "csv":
            # Use pandas for CSV export
            import pandas as pd

            data = self.get_data()
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
        elif format == "parquet":
            # Use pandas for Parquet export
            import pandas as pd

            data = self.get_data()
            df = pd.DataFrame(data)
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Dataset saved to {path} in {format} format")
