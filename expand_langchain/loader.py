import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from pydantic import field_validator, model_validator
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SourceLoader(BaseModel):
    sort_key: str
    path: str

    # private
    data: Optional[Dataset] = None

    # pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def run(self):
        pass


class HuggingFaceSourceLoader(SourceLoader):
    dataset_name: str
    split: Optional[str] = None
    config_name: Optional[str] = None
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    streaming: bool = False
    num_proc: Optional[int] = None

    def run(self):
        """Load dataset from Hugging Face Hub"""
        try:
            dataset = load_dataset(
                self.dataset_name,
                name=self.config_name,
                split=self.split,
                revision=self.revision,
                cache_dir=self.cache_dir,
                streaming=self.streaming,
                num_proc=self.num_proc,
            )

            # Convert to regular Dataset if it's a streaming dataset
            if self.streaming:
                dataset = Dataset.from_generator(lambda: dataset)

            # Add sort key if not present
            if self.sort_key not in dataset.column_names:
                logger.warning(
                    f"Sort key '{self.sort_key}' not found in dataset '{self.dataset_name}'. Adding sequential index as sort key."
                )
                dataset = dataset.add_column(self.sort_key, list(range(len(dataset))))
            else:
                dataset = dataset.sort(self.sort_key)

            self.data = dataset
            logger.info(f"Loaded {len(dataset)} examples from {self.dataset_name}")

        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise


class LocalSourceLoader(SourceLoader):
    format: str = "json"  # json, csv, parquet, arrow, yaml

    def run(self):
        """Load dataset from local file"""
        try:
            if self.format == "json":
                dataset = Dataset.from_json(self.path)
            elif self.format == "csv":
                dataset = Dataset.from_csv(self.path)
            elif self.format == "parquet":
                dataset = Dataset.from_parquet(self.path)
            elif self.format == "arrow":
                dataset = Dataset.from_file(self.path)
            elif self.format == "yaml" or self.format == "yml":
                # Load YAML file and convert to dataset
                data_yaml = yaml.load(
                    Path(self.path).read_text(), Loader=yaml.FullLoader
                )
                dataset = Dataset.from_list(data_yaml)
            else:
                raise ValueError(f"Unsupported format: {self.format}")

            # Add sort key if not present
            if self.sort_key not in dataset.column_names:
                dataset = dataset.add_column(self.sort_key, list(range(len(dataset))))
            else:
                dataset = dataset.sort(self.sort_key)

            self.data = dataset
            logger.info(f"Loaded {len(dataset)} examples from {self.path}")

        except Exception as e:
            logger.error(f"Failed to load dataset from {self.path}: {e}")
            raise


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
                return cls(source=source, key=key)
            elif len(field_tuple) == 3:
                # (source, key, default_value)
                source, key, default_value = field_tuple
                return cls(source=source, key=key, default_value=default_value)
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

    fields: Dict[
        str,
        Union[
            MergerField,
            Tuple[str, str],
            Tuple[str, str, Any],
            Tuple[str, str, Any, Callable],
        ],
    ] = PydanticField(..., description="필드 정의 딕셔너리 (Field 객체 또는 튜플)")
    filter_func: Optional[Callable] = PydanticField(None, description="전역 필터 함수")
    max_samples: Optional[int] = None

    # private
    data: Optional[Dataset] = None
    lookup_tables: Dict[str, Dict[Any, Dict[str, Any]]] = PydanticField(
        default_factory=dict, exclude=True
    )

    # pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def convert_tuple_fields(self):
        """모델 검증 후 튜플 형태의 필드를 Field 객체로 변환"""
        converted_fields = {}

        for field_name, field_config in self.fields.items():
            if isinstance(field_config, tuple):
                try:
                    converted_fields[field_name] = self.MergerField.from_tuple(
                        field_config
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert field '{field_name}' from tuple: {e}"
                    )
            elif isinstance(field_config, self.MergerField):
                converted_fields[field_name] = field_config
            else:
                raise ValueError(
                    f"Field '{field_name}' must be MergerField object or tuple"
                )

        self.fields = converted_fields
        return self

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

        # 1. Load all source datasets
        logger.info("Loading source datasets...")
        for source_name, source_loader in tqdm(
            self.sources.items(), desc="Loading sources"
        ):
            source_loader.run()
            logger.info(
                f"Loaded source '{source_name}' with {len(source_loader.data)} samples"
            )

        # 2. Build lookup tables for fast access
        logger.info("Building lookup tables...")
        self._buildlookup_tables()

        # 3. Find common primary keys across all sources
        logger.info("Finding common primary keys...")
        primary_keys = self._find_common_primary_keys()
        logger.info(f"Found {len(primary_keys)} common primary keys")

        # 4. Build reconstructed dataset
        logger.info("Reconstructing dataset with custom fields...")
        reconstructed_data = []

        for pk in tqdm(primary_keys, desc="Reconstructing samples"):
            try:
                sample = self._build_sample(pk)
                if sample is not None:
                    reconstructed_data.append(sample)
            except Exception as e:
                logger.warning(f"Failed to build sample for primary key {pk}: {e}")
                continue

        # 5. Create final dataset
        if not reconstructed_data:
            logger.warning("No samples were successfully reconstructed")
            self.data = Dataset.from_list([])
        else:
            self.data = Dataset.from_list(reconstructed_data)

            # Apply global filter if provided
            if self.filter_func:
                logger.info("Applying global filter...")
                self.data = self.data.filter(self.filter_func)

            # Limit samples if specified
            if self.max_samples and len(self.data) > self.max_samples:
                logger.info(f"Limiting dataset to {self.max_samples} samples")
                self.data = self.data.select(range(self.max_samples))

        logger.info(
            f"Dataset reconstruction complete. Final dataset has {len(self.data)} samples"
        )
        return self

    def _buildlookup_tables(self):
        """각 소스별로 primary key를 인덱스로 하는 lookup table을 생성합니다"""
        self.lookup_tables = {}

        for source_name, source_loader in self.sources.items():
            if self.primary_key not in source_loader.data.column_names:
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

    def get_data(self) -> Dataset:
        """재구축된 데이터셋을 반환합니다"""
        if not hasattr(self, "data") or self.data is None:
            raise ValueError("Dataset not built yet. Call run() first.")
        return self.data

    def save(self, path: str, format: str = "json"):
        """재구축된 데이터셋을 파일로 저장합니다"""
        if not hasattr(self, "data") or self.data is None:
            raise ValueError("Dataset not built yet. Call run() first.")

        if format == "json":
            self.data.to_json(path)
        elif format == "csv":
            self.data.to_csv(path)
        elif format == "parquet":
            self.data.to_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Dataset saved to {path} in {format} format")


if __name__ == "__main__":
    """테스트 코드"""
    import json
    import os
    import tempfile

    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    print("=== DatasetBuilder 테스트 시작 ===")

    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:

        # 1. 테스트 데이터 생성
        print("\n1. 테스트 데이터 생성 중...")

        # JSON 데이터 생성
        json_data = [
            {"id": 1, "question": "What is diabetes?", "category": "medical"},
            {"id": 2, "question": "How to treat fever?", "category": "medical"},
            {"id": 3, "question": "What is Python?", "category": "programming"},
        ]
        json_path = os.path.join(temp_dir, "questions.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        # YAML 데이터 생성
        yaml_data = [
            {"id": 1, "answer": "A metabolic disorder", "difficulty": 3},
            {"id": 2, "answer": "Use fever reducers", "difficulty": 2},
            {"id": 3, "answer": "A programming language", "difficulty": 1},
        ]
        yaml_path = os.path.join(temp_dir, "answers.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        # CSV 데이터 생성
        csv_data = "id,author,source\n1,Dr. Smith,Medical Journal\n2,Dr. Jones,Health Guide\n3,John Doe,Tech Blog\n"
        csv_path = os.path.join(temp_dir, "metadata.csv")
        with open(csv_path, "w") as f:
            f.write(csv_data)

        print(f"테스트 데이터 생성 완료:")
        print(f"  - JSON: {json_path}")
        print(f"  - YAML: {yaml_path}")
        print(f"  - CSV: {csv_path}")

        # 2. 소스 로더 생성 및 테스트
        print("\n2. 소스 로더 테스트 중...")

        # JSON 로더 테스트
        json_loader = LocalSourceLoader(sort_key="id", path=json_path, format="json")
        json_loader.run()
        print(f"JSON 로더: {len(json_loader.data)} 샘플 로드됨")

        # YAML 로더 테스트
        yaml_loader = LocalSourceLoader(sort_key="id", path=yaml_path, format="yaml")
        yaml_loader.run()
        print(f"YAML 로더: {len(yaml_loader.data)} 샘플 로드됨")

        # CSV 로더 테스트
        csv_loader = LocalSourceLoader(sort_key="id", path=csv_path, format="csv")
        csv_loader.run()
        print(f"CSV 로더: {len(csv_loader.data)} 샘플 로드됨")

        # 3. 변환 함수 정의
        def text_length(text):
            return len(str(text)) if text else 0

        def categorize_difficulty(diff):
            if diff <= 1:
                return "easy"
            elif diff <= 2:
                return "medium"
            else:
                return "hard"

        # 4. DatasetBuilder 테스트
        print("\n3. DatasetBuilder 테스트 중...")

        sources = {
            "questions": json_loader,
            "answers": yaml_loader,
            "metadata": csv_loader,
        }

        fields = {
            "question": MultiSourceDatasetMerger.MergerField(
                source="questions", key="question", default_value=""
            ),
            "answer": MultiSourceDatasetMerger.MergerField(
                source="answers", key="answer", default_value=""
            ),
            "question_length": MultiSourceDatasetMerger.MergerField(
                source="questions",
                key="question",
                transform=text_length,
                default_value=0,
            ),
            "difficulty_category": MultiSourceDatasetMerger.MergerField(
                source="answers",
                key="difficulty",
                transform=categorize_difficulty,
                default_value="medium",
            ),
            "author": MultiSourceDatasetMerger.MergerField(
                source="metadata", key="author", default_value="Unknown"
            ),
            "category": MultiSourceDatasetMerger.MergerField(
                source="questions", key="category", default_value="general"
            ),
        }

        # 의료 관련 필터
        def is_medical(sample):
            return sample.get("category") == "medical"

        builder = MultiSourceDatasetMerger(
            sources=sources,
            primary_key="id",
            fields=fields,
            filter_func=is_medical,  # 의료 관련 내용만 필터링
            max_samples=10,
        )

        # 데이터셋 재구축 실행
        result = builder.run()

        # 5. 결과 확인
        print("\n4. 재구축 결과:")
        reconstructed_dataset = builder.get_data()
        print(f"  - 총 샘플 수: {len(reconstructed_dataset)}")
        print(f"  - 컬럼: {reconstructed_dataset.column_names}")

        if len(reconstructed_dataset) > 0:
            print("\n5. 첫 번째 샘플:")
            first_sample = reconstructed_dataset[0]
            for key, value in first_sample.items():
                print(f"  - {key}: {value}")

        # 6. 데이터셋 저장 테스트
        print("\n6. 데이터셋 저장 테스트...")
        output_path = os.path.join(temp_dir, "reconstructed_dataset.json")
        builder.save(output_path, format="json")

        # 저장된 파일 확인
        if os.path.exists(output_path):
            # Hugging Face datasets의 to_json은 JSONL 형식으로 저장함
            saved_data = []
            with open(output_path, "r") as f:
                for line in f:
                    if line.strip():
                        saved_data.append(json.loads(line))
            print(f"  - 저장 완료: {len(saved_data)} 샘플")

        print("\n=== 모든 테스트 완료 ===")

        # 7. 예외 상황 테스트
        print("\n7. 예외 상황 테스트...")

        try:
            # 존재하지 않는 파일로 테스트
            invalid_loader = LocalSourceLoader(
                sort_key="id", path="/nonexistent/file.json", format="json"
            )
            invalid_loader.run()
        except Exception as e:
            print(f"  - 예상된 오류 처리됨: {type(e).__name__}")

        try:
            # 잘못된 형식으로 테스트
            invalid_format_loader = LocalSourceLoader(
                sort_key="id", path=json_path, format="invalid_format"
            )
            invalid_format_loader.run()
        except Exception as e:
            print(f"  - 예상된 오류 처리됨: {type(e).__name__}")

        print("  - 예외 처리 테스트 완료")

        # 8. 튜플 형태 필드 정의 테스트
        print("\n8. 튜플 형태 필드 정의 테스트...")

        # 테스트용 쿼리 데이터 생성
        queries_data = [
            {
                "task_id": "T001",
                "difficulty": 3,
                "patient_id": "P001",
                "department": "cardiology",
                "question": "Find all patients with heart disease",
                "sql_answer-gt": "SELECT * FROM patients WHERE diagnosis = 'heart disease'",
                "nl_answer-gt": "Patients with heart disease diagnosis",
            },
            {
                "task_id": "T002",
                "difficulty": 2,
                "patient_id": "P002",
                "department": "neurology",
                "question": "List neurological conditions",
                "sql_answer-gt": "SELECT * FROM conditions WHERE type = 'neurological'",
                "nl_answer-gt": "All neurological conditions",
            },
        ]

        queries_path = os.path.join(temp_dir, "queries.json")
        with open(queries_path, "w") as f:
            json.dump(queries_data, f)

        # 쿼리 로더 생성
        queries_loader = LocalSourceLoader(
            sort_key="task_id", path=queries_path, format="json"
        )

        # 튜플 형태로 필드 정의 (사용자가 제공한 형태)
        tuple_fields = {
            "task_id": ("queries", "task_id"),
            "difficulty": ("queries", "difficulty"),
            "patient_id": ("queries", "patient_id"),
            "department": ("queries", "department"),
            "question": ("queries", "question"),
            "sql_answer-gt": ("queries", "sql_answer-gt"),
            "nl_answer-gt": ("queries", "nl_answer-gt"),
        }

        try:
            tuple_builder = MultiSourceDatasetMerger(
                sources={"queries": queries_loader},
                primary_key="task_id",
                fields=tuple_fields,
            )

            print("  - 튜플 필드 정의 성공적으로 파싱됨")

            # 데이터셋 재구축 실행
            tuple_result = tuple_builder.run()
            tuple_dataset = tuple_builder.get_data()

            print(f"  - 튜플 기반 재구축 완료: {len(tuple_dataset)} 샘플")
            print(f"  - 컬럼: {tuple_dataset.column_names}")

            if len(tuple_dataset) > 0:
                print("  - 첫 번째 샘플:")
                first_sample = tuple_dataset[0]
                for key, value in first_sample.items():
                    if isinstance(value, str) and len(value) > 50:
                        print(f"    {key}: {value[:50]}...")
                    else:
                        print(f"    {key}: {value}")

        except Exception as e:
            print(f"  - 튜플 테스트 실패: {e}")
            import traceback

            traceback.print_exc()

        # 9. 다양한 튜플 형태 테스트
        print("\n9. 다양한 튜플 형태 테스트...")

        # 2-4개 요소를 가진 다양한 튜플 테스트
        def upper_transform(text):
            return str(text).upper()

        mixed_fields = {
            # 2개 요소: (source, key)
            "simple_field": ("queries", "department"),
            # 3개 요소: (source, key, default_value)
            "with_default": ("queries", "missing_field", "DEFAULT_VALUE"),
            # 4개 요소: (source, key, default_value, transform)
            "with_transform": ("queries", "department", "unknown", upper_transform),
        }

        try:
            mixed_builder = MultiSourceDatasetMerger(
                sources={"queries": queries_loader},
                primary_key="task_id",
                fields=mixed_fields,
            )

            print("  - 다양한 튜플 형태 파싱 성공")

            mixed_result = mixed_builder.run()
            mixed_dataset = mixed_builder.get_data()

            print(f"  - 혼합 필드 재구축 완료: {len(mixed_dataset)} 샘플")

            if len(mixed_dataset) > 0:
                print("  - 변환 함수 적용 결과:")
                first_sample = mixed_dataset[0]
                print(f"    원본 department: {queries_data[0]['department']}")
                print(
                    f"    변환된 with_transform: {first_sample.get('with_transform')}"
                )
                print(f"    기본값 with_default: {first_sample.get('with_default')}")

        except Exception as e:
            print(f"  - 혼합 튜플 테스트 실패: {e}")

        print("\n  - 튜플 필드 정의 테스트 완료")

    print("\n=== 전체 테스트 완료 ===")
