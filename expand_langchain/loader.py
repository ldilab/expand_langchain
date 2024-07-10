import json
import logging
from pathlib import Path

from datasets import Dataset, load_dataset
from expand_langchain.config import Config, DatasetConfig, SourceConfig
from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Loader(BaseModel):
    config: Config = None
    path: str = None

    result: dict = None

    def __init__(self, **data):
        super().__init__(**data)

        if self.path is not None:
            self.config = Config(path=self.path)
        elif self.config is None:
            raise ValueError("Either config_path or config should be provided")
        else:
            pass

    def run(self):
        sources = self.load_sources()
        self.result = self.load_datasets(sources)

        return self

    def load_sources(self):
        sources = {}
        for source in self.config.source:
            source: SourceConfig
            if source.type == "huggingface":
                name = source.name
                path = source.path
                sort_key = source.sort_key
                split = source.kwargs.get("split")

                logger.info(f"loading source: {name} from {path}")
                sources[name] = load_dataset(path)[split].sort(sort_key)
            elif source.type == "json":
                name = source.name
                path = source.path
                sort_key = source.sort_key

                logger.info(f"loading source: {name} from {path}")
                sources[name] = Dataset.from_list(
                    json.loads(Path(path).read_text())
                ).sort(sort_key)
            else:
                raise ValueError(f"Unsupported source type: {source.type}")

        return sources

    def load_datasets(self, sources):
        datasets = {}
        for dataset in self.config.dataset:
            dataset: DatasetConfig

            name = dataset.name
            primary_key = dataset.primary_key
            fields = dataset.fields

            logger.info(f"loading dataset: {name}")
            datasets[name] = {}
            primary_field = list(filter(lambda x: x.name == primary_key, fields))[0]
            ids = sources[primary_field.source][primary_field.key]
            for i, id in tqdm(enumerate(ids)):
                id = id.replace("/", "_")
                datasets[name][id] = {}
                for field in fields:
                    source = sources[field.source]
                    datasets[name][id][field.name] = source[i][field.key]

        return datasets

    def print_json(self):
        print(json.dumps(self.result, indent=4, ensure_ascii=False))

    def save_json(self, output_path):
        output_path = Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)

        try:
            output_path.write_text(
                json.dumps(self.result, indent=4, ensure_ascii=False)
            )
            print("Output file saved successfully")
        except:
            print("Failed to save the output file")
