import fire

from expand_langchain.config_registry import get_config, list_configs, register_config
from expand_langchain.dataset_registry import (
    get_dataset,
    list_datasets,
    register_dataset,
)
from expand_langchain.generator import Generator
from expand_langchain.loader import (
    HuggingFaceSourceLoader,
    LocalSourceLoader,
    MultiSourceDatasetMerger,
)

if __name__ == "__main__":
    fire.Fire(
        {
            "get_config": get_config,
            "register_config": register_config,
            "list_configs": list_configs,
            "get_dataset": get_dataset,
            "register_dataset": register_dataset,
            "list_datasets": list_datasets,
            "dataset_merger": MultiSourceDatasetMerger,
            "local_loader": LocalSourceLoader,
            "huggingface_loader": HuggingFaceSourceLoader,
            "generator": Generator,
        }
    )
