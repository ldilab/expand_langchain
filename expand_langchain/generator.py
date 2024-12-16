import asyncio
import json
import logging
import os
from pathlib import Path
from traceback import format_exc
from typing import Any, List, Optional

import wandb
import yaml
from expand_langchain.config import Config
from expand_langchain.graph import CustomLangGraph
from expand_langchain.loader import Loader
from langchain_core.documents import Document
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

"""registry """
from expand_langchain.utils import registry  # isort:skip
from expand_langchain.chain import *
from expand_langchain.model import *
from expand_langchain.parser import *
from expand_langchain.prompt import *
from expand_langchain.transition import *


class Generator(BaseModel):
    verbose: bool = False
    debug: bool = False
    save_on: bool = True
    user_input_mode: bool = False

    api_keys_path: str = "api_keys.json"
    api_keys: Optional[dict] = None

    target_dataset_name: str = "target"
    example_dataset_name: str = "example"
    wandb_on: bool = False
    langfuse_on: bool = False
    rerun: bool = False
    max_concurrency: int = 5

    run_name: str = None  # if None, config_path.stem is used
    config_path: Path = None
    config: Config = None
    cache_root: Optional[Path] = None

    # private variables
    output_dir: Path = None
    result_root: Path = None
    datasets: dict = {}
    target_dataset: dict = {}
    example_dataset: dict = {}
    graph: CustomLangGraph = None

    # pydantic config
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        if self.verbose:
            logging.basicConfig(level=logging.INFO)

        self._load_config()
        self._init_result_dir()
        self._load_api_keys()
        self._load_datasets()
        self._init_wandb()
        self._compile_graph()

    def _load_config(self):
        if self.config_path is not None:
            self.config = Config(path=self.config_path)
        elif self.config is None:
            raise ValueError("Either config_path or config should be provided")
        else:
            pass

    def _init_result_dir(self):
        if self.run_name is None:
            self.run_name = self.config_path.stem
        if self.save_on:
            self.output_dir = Path(f"results/{self.run_name}")
            self.result_root = self.output_dir / "results"
            self.result_root.mkdir(parents=True, exist_ok=True)
        if not self.cache_root and not self.rerun:
            self.cache_root = self.result_root

    def _load_api_keys(self):
        if self.api_keys_path and Path(self.api_keys_path).exists():
            api_keys = json.loads(Path(self.api_keys_path).read_text())
            for k, v in api_keys.items():
                os.environ[k] = v

        if self.api_keys:
            for k, v in self.api_keys.items():
                os.environ[k] = v

        if not (self.api_keys_path or self.api_keys):
            logging.warning("No api keys are loaded")

    def _load_datasets(self):
        loader = Loader(config=self.config)
        self.datasets = loader.run().result

        if not self.user_input_mode:
            self.target_dataset = self.datasets.get(self.target_dataset_name, {})
            del self.datasets[self.target_dataset_name]

        self.example_dataset = self.datasets.get(self.example_dataset_name, {})
        if self.example_dataset_name in self.datasets:
            del self.datasets[self.example_dataset_name]

    def _init_wandb(self):
        wandb.require("core")

        mode = "disabled"
        if self.wandb_on:
            logging.info("Wandb mode is online")
            mode = "online"

        wandb.init(
            mode=mode,
            entity=os.environ.get("WANDB_ENTITY", None),
            project=os.environ.get("WANDB_PROJECT", None),
            name=self.run_name,
            notes=self.config.description,
        )

        wandb.config.update(self.config.model_dump())

    def _compile_graph(self):
        self.graph = CustomLangGraph(
            config=self.config.graph,
            examples=self.example_dataset,
            etc_datasets=self.datasets,
        ).compile()

    def run(
        self,
        n: Optional[int] = None,
        ids: Optional[list] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        targets = self.target_dataset

        if n is not None:
            targets = {k: v for k, v in list(targets.items())[:n]}
        elif start is not None and end is not None:
            targets = {k: v for k, v in list(targets.items())[start:end]}
        elif ids is not None:
            targets = {k: v for k, v in targets.items() if k in ids}
        else:
            pass

        asyncio.run(self._run(targets))

        return self

    async def _run(
        self,
        targets: dict,
    ):
        tasks = []
        sem = asyncio.Semaphore(self.max_concurrency)
        for id, target in targets.items():
            task = self._run_one(id, target, sem)
            tasks.append(task)

        await tqdm_asyncio.gather(*tasks)

    async def _run_one(
        self,
        id: str,
        target: dict,
        sem: asyncio.Semaphore,
    ):
        """
        Run the target and save the result as json file
        """
        id = str(id).replace("/", "_")

        async with sem:
            config = {
                "callbacks": [],
                "tags": [id],
                "metadata": {
                    "id": id,
                    "cache_root": self.cache_root,
                    "result_root": self.result_root,
                },
            }

            if self.langfuse_on:
                from langfuse.callback import CallbackHandler

                langfuse_handler = CallbackHandler()
                config["callbacks"].append(langfuse_handler)

            try:
                result = await self.graph.ainvoke(
                    [target],
                    config=config,
                )
                logging.info(f"Done: {id}")

            except Exception as e:
                logging.error(f"Error in running {id}")
                logging.error(format_exc())
                result = [{"error": format_exc()}]

                if self.debug:
                    raise e

            self._save_json(id, result)

        return result

    def _save_json(self, id: str, result: dict):
        """
        Save result as json file
        """
        path = self.result_root / f"{id}.json"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    return super().default(obj)
                except TypeError:
                    return "Not serializable"

        with open(self.result_root / f"{id}.json", "w") as f:
            json.dump(result, f, indent=4, cls=CustomEncoder, ensure_ascii=False)

    def merge_json(self):
        """
        Merge json files distributed by problem into one file
        """
        data = {}
        for file in os.listdir(f"{self.output_dir}/results"):
            if file.endswith(".json"):
                with open(f"{self.output_dir}/results/{file}", "r") as f:
                    id = file.split(".")[0]
                    data[id] = json.load(f)

        # sort by id and turn into list
        data = dict(sorted(data.items(), key=lambda x: x[0]))
        data = list(data.values())

        # max length of the result
        max_len = max([len(d) for d in data])

        dump_data = [{} for _ in range(len(data))]
        for i in range(max_len):
            for j, d in enumerate(data):
                if isinstance(d, list):
                    if i < len(d):
                        dump_data[j].update(d[i])
                    else:
                        dump_data[j].update({"max_depth": len(d)})
                elif isinstance(d, dict):
                    dump_data[j] = d
                else:
                    raise ValueError("Invalid data type")

            filename = f"{self.output_dir}/results_merged_{i}.json"
            with open(filename, "w") as f:
                json.dump(dump_data, f, indent=4, ensure_ascii=False)

        return self

    def exit(self):
        """
        Exit the generator
        """
        pass

    async def astream_user_input(
        self,
        nl_query: str,
        event_names: Optional[List[str]] = None,
    ):
        """
        Run user input
        """
        target = {"prompt": nl_query}

        gen = self.graph.astream_events(
            [target],
            version="v2",
            include_names=event_names,
        )
        async for result in gen:
            if result["event"] == "on_chain_end":
                yield result["data"]["output"]

    def run_user_input(
        self,
        inputs: List[dict],
    ):
        return self.graph.invoke(inputs)

    def lark_message(
        self,
        msg: str = "Done",
    ):
        """
        Send message to the webhook
        """
        webhook = os.environ.get("LARK_WEBHOOK", None)
        if webhook is None:
            return self

        import requests

        msg = f"{self.run_name}: {msg}"

        requests.post(
            webhook,
            json={"msg_type": "text", "content": {"text": msg}},
        )

        return self
