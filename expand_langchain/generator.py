import asyncio
import json
import logging
import os
from pathlib import Path
from traceback import format_exc
from typing import Optional

from expand_langchain.config import Config
from expand_langchain.graph import Graph
from expand_langchain.loader import Loader
from langsmith import Client, trace
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

import wandb

"""registry """
from expand_langchain.utils import registry  # isort:skip
from expand_langchain.chain import *
from expand_langchain.model import *
from expand_langchain.parser import *
from expand_langchain.prompt import *
from expand_langchain.transition import *


class Generator(BaseModel):
    verbose: bool = False
    api_keys_path: str = "api_keys.json"
    target_dataset_name: str = "target"
    example_dataset_name: str = "example"
    wandb_mode: str = "offline"  # "online", "offline", "disabled"
    langsmith_mode: str = "disabled"  # "online", "disabled"
    rerun: bool = False

    run_name: str = None  # if None, config_path.stem is used
    config_path: Path = None
    config: Config = None

    # private variables
    output_dir: Path = None
    results_dir: Path = None
    datasets: dict = {}
    graph: Graph = None

    def __init__(self, **data):
        super().__init__(**data)

        self._load_config()
        self._init_result_dir()
        self._load_api_keys()
        self._load_datasets()
        self._init_wandb()
        self._init_langsmith()
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
        self.output_dir = Path(f"results/{self.run_name}")
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_api_keys(self):
        api_keys = json.loads(Path(self.api_keys_path).read_text())
        for k, v in api_keys.items():
            os.environ[k] = v

    def _load_datasets(self):
        loader = Loader(config=self.config)
        self.datasets = loader.run().result

    def _init_wandb(self):
        wandb.require("core")

        if self.wandb_mode == "online":
            logging.info("Wandb mode is online")

        wandb.init(
            mode=self.wandb_mode,
            entity=os.environ.get("WANDB_ENTITY", None),
            project=os.environ.get("WANDB_PROJECT", None),
            name=self.config_path.stem,
            notes=self.config.description,
        )

        wandb.config.update(self.config.model_dump())

    def _init_langsmith(self):
        if self.langsmith_mode == "online":
            logging.info("Langsmith mode is online")
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.run_name
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = ""
            os.environ["LANGCHAIN_PROJECT"] = ""

    def _compile_graph(self):
        self.graph = Graph(
            config=self.config.graph,
            examples=self.datasets.get(self.example_dataset_name, {}),
        ).run()

    def run(
        self,
        n: Optional[int] = None,
        ids: Optional[list] = None,
    ):
        targets = self.datasets[self.target_dataset_name]

        if n is not None:
            targets = {k: v for k, v in list(targets.items())[:n]}
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
        for id, target in targets.items():
            task = self._run_one(id, target)
            tasks.append(task)

        await tqdm_asyncio.gather(*tasks)

    async def _run_one(
        self,
        id: str,
        target: dict,
    ):
        """
        Run the target and save the result as json file
        """

        def hide_inputs(inputs):
            if isinstance(inputs, dict):
                keys = set(inputs.keys())
                if keys == {"input"}:
                    return {}
                elif keys == {"args", "kwargs"}:
                    return {}
                else:
                    return inputs
            elif isinstance(inputs, list):
                return {}

        def hide_outputs(outputs):
            if isinstance(outputs, dict):
                keys = set(outputs.keys())
                if keys == {"output"}:
                    return {}
                else:
                    return outputs
            elif isinstance(outputs, list):
                return {}

        client = Client(
            hide_inputs=hide_inputs,
            hide_outputs=hide_outputs,
        )

        if self.langsmith_mode == "online":
            rt = trace(
                name="ROOT",
                run_type="chain",
                inputs=target,
                client=client,
            )
        path = self.results_dir / f"{id}.json"
        if path.exists() and not self.rerun:
            logging.info(f"{id} already exists. Skipping...")
            result = json.loads(path.read_text())

        else:
            result = await self.graph.ainvoke([target])

        self._save_json(id, result)
        self._save_files(id, result)
        if self.langsmith_mode == "online":
            rt.end(outputs=result)


    def _save_json(self, id: str, result: dict):
        """
        Save result as json file
        """
        with open(f"{self.output_dir}/results/{id}.json", "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    def _save_files(self, id: str, result: str):
        output_dir = self.results_dir / id

        def _rec_save_files(result, dir, key):
            dir.mkdir(parents=True, exist_ok=True)
            key = str(key)

            if isinstance(result, str):
                file = dir / f"{key}.txt"
                with open(file, "w") as f:
                    f.write(result)
                wandb.save(file, base_path=self.results_dir, policy="now")
            elif isinstance(result, list):
                for i in range(len(result)):
                    _rec_save_files(result[i], dir / key, i)
            elif isinstance(result, dict):
                for k, v in result.items():
                    _rec_save_files(v, dir / key, k)
            else:
                file = dir / f"{key}.json"
                with open(file, "w") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                wandb.save(file, base_path=self.results_dir, policy="now")

        _rec_save_files(result, output_dir, "result")

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
                if i < len(d):
                    dump_data[j].update(d[i])
                else:
                    dump_data[j].update({"max_depth": len(d)})

            filename = f"{self.output_dir}/results_merged_{i}.json"
            with open(filename, "w") as f:
                json.dump(dump_data, f, indent=4, ensure_ascii=False)

        return self

    def exit(self):
        """
        Exit the generator
        """
        pass