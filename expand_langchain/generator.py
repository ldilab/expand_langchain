import asyncio
import json
import logging
import os
from pathlib import Path
from traceback import format_exc
from typing import Optional

from datasets import Dataset
from langgraph.graph import StateGraph
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from .graph import RootCustomStateGraph
from .utils import misc


class Generator(BaseModel):
    root_node: StateGraph

    run_name: str = None  # if None, config_path.stem is used
    cache_root: Optional[Path] = None

    save_on: bool = True
    rerun: bool = False
    max_concurrency: int = 4
    recursion_limit: int = 100

    verbose: bool = False
    debug: bool = False

    api_keys_path: str = "api_keys.json"

    langfuse_on: bool = False

    target_dataset: Dataset

    # private variables
    output_dir: Path = None
    result_root: Path = None
    graph: RootCustomStateGraph = None

    # pydantic config
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        elif self.verbose:
            logging.basicConfig(level=logging.INFO)

        logging.getLogger("snowflake").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        self._init_result_dir()
        self._load_api_keys()

    def _init_result_dir(self):
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
                if not os.environ.get(k, None):
                    logging.warning(f"Set {k} from api_keys file")
                    os.environ[k] = v

    def compile_graph(self):
        self.graph = RootCustomStateGraph(self.root_node.compile()).compile()

    def run(
        self,
        n: Optional[int] = None,
        ids: Optional[list] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        dataset = self.target_dataset
        if self.graph is None:
            self.compile_graph()

        if n is not None:
            dataset = dataset.select(range(min(n, len(dataset))))
        elif start is not None and end is not None:
            dataset = dataset.select(range(start, min(end, len(dataset))))
        elif ids is not None:
            # ids가 string 키 리스트인 경우, 해당 키를 가진 데이터만 선택
            if len(ids) > 0 and isinstance(ids[0], str):
                # Dataset에서 id 필드를 기준으로 필터링
                # 일반적으로 Dataset에 'id' 컬럼이 있다고 가정
                if "id" in dataset.column_names:
                    indices = []
                    for i in range(len(dataset)):
                        if str(dataset[i]["id"]) in ids:
                            indices.append(i)
                    dataset = dataset.select(indices)
                else:
                    # id 컬럼이 없는 경우, 첫 번째 컬럼을 키로 사용하거나 인덱스를 문자열로 처리
                    indices = []
                    for i in range(len(dataset)):
                        if str(i) in ids:
                            indices.append(i)
                    dataset = dataset.select(indices)
            else:
                # ids가 정수 인덱스 리스트인 경우
                indices = [i for i in ids if isinstance(i, int) and i < len(dataset)]
                dataset = dataset.select(indices)
        else:
            pass

        asyncio.run(self._run(dataset))

        return self

    async def _run(
        self,
        dataset: Dataset,
    ):
        tasks = []
        sem = asyncio.Semaphore(self.max_concurrency)

        # Dataset을 이터레이션하면서 인덱스와 데이터를 함께 가져옴
        for i in range(len(dataset)):
            target = dataset[i]
            # target에 id 필드가 있으면 사용하고, 없으면 인덱스를 사용
            data_id = target.get("id", i) if isinstance(target, dict) else i
            task = self._run_one(data_id, target, sem)
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
                "recursion_limit": self.recursion_limit,
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

            # try:
            result = await self.graph.ainvoke(
                target,
                config=config,
            )

            logging.info(f"Done: {id}")

            # except Exception as e:
            #     logging.error(f"Error in running {id}")
            #     logging.error(format_exc())
            #     result = {"error": format_exc()}

            #     if self.debug:
            #         raise e

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
                from langchain_core.messages.base import BaseMessage

                if isinstance(obj, BaseMessage):
                    return {
                        "type": obj.type,
                        "content": obj.content,
                        "additional_kwargs": obj.additional_kwargs,
                        "response_metadata": obj.response_metadata,
                        "name": obj.name,
                        "id": obj.id,
                    }
                try:
                    return super().default(obj)
                except TypeError:
                    return "Not serializable"

        with open(self.result_root / f"{id}.json", "w") as f:
            json.dump(result, f, indent=4, cls=CustomEncoder, ensure_ascii=False)

    def _save_yaml(self, id: str, result: dict):
        """
        Save result as yaml file
        """
        path = self.result_root / f"{id}.yaml"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        misc.pretty_yaml_dump(result, path)

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

        dump_data = [{} for _ in range(len(data))]
        for j, d in enumerate(data):
            if isinstance(d, list):
                dump_data[j].update(d[-1])
            elif isinstance(d, dict):
                dump_data[j] = d
            else:
                raise ValueError("Invalid data type")

        filename = f"{self.output_dir}/results_merged.json"
        with open(filename, "w") as f:
            json.dump(dump_data, f, indent=4, ensure_ascii=False)

        filename = f"{self.output_dir}/results_merged.yaml"
        misc.pretty_yaml_dump(dump_data, filename)

        return self

    def exit(self):
        """
        Exit the generator
        """
        pass
