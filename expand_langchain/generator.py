import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from traceback import format_exc
from typing import Optional, cast

from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from .config import EnvironmentConfig
from .config_registry import get_config
from .dataset_registry import get_dataset
from .graph import CustomStateGraph
from .utils import misc


class Generator(BaseModel):
    config_name: str
    dataset_name: str
    run_name: Optional[str] = None  # if None, config_name is used
    cache_root: Optional[Path] = None

    save_on: bool = True
    rerun: bool = False
    max_concurrency: int = 4
    recursion_limit: int = 100

    verbose: bool = False
    debug: bool = False

    api_keys_path: str = "api_keys.json"
    id_key: str = "task_id"  # key name for id field in dataset

    langfuse_on: bool = False

    # private variables
    root_node: Optional[Runnable] = None
    output_dir: Optional[Path] = None
    result_root: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None

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
        if not self.run_name:
            # Generate run_name with config_name + current datetime
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_name = f"{self.config_name}-{current_time}"

        # Append -debug suffix if debug mode is enabled
        if self.debug and not self.run_name.endswith("-debug"):
            self.run_name = f"{self.run_name}-debug"

        if self.save_on:
            self.output_dir = Path(f"results/{self.run_name}")
            self.result_root = self.output_dir / "results"
            self.result_root.mkdir(parents=True, exist_ok=True)
            # Initialize checkpoint directory for LangGraph time travel
            self.checkpoint_dir = self.output_dir / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not self.cache_root and not self.rerun:
            self.cache_root = self.result_root

    def _load_api_keys(self):
        """Load API keys using the centralized environment config."""
        env_config = EnvironmentConfig(validate_on_init=False)
        try:
            env_config.set_from_api_keys_file(self.api_keys_path)
        except Exception as e:
            logging.warning(f"Failed to load API keys from {self.api_keys_path}: {e}")

    def run(
        self,
        n: Optional[int] = None,
        ids: Optional[list] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        dataset = get_dataset(self.dataset_name).run().get_data()
        self.root_node = get_config(self.config_name)

        # Filter dataset according to parameters
        if n is not None:
            dataset = dataset[: min(n, len(dataset))]
        elif start is not None and end is not None:
            dataset = dataset[start : min(end, len(dataset))]
        elif ids is not None:
            if len(ids) > 0 and isinstance(ids[0], str):
                # Check if dataset has specified id_key field
                if dataset and self.id_key in dataset[0]:
                    filtered_dataset = []
                    for item in dataset:
                        if str(item[self.id_key]) in ids:
                            filtered_dataset.append(item)
                    dataset = filtered_dataset
                else:
                    # Use index as id
                    filtered_dataset = []
                    for i, item in enumerate(dataset):
                        if str(i) in ids:
                            filtered_dataset.append(item)
                    dataset = filtered_dataset
            else:
                # Integer ids - use as indices
                filtered_dataset = []
                for i in ids:
                    if isinstance(i, int) and 0 <= i < len(dataset):
                        filtered_dataset.append(dataset[i])
                dataset = filtered_dataset
        else:
            pass

        # If root_node is a StateGraph, compile it with checkpointer
        if isinstance(self.root_node, StateGraph):
            asyncio.run(self._run_with_checkpointer(dataset))
        else:
            asyncio.run(self._run(dataset))

        return self

    async def _run_with_checkpointer(self, dataset: list):
        """Run with StateGraph compilation and checkpointer management"""
        if self.save_on and self.checkpoint_dir:
            # Use SQLiteSaver for persistent checkpointing
            # Ensure the directory exists and use absolute path
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(self.checkpoint_dir.absolute() / "checkpoints.db")

            # Use the async context manager properly - just the file path
            async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
                # Cast to StateGraph to access compile method
                state_graph = cast(StateGraph, self.root_node)
                compiled_graph = state_graph.compile(checkpointer=checkpointer)
                # Temporarily replace root_node for execution
                original_node = self.root_node
                self.root_node = compiled_graph
                await self._run(dataset)
                self.root_node = original_node
        else:
            # Use in-memory checkpointer if saving is disabled
            checkpointer = InMemorySaver()
            # Cast to StateGraph to access compile method
            state_graph = cast(StateGraph, self.root_node)
            compiled_graph = state_graph.compile(checkpointer=checkpointer)
            # Temporarily replace root_node for execution
            original_node = self.root_node
            self.root_node = compiled_graph
            await self._run(dataset)
            self.root_node = original_node

    async def _run(
        self,
        dataset: list,
    ):
        tasks = []
        sem = asyncio.Semaphore(self.max_concurrency)

        for i in range(len(dataset)):
            target = dataset[i]
            data_id = (
                str(target.get(self.id_key, i)) if isinstance(target, dict) else str(i)
            )
            task = self._run_one(data_id, target, sem, priority=i)
            tasks.append(task)

        await tqdm_asyncio.gather(*tasks)

    async def _check_and_log_checkpoint_loading(self, task_id: str, config: dict):
        """Check if checkpoint exists and log if loading from checkpoint"""
        if not self.save_on or not self.checkpoint_dir:
            logging.debug(
                f"Checkpoint check skipped for {task_id}: "
                f"save_on={self.save_on}, checkpoint_dir={self.checkpoint_dir}"
            )
            return

        # Check if checkpoint database exists first
        db_path = self.checkpoint_dir / "checkpoints.db"
        if not db_path.exists():
            logging.debug(f"Checkpoint database not found for {task_id}: {db_path}")
            return

        logging.debug(f"Checking checkpoint for task {task_id} at {db_path}")

        try:
            # Use AsyncSqliteSaver's built-in methods to check for checkpoints
            async with AsyncSqliteSaver.from_conn_string(
                str(db_path.absolute())
            ) as temp_checkpointer:
                thread_config = config.get("configurable", {})
                thread_id = thread_config.get("thread_id", "unknown")

                # Try to get the latest checkpoint for this thread
                from typing import cast

                from langchain_core.runnables import RunnableConfig

                runnable_config = cast(RunnableConfig, config)
                existing_checkpoint = await temp_checkpointer.aget_tuple(
                    runnable_config
                )

                if existing_checkpoint is not None:
                    # Checkpoint exists, log the loading
                    cp_config = existing_checkpoint.config.get("configurable", {})
                    checkpoint_id = cp_config.get("checkpoint_id", "unknown")
                    msg = (
                        f"Loading from checkpoint for task {task_id} "
                        f"(thread_id: {thread_id}, "
                        f"checkpoint_id: {checkpoint_id})"
                    )
                    logging.info(msg)
                    print(f"CHECKPOINT_LOG: {msg}")
                else:
                    # No checkpoint found, try to list all checkpoints
                    checkpoint_list = []
                    async for cp in temp_checkpointer.alist(runnable_config, limit=1):
                        checkpoint_list.append(cp)

                    if checkpoint_list:
                        # Found checkpoints but couldn't get the specific one
                        msg = (
                            f"Checkpoints exist for task {task_id} "
                            f"but none matches current config"
                        )
                        logging.debug(msg)
                    else:
                        # No checkpoints at all for this thread
                        logging.debug(f"No checkpoints found for task {task_id}")

        except Exception as e:
            # If checkpoint retrieval fails, continue without logging
            logging.debug(f"Could not check checkpoint for task {task_id}: {e}")

    async def _run_one(
        self,
        id: str,
        target: dict,
        sem: asyncio.Semaphore,
        priority: int = 10,
    ):
        """
        Run the target and save the result as json file
        """
        id = str(id).replace("/", "_")

        # Skip if result file already exists and rerun is False
        if not self.rerun and self.result_root:
            result_file = self.result_root / f"{id}.json"
            if result_file.exists():
                # Load and check existing result
                try:
                    with open(result_file, "r") as f:
                        existing_result = json.load(f)

                    # If the result contains an error, rerun the task
                    if "error" in existing_result:
                        logging.info(f"Rerunning {id}: previous result contains error")
                    else:
                        logging.info(f"Skipping {id}: result file already exists")
                        return existing_result
                except Exception as e:
                    logging.warning(f"Failed to load existing result for {id}: {e}")
                    logging.warning("Continuing with execution")

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
                # Add configurable keys for LangGraph checkpointer
                "configurable": {
                    "thread_id": f"thread_{id}",
                    "checkpoint_ns": "",  # Match existing checkpoints
                },
            }

            # Check if checkpoint exists and log if loading from checkpoint
            # Use the converted id (with _ instead of /)
            await self._check_and_log_checkpoint_loading(id, config)

            if self.langfuse_on:
                try:
                    from langfuse.langchain import CallbackHandler

                    langfuse_handler = CallbackHandler()
                    config["callbacks"].append(langfuse_handler)
                except ImportError as e:
                    logging.warning(f"Failed to import langfuse: {e}")
                    logging.warning("Continuing without langfuse logging")
                except Exception as e:
                    logging.warning(f"Failed to initialize langfuse: {e}")
                    logging.warning("Continuing without langfuse logging")

            try:
                result = await self.root_node.ainvoke(  # type: ignore
                    target,
                    config=config,  # type: ignore
                )

                logging.info(f"Done: {id}")

            except Exception as e:
                logging.error(f"Error in running {id}")
                logging.error(format_exc())
                result = {"error": format_exc()}

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
                from langchain_core.messages.base import BaseMessage
                from pydantic import BaseModel

                if isinstance(obj, BaseMessage):
                    return {
                        "type": obj.type,
                        "content": obj.content,
                        "additional_kwargs": obj.additional_kwargs,
                        "response_metadata": obj.response_metadata,
                        "name": obj.name,
                        "id": obj.id,
                    }
                elif isinstance(obj, BaseModel):
                    # Handle Pydantic models by converting to dict
                    return obj.model_dump()
                elif hasattr(obj, "__dict__"):
                    # Handle objects with __dict__ attribute
                    return obj.__dict__
                elif hasattr(obj, "_asdict"):
                    # Handle namedtuples
                    return obj._asdict()
                elif isinstance(obj, (set, frozenset)):
                    # Handle sets
                    return list(obj)
                elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                    # Handle other iterables (but not strings/bytes)
                    try:
                        return list(obj)
                    except:
                        pass
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

    def lark_message(
        self,
        msg: str = "Done",
    ):
        """
        Send message to the webhook using EnvironmentConfig
        """
        env_config = EnvironmentConfig(validate_on_init=False)
        webhook = env_config.lark_webhook

        if webhook is None:
            logging.debug("LARK_WEBHOOK not configured, skipping notification")
            return self

        try:
            import requests

            msg = f"{self.run_name}: {msg}"

            requests.post(
                webhook,
                json={"msg_type": "text", "content": {"text": msg}},
            )
            logging.info(f"Sent Lark notification: {msg}")

        except Exception as e:
            logging.warning(f"Failed to send Lark notification: {e}")

        return self

    def exit(self):
        """
        Exit the generator
        """
        pass
