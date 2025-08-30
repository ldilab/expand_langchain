import asyncio
import importlib.util
import logging
import operator
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import networkx as nx
from expand_langchain.utils.cache import load_cache, save_cache
from expand_langchain.utils.registry import chain_registry
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSerializable
from langgraph.graph import StateGraph
from langgraph.constants import END, START
from langgraph.types import Send
from pydantic import Field
from typing_extensions import Annotated, TypedDict


class CacheChain(RunnableSerializable):
    key: str
    type: Optional[str] = Field(default=None)
    key_map: Dict[str, str] = Field(default_factory=dict)
    input_keys: List[str] = Field(default_factory=list)
    output_keys: List[str] = Field(default_factory=list)
    chain: Optional[Runnable] = Field(default=None)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    cache_root: Optional[Any] = Field(default=None)

    # pydantic
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)

        if self.chain is None and self.type is not None:
            self.chain = chain_registry[self.type](
                key=self.key,
                **self.kwargs,
            )

    def invoke(
        self,
        input: dict,
        config: dict = None,
        **kwargs,
    ):
        """Synchronous invoke method that runs the async version."""
        if config is None:
            config = {}
        return asyncio.run(self.ainvoke(input, config, **kwargs))

    async def ainvoke(
        self,
        input: dict,
        config: dict,
        **kwargs,
    ):
        cache_hit = False
        cache_root = self.cache_root or config.get("metadata", {}).get("cache_root")
        if cache_root:
            langgraph_step = config["metadata"]["langgraph_step"]
            id = config.get("metadata", {}).get("id")
            path = cache_root / id / "result" / str(langgraph_step)
            new_result = {}
            for k in self.output_keys:
                try:
                    new_result[k] = load_cache(path, k)
                    logging.info(f"Loaded cache from {path}, key: {k}")
                    cache_hit = True
                except FileNotFoundError:
                    logging.info(f"Cache not found: {path}, key: {k}")
                    cache_hit = False
                    break

        if not cache_hit:
            data = {**input}

            mapped_data = {}
            for k in self.input_keys:
                mapped_key = self.key_map.get(k, k)
                _data = data.get(k, None)
                mapped_data[mapped_key] = _data

            new_result = await self.chain.ainvoke(mapped_data, config=config)

        result_root = config.get("metadata", {}).get("result_root")
        if result_root and not cache_hit:
            langgraph_step = config["metadata"]["langgraph_step"]
            id = config.get("metadata", {}).get("id")
            path = result_root / id / "result" / str(langgraph_step)
            for k, v in new_result.items():
                save_cache(path, k, v)
                logging.info(f"Saved result to {path}, key: {k}")

        return new_result


def send_edge(end: List[str]):
    def route_func(state):
        return [Send(e, state) for e in end]

    return route_func


def reducer(value, values):
    result = {}
    if isinstance(value, dict):
        result.update(value)
    if isinstance(values, list):
        for v in values:
            if isinstance(v, dict):
                result.update(v)

    return result


class DAGChain(StateGraph):
    def __init__(
        self,
        nodes: Dict[str, Runnable],
        adj_list: Dict[str, List[str]],
        config_schema: Optional[Type[Any]] = None,
    ):
        super().__init__(
            state_schema=Annotated[dict, reducer],
            config_schema=config_schema,
        )

        for name, node in nodes.items():
            self.add_node(name, node)

        adj_list[START] = [name for name in nodes.keys()]

        for start, ends in adj_list.items():
            # self.add_conditional_edges(start, send_edge(ends))
            for end in ends:
                self.add_edge(start, end)


class CustomStateGraph(StateGraph):
    def __init__(
        self,
        entry_point: str,
        nodes: Dict[str, Runnable],
        edges: List[Tuple[str, Union[str, Any]]],
        config_schema: Optional[Type[Any]] = None,
    ):
        super().__init__(
            state_schema=Annotated[dict, reducer],
            config_schema=config_schema,
        )

        for _k, node in nodes.items():
            self.add_node(_k, node)

        self.set_entry_point(entry_point)

        for start, end in edges:
            if isinstance(end, str):
                self.add_edge(start, end)
            else:
                self.add_conditional_edges(start, end)


class RootCustomStateGraph(StateGraph):
    root_node: Runnable

    def __init__(
        self,
        root_node: Runnable,
        config_schema: Optional[Type[Any]] = None,
    ):
        super().__init__(
            state_schema=Annotated[dict, reducer],
            config_schema=config_schema,
        )

        self.root_node = root_node
        self.add_node("root", self.root_node)
        self.set_entry_point("root")
        self.add_edge("root", END)
