import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from expand_langchain.utils.cache import load_cache, save_cache
from langchain_core.runnables import Runnable, RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import Field


class CacheChain(RunnableSerializable):
    chain: Runnable
    key_map: Dict[str, str] = Field(default_factory=dict)
    input_keys: List[str] = Field(default_factory=list)
    output_keys: List[str] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    cache_root: Optional[Any] = Field(default=None)

    # pydantic
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)

    def invoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ):
        """Synchronous invoke method that runs the async version."""
        return asyncio.run(self.ainvoke(input, config, **kwargs))

    async def ainvoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ):
        cache_hit = False
        if self.cache_root:
            cache_root = self.cache_root
        else:
            if config:
                cache_root = config.get("metadata", {}).get("cache_root")
            else:
                cache_root = None

        if cache_root:
            assert config is not None, "config must be provided when cache_root is used"

            langgraph_step = config.get("metadata", {}).get("langgraph_step")
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

        result_root = config.get("metadata", {}).get("result_root")  # type: ignore
        if result_root and not cache_hit:
            langgraph_step = (
                config.get("metadata", {}).get("langgraph_step") if config else None
            )
            id = config.get("metadata", {}).get("id") if config else None
            path = result_root / id / "result" / str(langgraph_step)
            for k, v in new_result.items():  # type: ignore
                save_cache(path, k, v)

        return new_result  # type: ignore


class CustomStateGraph(StateGraph):
    def __init__(
        self,
        entry_point: str,
        state_schema: Type[Any],
        nodes: Dict[str, Runnable],
        edges: List[Tuple[str, Union[str, Any]]],
        context_schema: Optional[Type[Any]] = None,
        config_schema: Optional[Type[Any]] = None,
    ):
        if context_schema is None and config_schema is not None:
            context_schema = config_schema
        super().__init__(
            state_schema=state_schema,
            context_schema=context_schema,
        )

        for _k, node in nodes.items():
            self.add_node(_k, node)

        self.set_entry_point(entry_point)

        for start, end in edges:
            if isinstance(end, str):
                self.add_edge(start, end)
            else:
                self.add_conditional_edges(start, end)
