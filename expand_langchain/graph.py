import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import networkx as nx
from expand_langchain.config import ChainConfig, GraphConfig, NodeConfig
from expand_langchain.utils.cache import load_cache, save_cache
from expand_langchain.utils.registry import chain_registry, transition_registry
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph
from pydantic import BaseModel


class NodeChain(BaseModel, Runnable):
    key: str
    type: str
    key_map: Dict[str, str]
    input_keys: List[str]
    output_keys: List[str]
    cache_root: Optional[Path] = None
    examples: dict
    etc_datasets: dict
    kwargs: dict = {}

    chain: Runnable = None

    # pydantic config
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        self.name = "NodeChain"
        self.chain = chain_registry[self.type](
            key=self.key,
            examples=self.examples,
            **self.kwargs,
        )

    async def astream(
        self,
        input: dict,
        config: dict,
        graph: nx.DiGraph,
        results: dict,
        lock: asyncio.Lock,
        **kwargs,
    ):
        if self.key in results:
            yield results
            return

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
            parents = list(graph.predecessors(self.key))
            for parent in parents:
                node = graph.nodes[parent]["chain"]
                gen = node.astream(
                    input=input,
                    config=config,
                    graph=graph,
                    results=results,
                    lock=lock,
                )
                async for _ in gen:
                    yield results

            data = {**input, **results}

            mapped_data = {}
            for k in self.input_keys:
                mapped_key = self.key_map.get(k, k)
                mapped_data[mapped_key] = data.get(k, None) or self.etc_datasets[k]

            new_result = await self.chain.ainvoke(mapped_data, config=config)

        result_root = config.get("metadata", {}).get("result_root")
        if result_root:
            langgraph_step = config["metadata"]["langgraph_step"]
            id = config.get("metadata", {}).get("id")
            path = result_root / id / "result" / str(langgraph_step)
            for k, v in new_result.items():
                save_cache(path, k, v)
                logging.info(f"Saved result to {path}, key: {k}")

        async with lock:
            results.update(new_result)

        yield results

    def invoke(
        self,
        input: dict,
        config: dict,
        graph: nx.DiGraph,
        results: dict,
        lock: asyncio.Lock,
        **kwargs,
    ):
        return asyncio.run(self.ainvoke(input, graph, results, lock, config))

    async def ainvoke(
        self,
        input: dict,
        config: dict,
        graph: nx.DiGraph,
        results: dict,
        lock: asyncio.Lock,
        **kwargs,
    ):
        result = [
            res
            async for res in self.astream(
                input=input,
                config=config,
                graph=graph,
                results=results,
                lock=lock,
            )
        ]

        return result[-1]


class GraphChain(BaseModel, Runnable):
    graph: nx.DiGraph = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        configs: List[ChainConfig],
        examples: dict = {},
        etc_datasets: dict = {},
        **data,
    ):
        super().__init__(**data)

        self.name = "GraphChain"

        nx_graph = nx.DiGraph()
        for config in configs:
            name = config.name
            dependencies = config.dependencies
            input_keys = config.input_keys
            output_keys = config.output_keys
            cache_root = config.cache_root
            key_map = config.key_map
            type = config.type
            kwargs = config.kwargs or {}

            chain = NodeChain(
                key=name,
                type=type,
                key_map=key_map,
                input_keys=input_keys,
                output_keys=output_keys,
                cache_root=cache_root,
                examples=examples,
                etc_datasets=etc_datasets,
                kwargs=kwargs,
            )
            nx_graph.add_node(name, chain=chain)

        for config in configs:
            name = config.name
            dependencies = config.dependencies
            for dependency in dependencies:
                if dependency in nx_graph.nodes:
                    nx_graph.add_edge(dependency, name)

        self.graph = nx_graph

    async def astream(
        self,
        input: List[dict],
        config: dict,
        **kwargs,
    ):
        node_input = {}
        for _input in input:
            node_input.update(_input)

        lock = asyncio.Lock()
        results = {}
        input.append(results)
        for node in nx.topological_sort(self.graph):
            chain = self.graph.nodes[node]["chain"]
            gen = chain.astream(
                input=node_input,
                config=config,
                graph=self.graph,
                results=results,
                lock=lock,
            )
            async for result in gen:
                input[-1] = result
                yield input

    def invoke(
        self,
        input: List[dict],
        config: dict,
        **kwargs,
    ):
        return asyncio.run(self.ainvoke(input, config))

    async def ainvoke(
        self,
        input: List[dict],
        config: dict,
        **kwargs,
    ):
        result = [res async for res in self.astream(input, config)]

        return result[-1]


class CustomLangGraph(StateGraph):
    def __init__(
        self,
        config: GraphConfig,
        examples: dict = {},
        etc_datasets: dict = {},
        cache_root: str = None,
        result_root: str = None,
        config_schema: Optional[Type[Any]] = None,
    ):
        super().__init__(
            state_schema=List[dict],
            config_schema=config_schema,
        )

        self.config = config
        self.examples = examples
        self.etc_datasets = etc_datasets

        nodes = self.config.nodes
        for node in nodes:
            node: NodeConfig
            chain = GraphChain(
                configs=node.chains,
                examples=self.examples,
                etc_datasets=self.etc_datasets,
                cache_root=cache_root,
                result_root=result_root,
            )

            self.add_node(node.name, chain)

        entry_point = self.config.entry_point
        self.set_entry_point(entry_point)

        edges = self.config.edges
        for edge in edges:
            pair = edge.pair
            if edge.type == "always":
                self.add_edge(pair[0], pair[1])
            else:
                func = transition_registry.get(edge.type)(dest=pair[1], **edge.kwargs)
                self.add_conditional_edges(pair[0], func)
