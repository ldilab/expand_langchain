import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
from expand_langchain.config import ChainConfig, GraphConfig, NodeConfig
from expand_langchain.utils.registry import chain_registry, transition_registry
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import StateGraph
from pydantic import BaseModel


class Graph(BaseModel):
    config: GraphConfig
    examples: dict = {}
    etc_datasets: dict = {}

    def __init__(self, **data):
        super().__init__(**data)

    def run(self):
        builder = StateGraph(List[dict])

        nodes = self.config.nodes
        for node in nodes:
            node: NodeConfig
            chain = langgraph_node_factory(
                configs=node.chains,
                examples=self.examples,
                etc_datasets=self.etc_datasets,
            )

            builder.add_node(node.name, chain)

        entry_point = self.config.entry_point
        builder.set_entry_point(entry_point)

        edges = self.config.edges
        for edge in edges:
            pair = edge.pair
            if edge.type == "always":
                builder.add_edge(pair[0], pair[1])
            else:
                func = transition_registry.get(edge.type)(dest=pair[1], **edge.kwargs)
                builder.add_conditional_edges(pair[0], func)

        return builder.compile()


def langgraph_node_factory(
    configs: List[ChainConfig],
    examples: dict = {},
    etc_datasets: dict = {},
):
    nx_graph = nx.DiGraph()
    for config in configs:
        name = config.name
        dependencies = config.dependencies
        input_keys = config.input_keys
        key_map = config.key_map
        type = config.type
        cache_path = config.cache_path
        kwargs = config.kwargs or {}

        chain = node_chain(
            key=name,
            input_keys=input_keys,
            key_map=key_map,
            type=type,
            cache_path=cache_path,
            examples=examples,
            etc_datasets=etc_datasets,
            **kwargs,
        )
        nx_graph.add_node(name, chain=chain)

    for config in configs:
        name = config.name
        dependencies = config.dependencies
        for dependency in dependencies:
            if dependency in nx_graph.nodes:
                nx_graph.add_edge(dependency, name)

    return GraphChain(nx_graph)


def node_chain(
    key: str,
    input_keys: List[str],
    key_map: Dict[str, str],
    type: str,
    cache_path: Optional[Path] = None,
    examples: dict = {},
    etc_datasets: dict = {},
    **kwargs,
):
    chain = chain_registry[type](
        key=key,
        input_keys=input_keys,
        examples=examples,
        **kwargs,
    )

    async def _func(data, config={}):
        verbose = config.get("verbose", False)
        id = config.get("id", None)
        if id and cache_path:
            path = cache_path / f"{id}.json"
            if path.exists():
                if verbose:
                    logging.info(f"Loading from cache: {path}")
                with open(path, "r") as f:
                    return json.load(f)

        cur_data = {}
        for key in input_keys:
            cur_data[key_map[key]] = data.get(key, None) or etc_datasets[key]

        result = await chain.ainvoke(cur_data, config=config)

        if id and cache_path:
            with open(path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            if verbose:
                logging.info(f"Saved to cache: {path}")

        return result

    return RunnableLambda(_func, name=key)


class GraphChain(Runnable):
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def invoke(
        self,
        input: List[dict],
        config=None,
        **kwargs,
    ):
        return asyncio.run(self.ainvoke(input, config))

    async def ainvoke(
        self,
        input: List[dict],
        config=None,
        **kwargs,
    ):
        result = [res async for res in self.astream(input, config)]

        return result[-1]

    async def astream(
        self,
        input: List[dict],
        config=None,
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
            node = NodeChain(chain)
            gen = node.astream(node_input, self.graph, results, lock, config)
            async for result in gen:
                input[-1] = result
                yield input


class NodeChain(Runnable):
    def __init__(self, chain: Runnable):
        self.chain = chain

    def invoke(
        self,
        input: dict,
        graph: nx.DiGraph,
        results: dict,
        lock: asyncio.Lock,
        config=None,
        **kwargs,
    ):
        return asyncio.run(self.ainvoke(input, graph, results, lock, config))

    async def ainvoke(
        self,
        input: dict,
        graph: nx.DiGraph,
        results: dict,
        lock: asyncio.Lock,
        config=None,
        **kwargs,
    ):
        result = [
            res async for res in self.astream(input, graph, results, lock, config)
        ]

        return result[-1]

    async def astream(
        self,
        input: dict,
        graph: nx.DiGraph,
        results: dict,
        lock: asyncio.Lock,
        config=None,
        **kwargs,
    ):
        if self.chain.name in results:
            yield results

        else:
            parents = list(graph.predecessors(self.chain.name))
            for parent in parents:
                _chain = graph.nodes[parent]["chain"]
                node = NodeChain(_chain)
                gen = node.astream(input, graph, results, lock, config)
                async for _ in gen:
                    yield results

            new_result = await self.chain.ainvoke({**input, **results}, config=config)
            async with lock:
                results.update(new_result)

            yield results
