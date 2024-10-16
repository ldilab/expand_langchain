import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
from expand_langchain.config import ChainConfig, GraphConfig, NodeConfig
from expand_langchain.utils.registry import chain_registry, transition_registry
from langchain_core.runnables import RunnableLambda
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
            chain = node_factory(
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


def node_factory(
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

    return graph_chain(nx_graph)


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


def graph_chain(graph: nx.DiGraph):
    async def run_node(chain, inputs, results, tasks, lock, config):
        if chain.name in results:
            return results[chain.name]

        parents = list(graph.predecessors(chain.name))
        new_tasks = []
        for parent in parents:
            _chain = graph.nodes[parent]["chain"]
            async with lock:
                if parent not in tasks:
                    task = asyncio.create_task(
                        run_node(_chain, inputs, results, tasks, lock, config)
                    )
                    tasks[parent] = task
                else:
                    task = tasks[parent]
                new_tasks.append(task)

        if new_tasks:
            await asyncio.gather(*new_tasks)

        new_result = await chain.ainvoke({**inputs, **results}, config=config)
        async with lock:
            results.update(new_result)

        return new_result

    async def _func(data: List[dict], config={}):
        inputs = {}
        for input in data:
            inputs.update(input)

        lock = asyncio.Lock()
        results = {}
        tasks = {}
        for node in nx.topological_sort(graph):
            chain = graph.nodes[node]["chain"]
            async with lock:
                if chain.name not in tasks:
                    task = asyncio.create_task(
                        run_node(chain, inputs, results, tasks, lock, config)
                    )
                    tasks[chain.name] = task

        await asyncio.gather(*list(tasks.values()))

        data.append(results)
        return data

    return RunnableLambda(_func, name="graph_chain")
