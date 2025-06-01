from dataclasses import dataclass, make_dataclass

import argparse
from pathlib import Path

import yaml
from graphviz import Digraph


@dataclass
class Node:
    name: str
    type: str
    dependencies: list[str] | list['Node']
    input_keys: list[str]
    output_keys: list[str]


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_yaml", type=Path, required=True, help="target yaml file path")
    argparser.add_argument("--output_graph", type=Path, required=True, help="output graph image save path")
    args = argparser.parse_args()

    input_yaml = args.input_yaml
    output_graph = args.output_graph

    # ─── 1) YAML file load ──────────────────────────────
    yaml_file = input_yaml
    with open(yaml_file, 'r', encoding='utf-8') as f:
        parsed = yaml.safe_load(f)

    graph_section = parsed["graph"]
    nodes = graph_section["nodes"][0]["chains"]
    nodes = {
        node["name"]: Node(**{
            k: v
            for k, v in node.items() if k in ['name', 'type', 'dependencies', 'input_keys', 'output_keys']
        })
        for node in nodes
    }

    for node in nodes:
        dep_strs = nodes[node].dependencies
        dep_objs = [
            nodes[dep_str]
            for dep_str in dep_strs
        ]
        nodes[node].dependencies = dep_objs

    # ─── (C) Graphviz Digraph ────────────────
    dot = Digraph(
        name='PipelineGraph',
        format='png',
        node_attr={'shape': 'plaintext', 'fontname': 'Helvetica'},
        edge_attr={'fontname': 'Helvetica'}
    )

    for node in nodes.values():
        label = f"{node.name}\nInputs: {node.input_keys}\nOutputs: {node.output_keys}"
        dot.node(node.name, label=label, shape='box')

        for dep in node.dependencies:
            dot.edge(dep.name, node.name)

    # ─── (F) render and save ────────────────────────────
    output_path = dot.render(directory=output_graph.parent, filename=output_graph.stem, cleanup=True)
    print(f"Graph visualization saved: {output_path}")
