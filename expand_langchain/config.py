from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
import yaml_include
from pydantic import BaseModel


class SourceConfig(BaseModel):
    name: str
    type: str
    kwargs: Optional[dict] = None


class TransformConfig(BaseModel):
    type: str
    kwargs: Optional[dict] = None


class FieldConfig(BaseModel):
    name: str
    source: Optional[str] = None
    key: Optional[str] = None
    value: Optional[Any] = None
    transform: Optional[TransformConfig] = None


class DatasetConfig(BaseModel):
    name: str
    type: str
    remove: bool = False
    kwargs: Optional[dict] = None


class EdgeConfig(BaseModel):
    pair: List[str]
    route: str
    kwargs: Optional[dict] = None


class ChainConfig(BaseModel):
    name: str
    dependencies: List[str] = []
    key_map: Dict[str, str] = {}
    type: str
    input_keys: List[str] = []
    output_keys: List[str] = []
    cache_root: Optional[Path] = None
    kwargs: dict = {}

    def __init__(self, **data):
        super().__init__(**data)


class NodeConfig(BaseModel):
    name: str
    chains: List[ChainConfig]


class GraphConfig(BaseModel):
    entry_point: str
    edges: List[EdgeConfig]
    nodes: List[NodeConfig]
    routes: Optional[Dict[str, str]] = None


class Config(BaseModel):
    description: Optional[str] = None
    source: List[SourceConfig] = None
    dataset: List[DatasetConfig] = None
    graph: GraphConfig = None

    def __init__(self, **data):
        path = data.get("path")
        if path is not None:
            yaml.add_constructor("!inc", yaml_include.Constructor())
            with open(path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            del data["path"]

            data = {**config, **data}

        super().__init__(**data)
