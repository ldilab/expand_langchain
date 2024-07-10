from typing import Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel


class SourceConfig(BaseModel):
    name: str
    path: str
    type: str
    sort_key: str
    kwargs: Optional[dict] = None


class FieldConfig(BaseModel):
    name: str
    source: str
    key: str


class DatasetConfig(BaseModel):
    name: str
    primary_key: str
    fields: List[FieldConfig]


class EdgeConfig(BaseModel):
    pair: Tuple[str, str]
    type: str
    kwargs: Optional[dict] = None


class ChainConfig(BaseModel):
    name: str
    dependencies: List[str]
    input_keys: List[str]
    key_map: Dict[str, str] = None
    type: str
    kwargs: dict = {}

    def __init__(self, **data):
        super().__init__(**data)

        if self.key_map is None:
            self.key_map = {key: key for key in self.input_keys}
        else:
            self.key_map = {**{key: key for key in self.input_keys}, **self.key_map}


class NodeConfig(BaseModel):
    name: str
    chains: List[ChainConfig]


class GraphConfig(BaseModel):
    entry_point: str
    edges: List[EdgeConfig]
    nodes: List[NodeConfig]


class Config(BaseModel):
    description: Optional[str] = None
    source: List[SourceConfig] = None
    dataset: List[DatasetConfig] = None
    graph: GraphConfig = None

    def __init__(self, **data):
        path = data.get("path")
        if path is not None:
            with open(path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            del data["path"]

            data = {**config, **data}

        super().__init__(**data)
