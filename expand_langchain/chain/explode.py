import os
from typing import List

from expand_langchain.utils.registry import chain_registry, model_registry
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_elasticsearch import ElasticsearchStore
from typing import *


@chain_registry(name="explode")
def explode_chain(
    target_key: str,
    id_key: str = "id",
    **kwargs,
):
    def _func(data, config={}):

        # Explode the target key
        exploded_data = []
        for idx, item in enumerate(data[target_key][0]):
            new_item = data.copy()
            new_item[target_key] = item
            new_item[id_key] = f"{data[id_key]}_{idx}"
            exploded_data.append(new_item)

        return exploded_data

    return RunnableLambda(_func, name="indexing")
