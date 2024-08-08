from typing import List

from expand_langchain.utils.registry import chain_registry
from langchain_core.runnables import RunnableLambda
from langfuse.decorators import langfuse_context, observe


@chain_registry(name="get_ith")
def get_ith_chain(
    key: str,
    target: str,
    idx: int = 0,
    **kwargs,
):
    def _func(data, config={}):
        result = {}
        result[key] = data[target][idx]

        return result

    return RunnableLambda(_func, name="get_ith")
