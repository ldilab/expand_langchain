from typing import List

from langchain_core.runnables import RunnableLambda

from expand_langchain.utils.custom_trace import traceable
from expand_langchain.utils.registry import chain_registry


@chain_registry(name="get_ith")
def get_ith_chain(
    key: str,
    target: str,
    idx: int = 0,
    **kwargs,
):
    def _func(data):
        result = {}
        result[key] = data[target][idx]

        return result

    return RunnableLambda(_func, name="get_ith")
