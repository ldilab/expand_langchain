from typing import List

from expand_langchain.utils.registry import chain_registry
from langchain_core.runnables import RunnableLambda
from langfuse.decorators import langfuse_context, observe


@chain_registry(name="assign")
def assign_chain(
    key: str,
    input_keys: List[str],
    **kwargs,
):
    def _func(data, config={}):
        result = {}
        result[key] = data[input_keys[0]]

        return result

    return RunnableLambda(_func, name="assign")
