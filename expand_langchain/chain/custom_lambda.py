import itertools
from typing import Any, List

from expand_langchain.utils.custom_trace import traceable
from expand_langchain.utils.registry import chain_registry
from langchain_core.runnables import RunnableLambda


@chain_registry(name="custom_lambda")
def custom_lambda_chain(
    key: str,
    src: List[str],
    func: str,
    **kwargs,
):
    def _func(data):
        try:
            func_obj = eval(func)
        except:
            local_namespace = {}
            exec(func, globals(), local_namespace)
            func_obj = local_namespace["func"]

        result = {}
        result[key] = []

        new_src = [data[x] if isinstance(data[x], list) else [data[x]] for x in src]
        for args in itertools.product(*new_src):
            result[key].append(func_obj(*args))

        return result

    return RunnableLambda(_func, name="custom_lambda")