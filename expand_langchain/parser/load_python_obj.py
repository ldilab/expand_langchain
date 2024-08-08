import logging
from typing import Dict, List

from expand_langchain.utils.registry import parser_registry
from langchain_core.runnables import RunnableLambda


@parser_registry(name="load_python_obj")
def load_python_obj_runner():
    def func_str(input: str):
        local_scope = {}
        exec(input, local_scope)
        result = local_scope.get("result")

        return result

    def func(input):
        if isinstance(input, str):
            return func_str(input)
        elif isinstance(input, List):
            return [func_str(text) for text in input]
        elif isinstance(input, Dict):
            return {key: func_str(text) for key, text in input.items()}
        else:
            raise ValueError(f"input type {type(input)} is not supported")

    return RunnableLambda(func, "load_python_obj")
