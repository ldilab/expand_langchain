import json
from typing import Dict, List

from langchain_core.runnables import RunnableLambda

from expand_langchain.utils.registry import parser_registry


@parser_registry(name="load_json")
def load_json_runner():
    def func_str(input: str):
        try:
            result = json.loads(input)
        except Exception as e:
            result = input

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

    return RunnableLambda(func, "load_json")