import re
from typing import Dict, List, Tuple, Union

from expand_langchain.utils.registry import parser_registry
from langchain_core.runnables import RunnableLambda


@parser_registry(name="code_block")
def code_block_runner():
    def func_str(input: str):
        pattern = r"```[a-z]*\n(.*?)```"
        match = re.search(pattern, input, re.DOTALL)
        if match:
            matches = re.findall(pattern, input, re.DOTALL)
            if matches:
                return matches[-1]
            else:
                return match.group(1)
        else:
            # if last ``` is missing
            pattern = r"```[a-z]*\n(.*?)$"
            match = re.search(pattern, input, re.DOTALL)
            if match:
                matches = re.findall(pattern, input, re.DOTALL)
                if matches:
                    return matches[-1]
                else:
                    return match.group(1)
            else:
                return input

    def func(input):
        if isinstance(input, str):
            return func_str(input)
        elif isinstance(input, List):
            return [func_str(text) for text in input]
        elif isinstance(input, Dict):
            return {key: func_str(text) for key, text in input.items()}
        else:
            raise ValueError(f"input type {type(input)} is not supported")

    return RunnableLambda(func, "code_block")
