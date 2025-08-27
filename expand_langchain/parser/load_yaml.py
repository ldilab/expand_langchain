from typing import Dict, List

from expand_langchain.utils.registry import parser_registry
from langchain_core.runnables import RunnableLambda


@parser_registry(name="load_yaml")
def load_yaml_runner():
    def func_str(input: str):
        try:
            # Strip whitespace and handle empty strings
            input = input.strip()
            if not input:
                return {}

            # Try to parse the YAML input
            import yaml

            # Use safe_load with allow_unicode=True for better Unicode support
            result = yaml.safe_load(input)
            return result
        except yaml.YAMLError as e:
            # If parsing fails, return an error message with more details
            return {
                "error": "YAML parsing error",
                "message": str(e),
                "input": input[:200] + "..." if len(input) > 200 else input,
            }
        except Exception as e:
            return {"error": "Unknown error", "message": str(e)}

    def func(input):
        if isinstance(input, str):
            return func_str(input)
        elif isinstance(input, List):
            return [func_str(text) if isinstance(text, str) else text for text in input]
        elif isinstance(input, Dict):
            return {
                key: func_str(text) if isinstance(text, str) else text
                for key, text in input.items()
            }
        else:
            raise ValueError(f"input type {type(input)} is not supported")

    return RunnableLambda(func, "load_yaml")
