from typing import Dict, List

import json5 as json
from expand_langchain.utils.registry import parser_registry
from langchain_core.runnables import RunnableLambda


@parser_registry(name="load_json")
def load_json_runner():
    def func_str(input: str):
        try:
            # Strip whitespace and handle empty strings
            input = input.strip()
            if not input:
                return {}

            # Try to parse with json5 first
            result = json.loads(input)
            return result
        except Exception as e:
            # If json5 fails, try with standard json
            try:
                import json as std_json

                result = std_json.loads(input)
                return result
            except (std_json.JSONDecodeError, ValueError):
                # If both fail, try to handle common issues
                try:
                    # Remove potential BOM or invisible characters
                    cleaned_input = input.encode("utf-8").decode("utf-8-sig").strip()
                    if cleaned_input:
                        result = json.loads(cleaned_input)
                        return result
                    else:
                        return {}
                except Exception:
                    # Try ast.literal_eval for Python literals (including tuples)
                    try:
                        import ast

                        result = ast.literal_eval(input)
                        # Convert tuples to lists recursively for JSON compatibility
                        result = _convert_tuples_to_lists(result)
                        return result
                    except (ValueError, SyntaxError):
                        # Try to fix common tuple syntax and parse again
                        try:
                            # Replace tuples with lists for JSON compatibility
                            fixed_input = _fix_tuple_syntax(input)
                            result = json.loads(fixed_input)
                            return result
                        except Exception:
                            # Last resort: return the original string wrapped in a dict
                            return {"raw_data": input, "parse_error": str(e)}

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

    return RunnableLambda(func, "load_json")


def _convert_tuples_to_lists(obj):
    """Recursively convert tuples to lists for JSON compatibility"""
    if isinstance(obj, tuple):
        return [_convert_tuples_to_lists(item) for item in obj]
    elif isinstance(obj, list):
        return [_convert_tuples_to_lists(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _convert_tuples_to_lists(value) for key, value in obj.items()}
    else:
        return obj


def _fix_tuple_syntax(input_str: str) -> str:
    """Try to convert Python tuple syntax to JSON array syntax"""
    import re

    # Replace tuple parentheses with square brackets
    # Handle simple cases like (a, b) -> [a, b]
    # This is a basic implementation and might need refinement
    fixed = re.sub(r"\(([^()]*)\)", r"[\1]", input_str)
    return fixed
