import logging
from typing import Dict, List

from expand_langchain.utils.registry import parser_registry
from langchain_core.runnables import RunnableLambda


@parser_registry(name="load_python_obj")
def load_python_obj_runner():
    def func_str(input: str):
        local_scope = {}
        try:
            # Execute the input string
            exec(input, local_scope)

            # Try to get 'result' from local scope
            if "result" in local_scope:
                return local_scope["result"]

            # If 'result' is not defined, try to evaluate the input as an expression
            try:
                return eval(input, local_scope)
            except:
                # If eval fails, check if there are any variables in local_scope
                # excluding built-in ones
                user_vars = {
                    k: v for k, v in local_scope.items() if not k.startswith("__")
                }

                if len(user_vars) == 1:
                    # If there's exactly one user-defined variable, return it
                    return list(user_vars.values())[0]
                elif len(user_vars) > 1:
                    # If there are multiple variables, return them as a dict
                    return user_vars
                else:
                    # If no variables and eval failed, return None
                    return None

        except Exception as e:
            logging.warning(f"Failed to execute input: {input}. Error: {e}")
            return None

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
