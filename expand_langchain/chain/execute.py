import asyncio
import os

from expand_langchain.utils.registry import chain_registry
from langchain_community.utilities.requests import JsonRequestsWrapper
from langchain_core.runnables import RunnableLambda
from langfuse.decorators import langfuse_context, observe


@chain_registry(name="execute")
def execute_chain(
    key: str,
    target: str,
    **kwargs,
):
    @observe()
    def _func(data):
        result = {}
        result[key] = []
        for input in data[target]:
            if isinstance(input, str):
                response = JsonRequestsWrapper().post(
                    os.environ["CODEEXEC_ENDPOINT"],
                    data={
                        "code": input,
                        **kwargs,
                    },
                )
                output = response["output"]
                result[key].append(output)

            elif isinstance(input, list):
                outputs = []
                for _input in input:
                    response = JsonRequestsWrapper().post(
                        os.environ["CODEEXEC_ENDPOINT"],
                        data={
                            "code": _input,
                            **kwargs,
                        },
                    )
                    output = response["output"]
                    outputs.append(output)
                result[key].append(outputs)
            else:
                raise ValueError("Invalid input type")

        return result

    return RunnableLambda(_func, name="execute")
