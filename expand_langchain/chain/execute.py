import os
from typing import List, Optional

from expand_langchain.utils.registry import chain_registry
from langchain_community.utilities.requests import JsonRequestsWrapper
from langchain_core.runnables import RunnableLambda


@chain_registry(name="execute")
def execute_chain(
    key: str,
    code_key: str,
    testcase_key: Optional[str] = None,
    stdin_key: Optional[str] = None,
    **kwargs,
):
    def _func(data, config={}):
        result = {}
        result[key] = []
        for target, testcase in zip(data[code_key], data[testcase_key]):
            if isinstance(target, str):
                response = JsonRequestsWrapper().post(
                    os.environ["CODEEXEC_ENDPOINT"],
                    data={
                        "code": target,
                        "stdin": testcase[stdin_key],
                        **kwargs,
                    },
                )
                output = response["output"]
                result[key].append(output)

            elif isinstance(target, list):
                outputs = []
                for _target, _testcase in zip(target, testcase):
                    response = JsonRequestsWrapper().post(
                        os.environ["CODEEXEC_ENDPOINT"],
                        data={
                            "code": _target,
                            "stdin": _testcase[stdin_key],
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
