import os
from itertools import zip_longest
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
        for target, testcase in zip_longest(
            data[code_key],
            data.get(testcase_key, []),
            fillvalue={},
        ):
            if isinstance(target, str):
                response = JsonRequestsWrapper().post(
                    os.environ["CODEEXEC_ENDPOINT"],
                    data={
                        "code": target,
                        "stdin": testcase.get(stdin_key, ""),
                        **kwargs,
                    },
                )
                output = response["output"]
                result[key].append(output)

            elif isinstance(target, list):
                outputs = []
                for _target, _testcase in zip_longest(
                    target,
                    testcase,
                    fillvalue={},
                ):
                    response = JsonRequestsWrapper().post(
                        os.environ["CODEEXEC_ENDPOINT"],
                        data={
                            "code": _target,
                            "stdin": _testcase.get(stdin_key, ""),
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
