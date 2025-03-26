import os
from itertools import zip_longest
from typing import List, Optional, Union

from expand_langchain.utils.registry import chain_registry
from langchain_community.utilities.requests import JsonRequestsWrapper
from langchain_core.runnables import RunnableLambda


@chain_registry(name="execute")
def execute_chain(
    key: str,
    code_key: str,
    testcase_key: Optional[str] = None,
    stdin_key: Optional[str] = None,
    is_direct_stdin: Optional[bool] = False,
    **kwargs,
):
    def _func(data, config={}):
        code: List[str] = data[code_key]
        testcase: Union[List[str], List[dict]] = data.get(testcase_key, [])
        if is_direct_stdin:
            stdin = testcase
        else:
            stdin = [x.get(stdin_key, "") for x in testcase]

        result = {}
        result[key] = []
        for _c in code:
            _r = []
            for s in stdin:
                output = send_code_to_codeexec(_c, s)
                _r.append(output)
            result[key].append(_r)

        return result

    return RunnableLambda(_func, name="execute")


def send_code_to_codeexec(code: str, stdin: str = ""):
    response = JsonRequestsWrapper().post(
        os.environ["CODEEXEC_ENDPOINT"],
        data={
            "code": code,
            "stdin": stdin,
        },
    )

    response["output"] = response["output"].replace("Exit Code: 0", "")

    return response["output"]
