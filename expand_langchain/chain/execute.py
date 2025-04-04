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
        if isinstance(code, list) and len(code) == 1 and isinstance(code[0], list):
            code = code[0]

        testcase: Union[List[str], List[dict]] = data.get(testcase_key, [])
        if is_direct_stdin:
            stdin = testcase
            if (
                isinstance(stdin, list)
                and len(stdin) == 1
                and isinstance(stdin[0], list)
            ):
                stdin = stdin[0]
        else:
            stdin = [x.get(stdin_key, "") for x in testcase]

        result = {}
        result[key] = []
        for _c in code:
            if isinstance(_c, list) and len(_c) == 1 and isinstance(_c[0], str):
                _c = _c[0]

            _r = []
            for s in stdin:
                if isinstance(s, list) and len(s) == 1 and isinstance(s[0], list):
                    s = s[0]
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

    try:
        return response["output"].replace("Exit Code: 0", "")
    except Exception:
        return str(response)
