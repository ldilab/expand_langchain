import os
from itertools import zip_longest
import itertools
from typing import List, Optional, Union

from expand_langchain.utils.registry import chain_registry
from langchain_community.utilities.requests import JsonRequestsWrapper
from langchain_core.runnables import RunnableLambda
import subprocess

def get_python_executable(base_path, venv_name):
    """
    base_path: str, path to the base directory.
    venv_name: str, name of the virtual environment.
    return: str, path to the Python executable in the virtual environment.
    """
    venv_path = base_path + venv_name
    bin_path = os.path.join(venv_path, "bin")
    python_executable = os.path.join(bin_path, "python")
    return python_executable

@chain_registry(name="execute_w_venv")
def execute_offline_chain(
    key: str, # library_example_usage_execute
    code_key: str,
    env_id_key: str,
    base_path: Optional[str] = None,
    why_bad = False,
    **kwargs,
):
    
    def _func(data, config={}):
        code: List[str] = data[code_key]
        if isinstance(code, list) and len(code) == 1 and isinstance(code[0], list):
            code = code[0]

        result = {}
        result[key] = []
        for _c in code:
            if isinstance(_c, list) and len(_c) == 1 and isinstance(_c[0], str):
                _c = _c[0]
            
            venv_name = data[env_id_key]
            assert isinstance(venv_name, str), f"env_id should be str, but got {type(venv_name)}, {data}"
            py_exec = get_python_executable(base_path, venv_name)
            try:
                assert py_exec is not None
                assert os.path.exists(py_exec)
            except Exception as e:
                print(f"Error: venv not found, skipping sample {venv_name}...", e)
                if why_bad:
                    result[key].extend([1, ""])
                else:
                    result[key].append(1)

            _r = []
            try:
                output = subprocess.run(
                    [py_exec, "-c", _c],
                    text=True,
                    capture_output=True,
                    timeout=60
                )
                #TODO: check with testcase. this version: only check syntax error
                if why_bad:
                    _r.extend([output.returncode, output.stderr])
                else:
                    _r.append(output.returncode)
            except subprocess.TimeoutExpired as e:
                print(f"Error: Timeout expired, skipping sample {venv_name}...", e)
                if why_bad:
                    _r.extend([0, ""])
                    # result[key].extend([0, ""])
                else:
                    # result[key].append(0)
                    _r.append(0)
            result[key].append(_r)
        if isinstance(data[code_key], list) and len(data[code_key]) == 1 and isinstance(data[code_key][0], list):
            result[key] = [result[key]]

        return result

    return RunnableLambda(_func, name="execute_w_venv")

