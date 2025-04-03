from typing import Any, List

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda


def sampling_chain(
    chain: Runnable,
    n: int = 1,
    flatten: bool = False,
    **kwargs,
):
    def divide_dict(d: dict) -> List[dict]:
        input_keys = list(d.keys())

        # find max n
        max_n = 1
        for k in input_keys:
            v = d[k]
            if isinstance(v, list):
                max_n = max(max_n, len(v))

        # divide dict
        result = []
        for i in range(max_n):
            _result = {}
            for k in input_keys:
                v = d[k]
                if isinstance(v, list):
                    if len(v) == max_n:
                        _result[k] = v[i]
                    else:
                        if len(v) == 1:
                            _result[k] = v[0]
                        else:
                            _result[k] = v
                else:
                    _result[k] = v
            result.append(_result)

        return result

    async def parallel_run(inputs: list, config: RunnableConfig):
        if any(
            isinstance(v, list) and len(v) > 1
            for input in inputs
            for v in input.values()
        ):
            response = []
            for input in inputs:
                new_input = divide_dict(input)
                _response = await parallel_run(new_input, config)
                response.append(_response)

        else:
            batch = []
            for input in inputs:
                for _ in range(n):
                    batch.append(input)

            response = await chain.abatch(batch, config)

            if isinstance(response[0], dict):
                response = [
                    merge_dicts(response[i * n : (i + 1) * n])
                    for i in range(len(response) // n)
                ]
                if isinstance(response, list) and len(response) == 1:
                    response = response[0]

        if isinstance(response[0], dict):
            result = merge_dicts(response, flatten=flatten)

            return result
        else:
            return response

    result = RunnableLambda(divide_dict) | RunnableLambda(parallel_run)
    result.name = "sampling_chain"

    return result


def merge_dicts(dicts: List[dict], flatten=False) -> dict:
    result = {}
    for d in dicts:
        for k, v in d.items():
            if k not in result:
                result[k] = []
            result[k].append(v)

    if flatten:
        for k in result.keys():
            result[k] = [item for sublist in result[k] for item in sublist]

    return result
