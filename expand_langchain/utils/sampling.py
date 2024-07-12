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
                        assert len(v) == 1
                        _result[k] = v[0]
                else:
                    _result[k] = v
            result.append(_result)

        return result

    async def parallel_run(inputs: list, config: RunnableConfig):
        batch = []
        for input in inputs:
            for _ in range(n):
                batch.append(input)

        response = await chain.abatch(batch, config)

        if isinstance(response[0], dict):
            result = {}
            for k in response[0].keys():
                result[k] = [r[k] for r in response]
                if flatten and isinstance(result[k][0], list):
                    result[k] = [item for sublist in result[k] for item in sublist]

            return result
        else:
            return response

    result = RunnableLambda(divide_dict) | RunnableLambda(parallel_run)
    result.name = "sampling_chain"

    return result
