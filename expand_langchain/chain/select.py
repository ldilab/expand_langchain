from typing import Any, List

from langchain_core.runnables import RunnableLambda

from expand_langchain.utils.registry import chain_registry


@chain_registry(name="select")
def select_chain(
    src: List[str],
    dst: List[str],
    tgt: str,
    func: str,
    **kwargs,
):
    func = eval(func)

    def _func(data):
        result = {}
        for d in dst:
            result[d] = []

        for i in range(len(data[tgt])):
            if func(data[tgt][i]):
                for s, d in zip(src, dst):
                    result[d].append(data[s][i])

        return result

    return RunnableLambda(_func, name="select")
