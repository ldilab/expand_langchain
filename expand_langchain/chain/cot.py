from typing import List, Optional

from langchain_core.runnables import RunnableLambda

from expand_langchain.chain.llm import llm_chain
from expand_langchain.utils.parser import parser_chain
from expand_langchain.utils.registry import chain_registry
from expand_langchain.utils.sampling import sampling_chain


@chain_registry(name="cot")
def cot_chain(
    key: str,
    examples: Optional[dict] = None,
    n=1,
    **kwargs,
):
    async def _func(data):
        chain = llm_chain(examples=list(examples.values()), **kwargs)
        parser = parser_chain(**kwargs)

        result = await chain.ainvoke(data)
        parsed_result = parser.invoke(result)

        return {
            f"{key}_raw": result,
            key: parsed_result,
        }

    chain = RunnableLambda(_func)

    result = sampling_chain(chain, n, **kwargs)
    result.name = key

    return result
