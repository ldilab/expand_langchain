import traceback
from operator import itemgetter
from typing import Any, List, Optional

from expand_langchain.chain.llm import llm_chain
from expand_langchain.utils.parser import parser_chain
from expand_langchain.utils.registry import chain_registry
from expand_langchain.utils.sampling import sampling_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langfuse.decorators import langfuse_context, observe


@chain_registry(name="cot")
def cot_chain(
    key: str,
    examples: Optional[dict] = None,
    n: int = 1,
    chat_history_len: int = 0,
    chat_history_key: str = "chat_history",
    **kwargs,
):

    async def _func(data, config={}):
        chain = llm_chain(
            examples=list(examples.values()),
            chat_history_len=chat_history_len,
            chat_history_key=chat_history_key,
            **kwargs,
        )

        if chat_history_len > 0:
            chat_history = data.get(chat_history_key)
            chat_history = [] if not chat_history else chat_history[0]
            data[chat_history_key] = chat_history
            if len(chat_history) > chat_history_len:
                data[chat_history_key] = chat_history[-chat_history_len:]
            response = await chain.ainvoke(data, config=config)

            chat_history.append(chain.steps[0].messages[-1].format(**data))
            chat_history.append(AIMessage(response))

            parser = parser_chain(**kwargs)
            parsed_result = parser.invoke(response, config=config)

            return {
                f"{key}_raw": response,
                key: parsed_result,
                chat_history_key: [chat_history],
            }
        else:
            response = await chain.ainvoke(data, config=config)

            parser = parser_chain(**kwargs)
            parsed_result = parser.invoke(response, config=config)

            return {
                f"{key}_raw": response,
                key: parsed_result,
            }

    chain = RunnableLambda(_func)

    result = sampling_chain(chain, n, **kwargs)
    result.name = key

    return result
