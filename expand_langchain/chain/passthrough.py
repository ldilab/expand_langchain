from langchain_core.runnables import RunnablePassthrough

from expand_langchain.utils.custom_trace import traceable
from expand_langchain.utils.registry import chain_registry


@chain_registry(name="passthrough")
def passthrough_chain(**kwargs):
    return RunnablePassthrough()
