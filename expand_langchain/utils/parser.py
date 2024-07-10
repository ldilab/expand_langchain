from typing import List

from langchain_core.runnables import RunnablePassthrough

from expand_langchain.utils.registry import parser_registry


def parser_chain(
    parsers: List[dict] = [],
    **kwargs,
):
    chain = RunnablePassthrough()
    for parser in parsers:
        type = parser["type"]
        parser_kwargs = parser.get("kwargs", {})

        parser = parser_registry[type](**parser_kwargs)
        chain = chain | parser

    chain.name = "parser_chain"
    return chain
