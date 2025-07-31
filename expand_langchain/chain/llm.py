from typing import Any, Dict, List, Optional

from expand_langchain.utils.registry import (
    chain_registry,
    model_registry,
    prompt_registry,
)
from langchain_core.output_parsers import StrOutputParser


@chain_registry(name="llm")
def llm_chain(
    prompt: dict,
    examples: Optional[List[Dict[str, str]]] = None,
    llm: Dict[str, Any] = {},
    disable_icl: bool = False,
    chat_history_len: int = 0,
    chat_history_key: str = "chat_history",
    **kwargs,
):
    prompt_type = prompt["type"]
    prompt_kwargs = prompt["kwargs"]

    prompt = prompt_registry[prompt_type](
        examples=examples if not disable_icl else None,
        chat_history_len=chat_history_len,
        chat_history_key=chat_history_key,
        **prompt_kwargs,
    )

    model = model_registry[prompt_type](**llm)

    result = prompt | model | StrOutputParser()
    result.name = "llm_chain"

    return result
