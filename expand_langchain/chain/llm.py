from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

from expand_langchain.utils.registry import chain_registry, model_registry, prompt_registry


@chain_registry(name="llm")
def llm_chain(
    prompt: dict,
    examples: Optional[List[Dict[str, str]]] = None,
    llm: Dict[str, Any] = {},
    **kwargs,
):
    prompt_type = prompt["type"]
    prompt_kwargs = prompt["kwargs"]

    prompt = prompt_registry[prompt_type](
        examples=examples,
        **prompt_kwargs,
    )

    model = model_registry[prompt_type](**llm)

    result = prompt | model | StrOutputParser()
    result.name = "llm_chain"

    return result
