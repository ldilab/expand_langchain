from typing import Any, Dict, List, Optional

from expand_langchain.utils.registry import (
    chain_registry,
    model_registry,
    prompt_registry,
)
from langchain_core.output_parsers import StrOutputParser

from pydantic import BaseModel, Field, create_model

@chain_registry(name="llm")
def llm_chain(
    prompt: dict,
    examples: Optional[List[Dict[str, str]]] = None,
    llm: Dict[str, Any] = {},
    disable_icl: bool = False,
    **kwargs,
):
    prompt_type = prompt["type"]
    prompt_kwargs = prompt["kwargs"]

    prompt = prompt_registry[prompt_type](
        examples=examples if not disable_icl else None,
        **prompt_kwargs,
    )

    model = model_registry[prompt_type](**llm)

    # if "structured_output" in kwargs:
    #     structured_output = kwargs["structured_output"]
    #     raw_fields = structured_output["fields"]
    #     print(raw_fields)
    #     fields = {
    #         name: (
    #             kvs["type"],
    #             Field(description=kvs["description"])
    #         )
    #         for field in raw_fields
    #         for name, kvs in field.items()
    #     }
    #
    #     structured_output_class_obj = create_model(
    #         "ResponseFormatter",      # name of the generated class
    #         __base__=BaseModel,       # inherit from BaseModel
    #         **fields                  # unpack your field definitions
    #     )
    #     model = model.with_structured_output(
    #         structured_output_class_obj,
    #     )
    #     result = prompt | model
    #
    # else:
    result = prompt | model | StrOutputParser()
    result.name = "llm_chain"

    return result
