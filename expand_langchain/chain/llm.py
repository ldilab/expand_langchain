from typing import Any, Dict, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableSerializable
from pydantic import Field, PrivateAttr

from .model.chat import GeneralChatModel
from .prompt.chat import chat_prompt


class LLMChain(RunnableSerializable):
    _prompt: ChatPromptTemplate = PrivateAttr()
    _model: GeneralChatModel = PrivateAttr()
    _chain: Runnable = PrivateAttr()

    # pydantic
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        prompt_kwargs: Dict[str, Any],
        llm_kwargs: Dict[str, Any],
        **data,
    ):
        super().__init__(**data)

        self._prompt = chat_prompt(**prompt_kwargs)
        self._model = GeneralChatModel(**llm_kwargs)
        self.name = "llm_chain"

        self._chain = self._prompt | self._model | StrOutputParser()

    def invoke(self, input, config=None, **kwargs):
        return self._chain.invoke(input, config, **kwargs)
