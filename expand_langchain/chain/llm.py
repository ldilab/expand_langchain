from typing import Any, Dict, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.runnables import Runnable, RunnableSerializable
from pydantic import Field, PrivateAttr

from .model.chat import GeneralChatModel
from .prompt.chat import CustomChatPromptTemplate


class LLMChain(RunnableSerializable):
    key: str
    parser: Runnable

    chat_history_key: Optional[str] = Field(default=None)
    chat_history_len: int = Field(default=0)

    _prompt: CustomChatPromptTemplate = PrivateAttr()
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

        self._prompt = CustomChatPromptTemplate.from_custom_config(
            chat_history_key=self.chat_history_key,
            chat_history_len=self.chat_history_len,
            **prompt_kwargs,
        )
        self._model = GeneralChatModel(**llm_kwargs)
        self.name = "llm_chain"

        self._chain = self._prompt | self._model | StrOutputParser()

    def invoke(self, input, config=None, **kwargs):
        result = self._chain.invoke(input, config, **kwargs)
        chat_history = self._prompt.invoke(input, config, **kwargs)

        output = {
            f"{self.key}_raw": result,
            self.key: self.parser.invoke(result, config, **kwargs),
        }

        if isinstance(self.chat_history_key, str):
            output[self.chat_history_key] = chat_history.to_messages()

        return output

    async def ainvoke(self, input, config=None, **kwargs):
        result = await self._chain.ainvoke(input, config, **kwargs)
        chat_history = await self._prompt.ainvoke(input, config, **kwargs)

        output = {
            f"{self.key}_raw": result,
            self.key: await self.parser.ainvoke(result, config, **kwargs),
        }

        if isinstance(self.chat_history_key, str):
            output[self.chat_history_key] = chat_history.to_messages()

        return output
