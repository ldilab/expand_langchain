import logging
import os
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult
from langchain_community.chat_models import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from expand_langchain.utils.registry import model_registry


@model_registry(name="chat")
class GeneralChatModel(BaseChatModel):
    model: Optional[str] = None
    max_tokens: int
    temperature: float
    top_p: float
    max_retries: int = 10000
    platform: str = "azure"
    stop: Optional[List[str]] = None

    llm: BaseChatModel = None

    @property
    def _llm_type(self) -> str:
        return self.llm._llm_type

    @property
    def llm(self):
        if self.platform == "azure":
            return AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_ENDPOINT"],
                api_version=os.environ["AZURE_API_VERSION"],
                api_key=os.environ["AZURE_API_KEY"],
                azure_deployment=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model_kwargs={"top_p": self.top_p},
                max_retries=self.max_retries,
            )

        elif self.platform == "openai":
            return ChatOpenAI(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model_kwargs={"top_p": self.top_p},
                max_retries=self.max_retries,
            )

        elif self.platform == "open_webui":
            return ChatOllama(
                model=self.model,
                num_ctx=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                base_url=os.environ["OPEN_WEBUI_BASE_URL"],
                headers={
                    "Authorization": f"Bearer {os.environ['OPEN_WEBUI_API_KEY']}",
                    "Content-Type": "application/json",
                },
            )

        else:
            raise ValueError(f"platform {self.platform} not supported")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[str] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return self.llm._generate(
                messages=messages,
                stop=stop + self.stop if stop is not None else self.stop,
                run_manager=run_manager,
                **kwargs,
            )
        except ValueError as e:
            if "content filter" in str(e):
                logging.error(f"content filter triggered")
                raise e
            elif "out of memory" in str(e):
                logging.error(f"out of memory")
                logging.error("retrying...")
                self.llm._generate(
                    messages=messages,
                    stop=stop + self.stop if stop is not None else self.stop,
                    run_manager=run_manager,
                    **kwargs,
                )
            else:
                raise e