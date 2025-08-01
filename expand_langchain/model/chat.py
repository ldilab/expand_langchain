import logging
import os
from typing import Any, List, Optional

from expand_langchain.model.custom_api.snowflake import ChatSnowflakeCortex
from expand_langchain.utils.registry import model_registry
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import BaseMessage, ChatResult
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI


@model_registry(name="chat")
class GeneralChatModel(BaseChatModel):
    model: Optional[str] = None
    max_tokens: int
    temperature: float
    top_p: float
    num_ctx: Optional[int] = None
    max_retries: int = 10
    platform: str
    stop: Optional[List[str]] = None
    base_url: Optional[str] = None

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
                top_p=self.top_p,
                max_retries=self.max_retries,
            )

        elif self.platform == "open_webui":
            return ChatOllama(
                model=self.model,
                num_predict=self.max_tokens,
                num_ctx=self.num_ctx,
                temperature=self.temperature,
                top_p=self.top_p,
                base_url=os.environ["OPEN_WEBUI_BASE_URL"],
                headers={
                    "Authorization": f"Bearer {os.environ['OPEN_WEBUI_API_KEY']}",
                    "Content-Type": "application/json",
                },
            )

        elif self.platform == "ollama":
            return ChatOllama(
                model=self.model,
                num_predict=self.max_tokens,
                num_ctx=self.num_ctx,
                temperature=self.temperature,
                top_p=self.top_p,
                base_url=self.base_url or os.environ["OLLAMA_BASE_URL"],
                headers={
                    "Content-Type": "application/json",
                },
            )

        elif self.platform == "vllm":
            return ChatOpenAI(
                openai_api_key=os.environ.get("VLLM_API_KEY"),
                openai_api_base=os.environ["VLLM_BASE_URL"],
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                max_retries=self.max_retries,
            )

        elif self.platform == "snowflake":
            return ChatSnowflakeCortex(
                model=self.model,
                cortex_function="complete",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                account=os.environ.get("SNOWFLAKE_ACCOUNT"),
                username=os.environ.get("SNOWFLAKE_USERNAME"),
                password=os.environ.get("SNOWFLAKE_PASSWORD"),
                database=os.environ.get("SNOWFLAKE_DATABASE"),
                schema=os.environ.get("SNOWFLAKE_SCHEMA"),
                role=os.environ.get("SNOWFLAKE_ROLE"),
                warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
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
        # Debug: Log the input messages before sending to LLM
        logging.debug(f"Input messages to LLM:")
        for i, msg in enumerate(messages):
            logging.debug(f"  Message {i}: {type(msg).__name__} - {repr(msg.content[:200])}")
        
        try:
            result = self.llm._generate(
                messages=messages,
                stop=stop + self.stop if stop is not None else self.stop,
                run_manager=run_manager,
                **kwargs,
            )
            return result
        except ValueError as e:
            if "content filter" in str(e):
                logging.error(f"content filter triggered")
                raise e
            elif "max tokens" in str(e):
                logging.error(f"max tokens exceeded")
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
