import logging
import os
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import BaseMessage, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .custom_api.snowflake import ChatSnowflakeCortex


class GeneralChatModel(BaseChatModel):
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    num_ctx: Optional[int] = None
    max_retries: int = 10
    platform: str
    stop: Optional[List[str]] = None
    base_url: Optional[str] = None
    extra_body: Optional[dict] = None

    @property
    def _llm_type(self) -> str:
        return self.llm._llm_type

    @property
    def llm(self):
        if self.platform == "azure":
            return AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_ENDPOINT"],
                api_version=os.environ["AZURE_API_VERSION"],
                api_key=SecretStr(os.environ["AZURE_API_KEY"]),
                azure_deployment=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model_kwargs={"top_p": self.top_p},
                max_retries=self.max_retries,
            )
        elif self.platform == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            return ChatOpenAI(
                api_key=SecretStr(api_key) if api_key else None,
                model=self.model,
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                max_retries=self.max_retries,
                base_url=os.environ.get("OPENAI_API_BASE", None),
                extra_body=self.extra_body,
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
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Debug: Log the input messages before sending to LLM
        logging.debug(f"Input messages to LLM:")
        for i, msg in enumerate(messages):
            logging.debug(
                f"  Message {i}: {type(msg).__name__} - {repr(msg.content[:200])}"
            )

        # Create a retry decorator with exponential backoff
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((ValueError, Exception)),
            reraise=True,
        )

        @retry_decorator
        def _generate_with_retry():
            try:
                # Combine stop sequences properly
                combined_stop = None
                if stop is not None and self.stop is not None:
                    combined_stop = stop + self.stop
                elif stop is not None:
                    combined_stop = stop
                elif self.stop is not None:
                    combined_stop = self.stop

                result = self.llm._generate(
                    messages=messages,
                    stop=combined_stop,
                    run_manager=run_manager,
                    **kwargs,
                )

                # Validate that the result contains meaningful content
                if not result or not result.generations:
                    raise ValueError("LLM returned empty result with no generations")

                # Check finish_reason for each generation
                for gen in result.generations:
                    generation_info = getattr(gen, "generation_info", {})
                    reason = generation_info.get(
                        "finish_reason"
                    ) or generation_info.get("done_reason")

                    # Check if we have a valid reason
                    if reason is not None:
                        if reason != "stop":
                            logging.error(
                                f"LLM generation failed with reason: {reason}"
                            )
                            raise ValueError(
                                f"LLM generation failed with reason: {reason}"
                            )
                    else:
                        # No finish/done reason found, check if we have generated content
                        content = ""
                        try:
                            # Try to get content from message attribute first
                            message = getattr(gen, "message", None)
                            if message:
                                content = str(getattr(message, "content", ""))
                            # If no message content, try text attribute
                            if not content:
                                content = str(getattr(gen, "text", ""))
                        except (AttributeError, TypeError):
                            pass

                        # If no content found, raise error
                        if not content or not content.strip():
                            logging.error(
                                "LLM generation has no finish/done reason and no content"
                            )
                            raise ValueError(
                                "LLM generation has no finish/done reason and no content"
                            )

                return result
            except ValueError as e:
                error_msg = str(e).lower()
                if "content filter" in error_msg:
                    logging.error(f"Content filter triggered: {e}")
                    raise e  # Don't retry content filter errors
                elif "max tokens" in error_msg:
                    logging.error(f"Max tokens exceeded: {e}")
                    raise e  # Don't retry max tokens errors
                elif "out of memory" in error_msg:
                    logging.warning(f"Out of memory error, will retry: {e}")
                    raise e  # Let tenacity handle the retry
                else:
                    logging.warning(f"Unexpected ValueError, will retry: {e}")
                    raise e
            except Exception as e:
                logging.warning(f"Unexpected error, will retry: {e}")
                raise e

        return _generate_with_retry()
