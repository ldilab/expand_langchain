import logging
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import BaseMessage, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...providers import LLMProviderError, LLMProviderFactory


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
        """Get the LLM instance using the factory pattern."""
        try:
            return LLMProviderFactory.create_chat_model(
                platform=self.platform,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                num_ctx=self.num_ctx,
                max_retries=self.max_retries,
                base_url=self.base_url,
                extra_body=self.extra_body,
            )
        except LLMProviderError as e:
            raise ValueError(f"Failed to create {self.platform} chat model: {e}") from e

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
