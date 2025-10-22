import logging
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models import init_chat_model
from langchain.schema import BaseMessage, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def truncate_messages(
    messages: List[BaseMessage], max_chars: int = 50000
) -> List[BaseMessage]:
    """
    Truncate messages to fit within a character limit while preserving structure.

    Strategy:
    1. Always keep the system message (first message) intact
    2. Keep the most recent user message intact
    3. Truncate middle messages if needed

    Args:
        messages: List of messages to truncate
        max_chars: Maximum total characters allowed

    Returns:
        Truncated list of messages
    """
    if not messages:
        return messages

    # Calculate total length
    total_chars = sum(len(str(msg.content)) for msg in messages)

    if total_chars <= max_chars:
        return messages

    logging.warning(
        f"Messages exceed {max_chars} chars ({total_chars} chars). Truncating..."
    )

    # Always keep first (system) and last (current user query) messages
    if len(messages) <= 2:
        # If only 2 messages, truncate the content of the first one if needed
        truncated = []
        remaining_chars = max_chars

        for i, msg in enumerate(messages):
            content = str(msg.content)
            if len(content) > remaining_chars:
                if i == 0:
                    # Truncate system message from the middle
                    truncated_content = (
                        content[: remaining_chars // 2]
                        + "\n...[truncated]...\n"
                        + content[-remaining_chars // 2 :]
                    )
                else:
                    # Truncate from the beginning for user messages
                    truncated_content = (
                        "...[truncated]...\n" + content[-remaining_chars:]
                    )

                msg_copy = msg.__class__(content=truncated_content)
                truncated.append(msg_copy)
                remaining_chars = 0
            else:
                truncated.append(msg)
                remaining_chars -= len(content)

        return truncated

    # Keep first and last messages, truncate the middle ones
    first_msg = messages[0]
    last_msg = messages[-1]
    middle_msgs = messages[1:-1]

    first_chars = len(str(first_msg.content))
    last_chars = len(str(last_msg.content))

    # Reserve space for first and last messages
    available_for_middle = max_chars - first_chars - last_chars - 100  # 100 char buffer

    if available_for_middle <= 0:
        # Not enough space, truncate first message
        max_first_chars = max_chars * 2 // 3
        max_last_chars = max_chars - max_first_chars - 100

        first_content = str(first_msg.content)
        if len(first_content) > max_first_chars:
            truncated_first = (
                first_content[: max_first_chars // 2]
                + "\n...[truncated]...\n"
                + first_content[-max_first_chars // 2 :]
            )
            first_msg = first_msg.__class__(content=truncated_first)

        last_content = str(last_msg.content)
        if len(last_content) > max_last_chars:
            truncated_last = "...[truncated]...\n" + last_content[-max_last_chars:]
            last_msg = last_msg.__class__(content=truncated_last)

        return [first_msg, last_msg]

    # Truncate middle messages proportionally
    middle_chars = sum(len(str(msg.content)) for msg in middle_msgs)

    if middle_chars <= available_for_middle:
        return messages

    # Keep only the most recent middle messages that fit
    truncated_middle = []
    current_chars = 0

    for msg in reversed(middle_msgs):
        msg_chars = len(str(msg.content))
        if current_chars + msg_chars <= available_for_middle:
            truncated_middle.insert(0, msg)
            current_chars += msg_chars
        else:
            # Add a truncated version of this message if there's space
            if available_for_middle - current_chars > 500:
                remaining_space = available_for_middle - current_chars
                content = str(msg.content)
                truncated_content = "...[truncated]...\n" + content[-remaining_space:]
                truncated_middle.insert(0, msg.__class__(content=truncated_content))
            break

    result = [first_msg] + truncated_middle + [last_msg]
    logging.info(f"Truncated messages from {len(messages)} to {len(result)} messages")

    return result


class GeneralChatModel(BaseChatModel):
    """
    A unified chat model wrapper that supports multiple LLM providers.

    Uses LangChain's init_chat_model for standard providers (OpenAI, Azure, Ollama, Anthropic)
    and falls back to custom LLMProviderFactory for specialized providers (vLLM, Snowflake, Open WebUI).

    This provides automatic payload error handling, retry logic, and message truncation.
    """

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
    timeout: Optional[float] = None  # Timeout in seconds

    @property
    def _llm_type(self) -> str:
        return self.llm._llm_type

    @property
    def llm(self):
        """Get the LLM instance using init_chat_model where applicable."""
        try:
            # Map platform names to init_chat_model format (provider:model or just model)
            platform_map = {
                "openai": "openai",
                "azure": "azure_openai",
                "ollama": "ollama",
                "anthropic": "anthropic",
            }

            # For platforms supported by init_chat_model, use it for standardization
            if self.platform in platform_map:
                model_string = f"{platform_map[self.platform]}:{self.model}"

                # Build kwargs - init_chat_model passes through provider-specific params
                kwargs = {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "max_retries": self.max_retries,
                    "top_p": self.top_p,
                }

                # Add optional parameters
                if self.timeout is not None:
                    kwargs["timeout"] = self.timeout
                if self.base_url is not None:
                    kwargs["base_url"] = self.base_url
                if self.stop is not None:
                    kwargs["stop"] = self.stop
                if self.extra_body is not None:
                    kwargs["extra_body"] = self.extra_body

                # Platform-specific parameters
                if self.platform == "ollama" and self.num_ctx is not None:
                    kwargs["num_ctx"] = self.num_ctx

                return init_chat_model(model_string, **kwargs)
            else:
                # For custom platforms (vllm, snowflake, open_webui), fall back to custom factory
                # These require special handling not supported by init_chat_model
                from ...providers import LLMProviderFactory

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
                    timeout=self.timeout,
                )
        except Exception as e:
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

        # Track if we've already truncated once
        attempt_count = 0
        current_messages = messages

        # Preemptive truncation to prevent automatic context window truncation
        # This applies to any platform when num_ctx is specified
        if self.num_ctx:
            # Calculate approximate token count (rough estimate: 1 token ≈ 4 characters)
            total_chars = sum(len(str(msg.content)) for msg in current_messages)
            approximate_tokens = total_chars // 4

            # If we're close to or exceeding the context window, preemptively truncate
            # Use 80% of context window as safety margin (reserve space for output)
            max_input_tokens = int(self.num_ctx * 0.8)

            if approximate_tokens > max_input_tokens:
                max_chars = max_input_tokens * 4  # Convert back to characters
                logging.warning(
                    f"Context window prevention: {approximate_tokens} tokens exceeds "
                    f"{max_input_tokens} tokens limit (80% of {self.num_ctx}). "
                    f"Preemptively truncating to {max_chars} chars..."
                )
                current_messages = truncate_messages(
                    current_messages, max_chars=max_chars
                )

                # Log the truncation result
                new_total_chars = sum(len(str(msg.content)) for msg in current_messages)
                new_approximate_tokens = new_total_chars // 4
                logging.info(
                    f"After truncation: {new_approximate_tokens} tokens ({new_total_chars} chars)"
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
            nonlocal attempt_count, current_messages
            attempt_count += 1

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
                    messages=current_messages,
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
                elif "max tokens" in error_msg or "maximum context length" in error_msg:
                    logging.error(f"Max tokens exceeded: {e}")
                    raise e  # Don't retry max tokens errors
                elif "out of memory" in error_msg:
                    logging.warning(f"Out of memory error, will retry: {e}")
                    raise e  # Let tenacity handle the retry
                else:
                    logging.warning(f"Unexpected ValueError, will retry: {e}")
                    raise e
            except Exception as e:
                error_msg = str(e).lower()

                # Check for rate limit errors (429) - handle with infinite retries
                if any(
                    phrase in error_msg
                    for phrase in [
                        "429",
                        "rate limit",
                        "too many requests",
                        "quota exceeded",
                    ]
                ):
                    import time

                    wait_time = min(
                        60, 2**attempt_count
                    )  # Exponential backoff, max 60s
                    logging.warning(
                        f"Rate limit error detected (429). "
                        f"Retry attempt {attempt_count}. "
                        f"Waiting {wait_time}s before retrying..."
                    )
                    time.sleep(wait_time)
                    # Don't raise - let tenacity retry infinitely for rate limits
                    # by resetting the attempt counter
                    attempt_count = 0
                    raise e

                # Check for payload too large errors
                if any(
                    phrase in error_msg
                    for phrase in [
                        "payload too large",
                        "request entity too large",
                        "413",
                        "context length exceeded",
                        "request too large",
                    ]
                ):
                    if attempt_count <= 3:  # Try truncating up to 3 times
                        # Progressive truncation: reduce by more each attempt
                        max_chars = 50000 // attempt_count
                        logging.warning(
                            f"Payload too large error detected (attempt {attempt_count}). "
                            f"Truncating messages to {max_chars} chars..."
                        )
                        current_messages = truncate_messages(
                            current_messages, max_chars=max_chars
                        )
                        raise e  # Let tenacity retry with truncated messages
                    else:
                        logging.error(
                            f"Payload too large even after {attempt_count} truncation attempts"
                        )
                        raise e

                logging.warning(f"Unexpected error, will retry: {e}")
                raise e

        return _generate_with_retry()
