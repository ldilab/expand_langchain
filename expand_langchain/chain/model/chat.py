import logging
from operator import itemgetter
from typing import Any, List, Literal, Optional, Type, Union, cast

try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = None  # type: ignore

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if an exception is a rate limit error.

    Handles both OpenAI RateLimitError and error messages containing rate limit indicators.
    """
    # Check OpenAI RateLimitError type
    if OpenAIRateLimitError and isinstance(exception, OpenAIRateLimitError):
        return True

    # Check error message for rate limit indicators
    error_msg = str(exception).lower()
    rate_limit_indicators = [
        "rate limit",
        "ratelimit",
        "429",
        "too many requests",
        "quota exceeded",
        "no deployments available",
    ]
    return any(indicator in error_msg for indicator in rate_limit_indicators)


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

        # Create a custom retry function that handles rate limits with infinite retry
        def _generate_with_smart_retry():
            nonlocal attempt_count, current_messages
            rate_limit_attempt = 0
            other_error_attempt = 0

            while True:
                try:
                    attempt_count += 1

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
                        raise ValueError(
                            "LLM returned empty result with no generations"
                        )

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

                except Exception as e:
                    error_msg = str(e).lower()

                    # Check if this is a rate limit error
                    if is_rate_limit_error(e):
                        rate_limit_attempt += 1
                        # Exponential backoff: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, max 600 seconds
                        wait_time = min(600, 2**rate_limit_attempt)
                        logging.warning(
                            f"Rate limit error detected. "
                            f"Retry attempt {rate_limit_attempt} (infinite retry). "
                            f"Waiting {wait_time}s before retrying..."
                        )
                        import time

                        time.sleep(wait_time)
                        continue  # Retry infinitely for rate limits

                    # Handle other errors with limited retry
                    other_error_attempt += 1

                    # Content filter errors - don't retry
                    if "content filter" in error_msg:
                        logging.error(f"Content filter triggered: {e}")
                        raise

                    # Max tokens errors - don't retry
                    if (
                        "max tokens" in error_msg
                        or "maximum context length" in error_msg
                    ):
                        logging.error(f"Max tokens exceeded: {e}")
                        raise

                    # Payload too large errors - try truncating
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
                        if other_error_attempt <= 3:  # Try truncating up to 3 times
                            # Progressive truncation: reduce by more each attempt
                            max_chars = 50000 // other_error_attempt
                            logging.warning(
                                f"Payload too large error detected (attempt {other_error_attempt}). "
                                f"Truncating messages to {max_chars} chars..."
                            )
                            current_messages = truncate_messages(
                                current_messages, max_chars=max_chars
                            )
                            continue  # Retry with truncated messages
                        else:
                            logging.error(
                                f"Payload still too large after {other_error_attempt} truncation attempts"
                            )
                            raise

                    # Other errors - limited retry (max 10 attempts)
                    if other_error_attempt >= 10:
                        logging.error(
                            f"Max retry attempts ({other_error_attempt}) reached for non-rate-limit error"
                        )
                        raise

                    # Exponential backoff for other errors
                    wait_time = min(60, 2**other_error_attempt)
                    logging.warning(
                        f"Error encountered (attempt {other_error_attempt}/10): {e}. "
                        f"Waiting {wait_time}s before retrying..."
                    )
                    import time

                    time.sleep(wait_time)
                    continue

        return _generate_with_smart_retry()

    def with_structured_output(
        self,
        schema: Union[Type[BaseModel], dict, None] = None,
        *,
        method: Literal["json_mode"] = "json_mode",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        This implementation adds format instructions to the system message,
        making it compatible with non-OpenAI models that don't support response_format.

        Args:
            schema: The output schema. Can be:
                - A Pydantic BaseModel class
                - A dictionary (JSON schema)
                - None (returns raw JSON)

                If schema is a Pydantic class, the output will be a Pydantic instance.
                Otherwise, output will be a dict.

            method: Currently only supports "json_mode". This will:
                - Add format instructions to the system message
                - Use PydanticOutputParser or JsonOutputParser to parse the output

            include_raw: If False, only the parsed output is returned.
                If True, returns a dict with keys:
                - 'raw': BaseMessage
                - 'parsed': Parsed output (or None if parsing error)
                - 'parsing_error': Exception or None

            **kwargs: Additional arguments passed to the model.

        Returns:
            A Runnable that takes same inputs as BaseChatModel and outputs:
            - If include_raw=False: Pydantic instance or dict
            - If include_raw=True: dict with 'raw', 'parsed', 'parsing_error' keys

        Example:
            ```python
            from pydantic import BaseModel, Field

            class Answer(BaseModel):
                answer: str
                justification: str

            llm = GeneralChatModel(model="gpt-4", platform="openai", ...)
            structured_llm = llm.with_structured_output(Answer)
            result = structured_llm.invoke("What is 2+2?")
            # result is an Answer instance
            ```
        """
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)

        if method != "json_mode":
            msg = (
                f"Method '{method}' is not supported. "
                f"GeneralChatModel currently only supports method='json_mode'."
            )
            raise ValueError(msg)

        # Create appropriate output parser
        if is_pydantic_schema:
            output_parser: Runnable = PydanticOutputParser(
                pydantic_object=cast(Type[BaseModel], schema)
            )
        elif schema is None:
            output_parser = JsonOutputParser()
        else:
            # For dict/JSON schema, use JsonOutputParser
            output_parser = JsonOutputParser()

        # Get format instructions from the parser
        format_instructions = output_parser.get_format_instructions()
        format_instructions = f"""\
# Output Format
First, reason through the problem step-by-step in natural language.
Explain your thinking process, analysis, and any intermediate steps needed to arrive at the final answer.
After your reasoning, provide the final structured output wrapped in three backticks (```).
Make sure to provide exactly ONE structured output in a SINGLE code block at the very end of your response.
Do not include multiple code blocks or multiple structured outputs.
The structured output must be the final element of your response.
{format_instructions}
"""

        # Create a wrapper that adds format instructions to system message
        def add_format_instructions_to_messages(
            messages: Union[List[BaseMessage], LanguageModelInput],
        ) -> List[BaseMessage]:
            """Add format instructions after the system message."""
            from langchain_core.messages import SystemMessage

            # Convert input to messages if needed
            if not isinstance(messages, list):
                # Handle string or other input types
                if isinstance(messages, str):
                    messages = [SystemMessage(content=messages)]
                elif hasattr(messages, "to_messages"):
                    messages = messages.to_messages()  # type: ignore
                else:
                    # Try to convert using BaseMessage
                    messages = [messages] if isinstance(messages, BaseMessage) else list(messages)  # type: ignore

            if not messages:
                return [SystemMessage(content=format_instructions)]

            # Find the last system message
            result_messages = []
            system_message_found = False

            for i, msg in enumerate(messages):
                if isinstance(msg, SystemMessage) and not system_message_found:
                    # Add format instructions to the first system message
                    new_content = f"{msg.content}\n\n{format_instructions}"
                    result_messages.append(SystemMessage(content=new_content))
                    system_message_found = True
                else:
                    result_messages.append(msg)

            # If no system message found, add one at the beginning
            if not system_message_found:
                result_messages.insert(0, SystemMessage(content=format_instructions))

            return result_messages

        # Create a runnable that preprocesses messages
        preprocessor = RunnableLambda(add_format_instructions_to_messages)

        # Chain: preprocess -> GeneralChatModel (with retry logic) -> parse
        # Use self (GeneralChatModel) instead of self.llm to preserve retry logic
        llm_chain = preprocessor | self

        if include_raw:
            # Return dict with raw, parsed, and parsing_error keys
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm_chain) | parser_with_fallback
        else:
            # Return only parsed output
            return llm_chain | output_parser
