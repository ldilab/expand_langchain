import json
import logging
import re
from operator import itemgetter
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Literal, Optional, Type, Union, cast

try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = None  # type: ignore

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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


def _get_message_role(msg: BaseMessage) -> str:
    """Get the role of a message.

    Args:
        msg: Message to get role for

    Returns:
        Role string: 'user', 'assistant', or 'system'
    """
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    elif isinstance(msg, SystemMessage):
        return "system"
    else:
        # Default to user for unknown types
        return "user"


def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that may contain markdown, headers, or explanations.

    This function tries multiple strategies to extract valid JSON:
    1. Look for JSON inside ```json code blocks
    2. Look for JSON inside ``` code blocks (any language)
    3. Look for JSON object/array patterns in the text
    4. Return original text if no extraction works

    Args:
        text: Text that may contain JSON with additional markup

    Returns:
        Extracted JSON string (or original text if extraction fails)
    """
    # Strategy 1: Try to find JSON in ```json code blocks
    json_block_pattern = r"```json\s*\n(.*?)\n```"
    match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Strategy 1b: Try unclosed ```json code block at end of text
    unclosed_json_pattern = r"```json\s*\n([\s\S]+)$"
    match = re.search(unclosed_json_pattern, text)
    if match:
        candidate = match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass

    # Strategy 2: Try to find JSON in any ``` code blocks
    code_block_pattern = r"```\s*\n(.*?)\n```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        # Check if it looks like JSON
        if extracted.startswith(("{", "[")):
            return extracted

    # Strategy 3: Try to find JSON object/array patterns in the text
    # Find the first { and match it with the last }
    json_obj_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
    match = re.search(json_obj_pattern, text, re.DOTALL)
    if match:
        candidate = match.group(0)
        # Validate it's actually JSON by trying to parse
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: Try to find JSON array patterns
    json_arr_pattern = r"\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]"
    match = re.search(json_arr_pattern, text, re.DOTALL)
    if match:
        candidate = match.group(0)
        # Validate it's actually JSON by trying to parse
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    # If nothing worked, return original text
    return text


class LenientOutputParser:
    """Wrapper parser that extracts JSON from text before parsing.

    This wrapper allows headers or explanations around JSON as long as
    a valid JSON object/array is present (e.g., inside a code block).

    Args:
        base_parser: The underlying LangChain output parser.
    """

    def __init__(self, base_parser: Runnable) -> None:
        self.base_parser = base_parser

    def get_format_instructions(self) -> str:
        """Return format instructions from the base parser.

        Returns:
            Format instructions string.
        """
        return self.base_parser.get_format_instructions()

    def parse(self, text: str) -> Any:
        """Parse output after extracting JSON from the response text.

        Args:
            text: Raw LLM response text.

        Returns:
            Parsed output from the base parser.
        """
        self._validate_json_codeblock(text)
        normalized_text = extract_json_from_text(text)
        return self.base_parser.parse(normalized_text)

    @staticmethod
    def _validate_json_codeblock(text: str) -> None:
        """Require a single JSON code block as the final response.

        Accepts both properly closed (```json ... ```) and unclosed
        (```json ... EOF) code blocks where the JSON itself is complete.
        """
        stripped = (text or "").strip()
        if not stripped:
            return

        # Try closed code block first (allow closing ``` on same line as content)
        codeblock_pattern = r"```json\s*\n[\s\S]*?```"
        matches = list(re.finditer(codeblock_pattern, stripped))

        if len(matches) == 1:
            match = matches[0]
            before = stripped[: match.start()]
            after = stripped[match.end() :]

            if "```" in before or "```" in after:
                raise ValueError(
                    "CRITICAL: Only one JSON code block is allowed. Remove other code blocks."
                )

            if after.strip():
                raise ValueError(
                    "CRITICAL: JSON code block must be the final content (no text after)."
                )
            return

        # Accept unclosed code block at end of output (model omitted closing ```)
        unclosed_pattern = r"```json\s*\n([\s\S]+)$"
        unclosed_match = re.search(unclosed_pattern, stripped)
        if unclosed_match:
            before = stripped[: unclosed_match.start()]
            # No other code blocks allowed before
            if "```" in before:
                raise ValueError(
                    "CRITICAL: Only one JSON code block is allowed. Remove other code blocks."
                )
            # Validate that the unclosed block contains plausible JSON
            body = unclosed_match.group(1).strip()
            if body.startswith("{") and body.endswith("}"):
                return

        raise ValueError(
            "CRITICAL: Output must contain exactly one ```json ...``` code block."
        )


def validate_message_sequence(messages: List[BaseMessage]) -> None:
    """Validate message sequence for strict chat APIs (e.g., Snowflake Cortex).

    Requirements:
    1. Only ONE system message, and it must be the FIRST message
    2. After system message, roles must strictly alternate: user-assistant-user-assistant
    3. No consecutive messages with the same role

    Args:
        messages: List of messages to validate

    Raises:
        ValueError: If message sequence violates requirements
    """
    if not messages:
        return

    violations = []

    # Check 1: System message must be first (if present)
    if messages and not isinstance(messages[0], SystemMessage):
        violations.append("First message is not SystemMessage")

    # Check 2: Only one system message allowed
    system_msg_count = sum(1 for msg in messages if isinstance(msg, SystemMessage))
    if system_msg_count > 1:
        violations.append(f"Found {system_msg_count} system messages (only 1 allowed)")

    # Check 3: System message must be first (if exists)
    system_indices = [
        i for i, msg in enumerate(messages) if isinstance(msg, SystemMessage)
    ]
    if system_indices and system_indices[0] != 0:
        violations.append(
            f"System message is at index {system_indices[0]}, not at index 0"
        )

    # Check 4: Strict user-assistant alternation after system message
    start_idx = 1 if isinstance(messages[0], SystemMessage) else 0
    if start_idx < len(messages):
        expected_role = "user"
        for i in range(start_idx, len(messages)):
            msg_role = _get_message_role(messages[i])
            if msg_role != expected_role:
                violations.append(
                    f"Message {i}: expected '{expected_role}', got '{msg_role}' "
                    f"(role sequence broken)"
                )
                break
            expected_role = "assistant" if msg_role == "user" else "user"

    # Raise error if violations found
    if violations:
        violation_str = "; ".join(violations)
        msg_structure = [_get_message_role(m) for m in messages]
        error_msg = (
            f"Message sequence validation FAILED - {violation_str}\n"
            f"Message structure: {msg_structure}\n"
            f"This indicates a bug in chat history management. "
            f"Please check bootstrap flow and tool summary additions."
        )
        raise ValueError(error_msg)

    logging.debug(
        f"Message sequence validation passed: {[_get_message_role(m) for m in messages]}"
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


def reduce_messages_for_payload(
    messages: List[BaseMessage], max_chars: int
) -> List[BaseMessage]:
    """Aggressively reduce messages to fit a strict payload budget.

    Strategy:
    - Keep only system + last message
    - Truncate both to fit max_chars
    """
    if not messages:
        return messages

    system_msg = messages[0]
    last_msg = messages[-1]

    system_content = str(system_msg.content)
    last_content = str(last_msg.content)

    if max_chars < 200:
        max_chars = 200

    system_budget = max_chars // 3
    last_budget = max_chars - system_budget - 50

    if len(system_content) > system_budget:
        system_content = (
            system_content[: system_budget // 2]
            + "\n...[truncated]...\n"
            + system_content[-system_budget // 2 :]
        )

    if len(last_content) > last_budget:
        last_content = "...[truncated]...\n" + last_content[-last_budget:]

    return [
        system_msg.__class__(content=system_content),
        last_msg.__class__(content=last_content),
    ]


def load_default_templates() -> Dict[str, str]:
    """
    Load default format instruction templates from files.

    Templates are loaded from:
    - templates/system_format.txt
    - templates/retry_error_format.txt
    - templates/retry_hint_format.txt

    Returns:
        Dictionary with keys: 'system_format', 'retry_error_format', 'retry_hint_format'
    """
    template_dir = Path(__file__).parent / "templates"

    templates = {}
    template_names = [
        "system_format",
        "retry_error_format",
        "retry_hint_format",
        "structured_output_format",
    ]

    for name in template_names:
        template_path = template_dir / f"{name}.txt"
        with open(template_path, "r", encoding="utf-8") as f:
            templates[name] = f.read()

    return templates


# Load default templates at module level (cached globally)
DEFAULT_TEMPLATES = load_default_templates()


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
        # Validate message sequence before processing
        validate_message_sequence(messages)

        # Debug: Log the input messages before sending to LLM
        logging.debug(f"Input messages to LLM:")
        for i, msg in enumerate(messages):
            logging.debug(
                f"  Message {i}: {type(msg).__name__} - {repr(msg.content[:200])}"
            )

        # Track if we've already truncated once
        attempt_count = 0
        current_messages = messages

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

                    # Invalid message role sequence - strict API rejection
                    # (e.g., Snowflake Cortex). Try reducing messages or raise immediately.
                    if "invalid message role sequence" in error_msg:
                        msg_roles = [_get_message_role(m) for m in current_messages]
                        logging.warning(
                            f"API rejected message role sequence (attempt {other_error_attempt}). "
                            f"Roles: {msg_roles}, Count: {len(current_messages)}"
                        )
                        # Try reducing to system + last user message only
                        if other_error_attempt >= 3:
                            logging.error(
                                f"Persistent 'invalid message role sequence' after "
                                f"{other_error_attempt} attempts. Raising."
                            )
                            raise
                        # Attempt truncation to reduce message count
                        if len(current_messages) > 4:
                            current_messages = reduce_messages_for_payload(
                                current_messages, max_chars=16000
                            )
                            logging.warning(
                                f"Reduced messages to {len(current_messages)} for retry."
                            )
                            continue
                        raise

                    # Payload too large / max tokens errors - try truncating
                    if any(
                        phrase in error_msg
                        for phrase in [
                            "payload too large",
                            "request entity too large",
                            "413",
                            "context length exceeded",
                            "request too large",
                            "max tokens",
                            "maximum context length",
                        ]
                    ):
                        truncation_steps = [50000, 25000, 16666, 10000, 8000, 6000]
                        if other_error_attempt <= len(truncation_steps):
                            max_chars = truncation_steps[other_error_attempt - 1]
                            logging.warning(
                                f"Payload too large/max tokens error detected (attempt {other_error_attempt}). "
                                f"Truncating messages to {max_chars} chars..."
                            )
                            current_messages = truncate_messages(
                                current_messages, max_chars=max_chars
                            )
                            if other_error_attempt >= 3:
                                current_messages = reduce_messages_for_payload(
                                    current_messages, max_chars=max_chars
                                )
                            continue  # Retry with truncated messages
                        else:
                            logging.error(
                                f"Payload still too large/max tokens after {other_error_attempt} truncation attempts"
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
        custom_parser_format_instructions: Optional[str] = None,
        custom_format_template: Optional[str] = None,
        custom_retry_error_template: Optional[str] = None,
        custom_retry_hint_template: Optional[str] = None,
        max_parsing_retries: int = 3,
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

            custom_parser_format_instructions: Optional custom format instructions from parser.
                This replaces the output of `output_parser.get_format_instructions()`.
                Use this when the default parser format instructions are not suitable.
                If None, uses `output_parser.get_format_instructions()`.

            custom_format_template: Optional custom template for initial format instructions.
                Should contain placeholder: {format_instructions}
                If None, uses default template from structured_output_format.txt

            custom_retry_error_template: Optional custom template for retry error messages.
                Should contain placeholders: {attempt}, {max_attempts}, {error_type}, {error_message}
                If None, uses default template from retry_error_format.txt

            custom_retry_hint_template: Optional custom template for retry hint messages.
                Should contain placeholder: {format_instructions}
                If None, uses default template from retry_hint_format.txt

            max_parsing_retries: Maximum number of parsing retry attempts.

            **kwargs: Additional arguments passed to the model.

        Returns:
            A Runnable that takes same inputs as BaseChatModel and outputs a dict with keys:
            - 'raw': BaseMessage (the raw LLM response)
            - 'parsed': Pydantic instance or dict (or None if parsing failed)
            - 'parsing_error': Exception or None

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

        # Wrap with lenient parser to allow headers/explanations around JSON
        output_parser = LenientOutputParser(output_parser)

        # Get format instructions from the parser (or use custom if provided)
        parser_format_instructions = (
            custom_parser_format_instructions
            if custom_parser_format_instructions is not None
            else output_parser.get_format_instructions()
        )

        # Use custom format template if provided, otherwise use default
        format_template = (
            custom_format_template or DEFAULT_TEMPLATES["structured_output_format"]
        )
        format_instructions = format_template.format(
            format_instructions=parser_format_instructions
        )

        parsing_retries = max(1, max_parsing_retries)

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

        # Create a wrapper with parsing retry logic
        def invoke_with_parsing_retry(
            messages: Union[List[BaseMessage], LanguageModelInput],
            max_parsing_retries: int = parsing_retries,
        ):
            """Invoke LLM with automatic retry on parsing failures.

            Uses a REPLACE-not-accumulate strategy for retry messages:
            Each retry replaces the previous retry's [AI, Human] pair rather than
            appending additional pairs. This prevents message count growth that
            can cause "invalid message role sequence" errors on strict APIs
            (e.g., Snowflake Cortex).
            """
            from langchain_core.messages import AIMessage, HumanMessage

            base_messages = add_format_instructions_to_messages(messages)
            current_messages = base_messages

            for attempt in range(1, max_parsing_retries + 1):
                # Invoke LLM
                try:
                    response = self.invoke(current_messages)
                except Exception as api_error:
                    # If the API call itself fails (e.g., "invalid message role sequence"
                    # from strict APIs like Snowflake Cortex when retry messages were added),
                    # fall back to base messages without retry context.
                    error_msg_lower = str(api_error).lower()
                    if "invalid message role sequence" in error_msg_lower and len(
                        current_messages
                    ) > len(base_messages):
                        logging.warning(
                            f"Parsing retry attempt {attempt}: API rejected expanded messages "
                            f"({len(current_messages)} msgs). Falling back to base messages "
                            f"({len(base_messages)} msgs)."
                        )
                        current_messages = base_messages
                        try:
                            response = self.invoke(current_messages)
                        except Exception:
                            # If base messages also fail, re-raise original error
                            raise api_error
                    else:
                        raise

                # Try to parse the response
                try:
                    response_text = (
                        str(response.content)
                        if hasattr(response, "content")
                        else str(response)
                    )
                    parsed = output_parser.parse(response_text)

                    # Success - return dict with raw, parsed, and no error
                    return {"raw": response, "parsed": parsed, "parsing_error": None}

                except Exception as e:
                    # Parsing failed
                    if attempt >= max_parsing_retries:
                        # Max retries reached - return dict with error
                        logging.error(
                            f"Parsing failed after {max_parsing_retries} attempts. "
                            f"Last error: {e}"
                        )
                        return {"raw": response, "parsed": None, "parsing_error": e}

                    # Build retry message with error feedback
                    error_type = type(e).__name__
                    error_message = str(e)
                    # Use string.Template to safely substitute error_message
                    # This avoids KeyError from .format() when error_message contains {key} patterns

                    # Use custom templates if provided, otherwise use defaults
                    retry_error_template_str = (
                        custom_retry_error_template
                        or DEFAULT_TEMPLATES["retry_error_format"]
                    )
                    retry_hint_template_str = (
                        custom_retry_hint_template
                        or DEFAULT_TEMPLATES["retry_hint_format"]
                    )

                    def safe_format(template_str: str, **kwargs) -> str:
                        """Safely format template with Jinja2 + brace placeholders."""
                        # First, try Jinja2 rendering (supports {{ var }} syntax)
                        try:
                            from jinja2 import Template

                            template_str = Template(template_str).render(**kwargs)
                        except Exception:
                            # Fall back to raw string if Jinja2 rendering fails
                            pass

                        # Then replace {key} placeholders safely
                        import uuid

                        placeholders = {}
                        result = template_str
                        for key, value in kwargs.items():
                            marker = (
                                f"__PLACEHOLDER_{key.upper()}_{uuid.uuid4().hex[:8]}__"
                            )
                            placeholders[marker] = str(value)
                            result = result.replace(f"{{{key}}}", marker)

                        # Now replace markers with actual values
                        for marker, value in placeholders.items():
                            result = result.replace(marker, value)

                        return result

                    retry_error_msg = safe_format(
                        retry_error_template_str,
                        attempt=attempt,
                        max_attempts=max_parsing_retries,
                        error_type=error_type,
                        error_message=error_message,
                    )

                    retry_hint_msg = safe_format(
                        retry_hint_template_str,
                        format_instructions=parser_format_instructions,
                    )

                    retry_feedback = f"{retry_error_msg}\n\n{retry_hint_msg}"

                    logging.warning(
                        f"Parsing attempt {attempt}/{max_parsing_retries} failed: {error_type}. "
                        f"Retrying with format reminder..."
                    )

                    # Build retry messages using REPLACE strategy:
                    # Always start from base_messages and add exactly ONE [AI, Human] pair.
                    # This prevents message count growth that breaks strict APIs
                    # (e.g., Snowflake Cortex rejects growing message chains).
                    response_content = (
                        str(response.content)
                        if hasattr(response, "content")
                        else str(response)
                    )
                    current_messages = list(base_messages) + [
                        AIMessage(content=response_content),
                        HumanMessage(content=retry_feedback),
                    ]

        # Create runnable from the retry wrapper
        return RunnableLambda(lambda x: invoke_with_parsing_retry(x))  # type: ignore
