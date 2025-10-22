from typing import List, Optional, Union

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate as HMPT
from langchain.prompts import SystemMessagePromptTemplate as SMPT
from langchain_core.messages import SystemMessage
from langchain_core.prompts import MessagesPlaceholder


class CustomChatPromptTemplate(ChatPromptTemplate):
    system_template_paths: Union[List[str], str] = []
    body_template_paths: Union[List[str], str] = []
    chat_history_len: int = 0
    chat_history_key: str = "chat_history"

    template: Optional[ChatPromptTemplate] = None

    @classmethod
    def from_custom_config(
        cls,
        system_template_paths: Union[List[str], str] = [],
        body_template_paths: Union[List[str], str] = [],
        chat_history_len: int = 0,
        chat_history_key: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(system_template_paths, str):
            system_template_paths = [system_template_paths]
        if isinstance(body_template_paths, str):
            body_template_paths = [body_template_paths]

        messages = []
        for key in system_template_paths:
            with open(f"{key}/system.txt", "r") as f:
                template_str = f.read()
            messages.append(SMPT.from_template(template_str, template_format="jinja2"))

        if not messages:
            messages.append(SMPT.from_template("", template_format="jinja2"))

        if chat_history_key and chat_history_len != 0:
            messages.append(
                MessagesPlaceholder(variable_name=chat_history_key),
            )

        for path in body_template_paths:
            with open(f"{path}/human.txt", "r") as f:
                human_str = f.read()
            human_message = HMPT.from_template(human_str, template_format="jinja2")
            messages.append(human_message)

        # 커스텀 생성자를 사용하여 속성과 함께 객체 생성
        result = cls(
            messages=messages,
            chat_history_key=chat_history_key or "chat_history",
            chat_history_len=chat_history_len,
            **kwargs,
        )
        return result

    def __init__(
        self, chat_history_key: str = "chat_history", chat_history_len: int = 0, **data
    ):
        super().__init__(**data)
        self.chat_history_key = chat_history_key
        self.chat_history_len = chat_history_len

    def invoke(self, input, config=None, **kwargs):
        """
        Invoke the prompt template with automatic chat history truncation.

        Truncates chat history to the most recent N messages based on chat_history_len.
        System messages are excluded from truncation and always preserved.
        This prevents the conversation context from growing unbounded.

        Chat history truncation behavior:
        - chat_history_len > 0: Keep only the most recent N non-system messages
        - chat_history_len == 0: Chat history disabled (no MessagesPlaceholder added)
        - chat_history_len < 0: No truncation, keep all messages

        Args:
            input: Input dictionary containing prompt variables
            config: Optional runtime configuration
            **kwargs: Additional keyword arguments

        Returns:
            Formatted prompt with truncated chat history
        """
        if self.chat_history_key in input:
            if self.chat_history_len > 0:
                # Truncate to most recent N messages
                chat_history = input.get(self.chat_history_key, [])

                # Separate system messages from other messages
                system_messages = []
                non_system_messages = []

                for msg in chat_history:
                    if isinstance(msg, SystemMessage):
                        system_messages.append(msg)
                    else:
                        non_system_messages.append(msg)

                # Keep only the most recent non-system messages
                truncated_non_system = non_system_messages[-self.chat_history_len :]

                # Combine: system messages first, then truncated chat history
                input[self.chat_history_key] = system_messages + truncated_non_system
            # else: chat_history_len <= 0 means keep all messages, no truncation needed

        return super().invoke(input, config, **kwargs)
