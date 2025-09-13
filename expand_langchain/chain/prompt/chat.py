from typing import Dict, List, Optional, Tuple, Union

from langchain.prompts import AIMessagePromptTemplate as AIMPT
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate as HMPT
from langchain.prompts import SystemMessagePromptTemplate as SMPT
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
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

        if chat_history_key and chat_history_len > 0:
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
            chat_history_key=chat_history_key,
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
        if self.chat_history_len > 0:
            input[self.chat_history_key] = input.get(self.chat_history_key, [])[
                -self.chat_history_len :
            ]

        return super().invoke(input, config, **kwargs)
