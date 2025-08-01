from typing import Dict, List, Tuple

from expand_langchain.utils.registry import prompt_registry
from langchain.prompts import AIMessagePromptTemplate as AIMPT
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate as HMPT
from langchain.prompts import SystemMessagePromptTemplate as SMPT
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder


@prompt_registry(name="chat")
def chat_prompt(
    examples: list,
    body_template_paths: List[str],
    system_template_paths: List[str] = [],
    chat_history_len: int = 0,
    chat_history_key: str = "chat_history",
    **kwargs,
):
    system = system_prompt(system_template_paths)
    body = body_prompt(body_template_paths)
    example = example_prompt(examples, body) if examples else None
    target = target_prompt(body)

    result = system
    if example:
        result = result + example
    if chat_history_len > 0:
        result = result + ChatPromptTemplate.from_messages(
            [MessagesPlaceholder(variable_name=chat_history_key)]
        )

    result = result + target

    return result


def system_prompt(paths: List[str]):
    messages = []
    for key in paths:
        with open(f"{key}/system.txt", "r") as f:
            template_str = f.read()
        messages.append(SMPT.from_template(template_str, template_format="jinja2"))

    return ChatPromptTemplate.from_messages(messages)


def body_prompt(paths: List[str]):
    messages = []
    for path in paths:
        with open(f"{path}/human.txt", "r") as f:
            human_str = f.read()
        human_message = HMPT.from_template(human_str, template_format="jinja2")
        messages.append(human_message)

        with open(f"{path}/ai.txt", "r") as f:
            ai_str = f.read()
        ai_message = AIMPT.from_template(ai_str, template_format="jinja2")
        messages.append(ai_message)

    return ChatPromptTemplate.from_messages(messages)


def example_prompt(
    examples: List[Dict[str, str]],
    body: ChatPromptTemplate,
):
    return FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=body,
    )


def target_prompt(
    body: ChatPromptTemplate,
):
    return ChatPromptTemplate.from_messages(body.messages[:-1])
