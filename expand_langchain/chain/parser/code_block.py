import re

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser


class CodeBlockOutputParser(BaseOutputParser[str]):
    """Custom parser for code blocks."""

    def get_format_instructions(self) -> str:
        return (
            "The output should be a code block enclosed in triple backticks. "
            "For example:\n```plaintext\n# Your code here\n```"
        )

    def parse(self, text: str) -> str:
        """
        Extract code block content from text enclosed with triple backticks.

        Args:
            text: Input text that may contain code blocks

        Returns:
            str: The content inside the first code block found

        Raises:
            OutputParserException: If no code block is found
        """
        # Pattern to match code blocks with optional language specifier
        pattern = r"```(?:\w+)?\s*\n?(.*?)```"

        match = re.search(pattern, text, re.DOTALL)

        if match:
            matches = re.findall(pattern, text, re.DOTALL)
            code_content = matches[-1].strip()
            return code_content
        else:
            raise OutputParserException(
                f"Could not parse code block from text: {text[:100]}..."
            )

    @property
    def _type(self) -> str:
        return "code_block_output_parser"
