import re

from langchain_core.output_parsers import BaseOutputParser


class RemoveThinkTagsParser(BaseOutputParser[str]):
    """
    Parser that removes reasoning model's <think> section from text.
    
    This parser removes everything before and including the </think> tag,
    which handles cases where reasoning models include their thinking process
    in <think>...</think> tags, including when code blocks appear inside
    the think tags.
    
    This parser is designed to be used in a chain with other parsers:
    Example: RemoveThinkTagsParser() | PydanticOutputParser(...)
    """

    def parse(self, text: str) -> str:
        """
        Remove everything before and including </think> tag.
        
        Args:
            text: Input text that may contain <think>...</think> section
            
        Returns:
            str: Cleaned text with <think> section removed
        """
        # Remove everything before and including </think> tag
        # This handles cases where code blocks might be inside <think> tags
        think_end_pattern = r"^.*?</think>\s*"
        cleaned_text = re.sub(think_end_pattern, "", text, flags=re.DOTALL)
        
        return cleaned_text

    @property
    def _type(self) -> str:
        return "remove_think_tags"
