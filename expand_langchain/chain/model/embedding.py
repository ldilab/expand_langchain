import logging
from typing import Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from ...providers import LLMProviderError, LLMProviderFactory


class GeneralEmbeddingModel(BaseModel, Embeddings):
    model: Optional[str] = None
    max_retries: int = 10
    platform: str
    base_url: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return self.llm._llm_type

    @property
    def llm(self):
        """Get the embedding model instance using the factory pattern."""
        try:
            return LLMProviderFactory.create_embedding_model(
                platform=self.platform,
                model=self.model,
                max_retries=self.max_retries,
                base_url=self.base_url,
            )
        except LLMProviderError as e:
            raise ValueError(
                f"Failed to create {self.platform} embedding model: {e}"
            ) from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        embedded_docs = self.llm.embed_documents(texts)
        return embedded_docs

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.llm.embed_query(text)
