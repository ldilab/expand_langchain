import logging
import os
from typing import Any, List, Optional

from expand_langchain.model.custom_api.snowflake import ChatSnowflakeCortex
from expand_langchain.utils.registry import model_registry
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import BaseMessage, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


@model_registry(name="embedding")
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
        if self.platform == "azure":
            raise ValueError(
                "Azure platform is not supported for embedding models. "
                "Please use a different platform."
            )

        elif self.platform == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                model=self.model,
                max_retries=self.max_retries,
            )

        elif self.platform == "open_webui":
            from langchain_ollama.embeddings import OllamaEmbeddings

            return OllamaEmbeddings(
                model=self.model,
                base_url=os.environ["OPEN_WEBUI_BASE_URL"],
                headers={
                    "Authorization": f"Bearer {os.environ['OPEN_WEBUI_API_KEY']}",
                },
            )

        elif self.platform == "ollama":
            from langchain_ollama.embeddings import OllamaEmbeddings

            return OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url or os.environ["OLLAMA_BASE_URL"],
            )

        elif self.platform == "vllm":
            raise ValueError(
                "VLLM platform is not supported for embedding models. "
                "Please use a different platform."
            )

        elif self.platform == "snowflake":
            raise ValueError(
                "Snowflake platform is not supported for embedding models. "
                "Please use a different platform."
            )

        else:
            raise ValueError(f"platform {self.platform} not supported")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        embedded_docs = self.llm.embed_documents(
            texts
        )
        return embedded_docs

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.llm.embed_query(text)
