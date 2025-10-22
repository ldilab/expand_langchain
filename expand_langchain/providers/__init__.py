"""
Factory pattern for LLM providers.
Eliminates duplicate platform-specific code in chat and embedding models.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr


class LLMProviderError(Exception):
    """Raised when LLM provider operations fail."""

    pass


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    def create_chat_model(self, **kwargs) -> BaseChatModel:
        """Create a chat model instance."""
        pass

    @abstractmethod
    def create_embedding_model(self, **kwargs) -> Embeddings:
        """Create an embedding model instance."""
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """Validate that required configuration is available."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""

    def validate_config(self) -> None:
        """Validate OpenAI configuration."""
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMProviderError("OPENAI_API_KEY environment variable is required")

    def create_chat_model(self, **kwargs) -> BaseChatModel:
        """Create OpenAI chat model."""
        from langchain_openai import ChatOpenAI

        self.validate_config()

        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = kwargs.get("base_url") or os.environ.get("OPENAI_API_BASE")

        # Set a reasonable default timeout to prevent hanging on large payloads
        timeout = kwargs.get("timeout", 120)

        return ChatOpenAI(
            api_key=SecretStr(api_key) if api_key else None,
            model=kwargs.get("model", "gpt-3.5-turbo"),
            max_completion_tokens=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 1.0),
            max_retries=kwargs.get("max_retries", 10),
            base_url=base_url,
            extra_body=kwargs.get("extra_body"),
            timeout=timeout,
        )

    def create_embedding_model(self, **kwargs) -> Embeddings:
        """Create OpenAI embedding model."""
        from langchain_openai import OpenAIEmbeddings

        self.validate_config()

        return OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model=kwargs.get("model", "text-embedding-ada-002"),
            max_retries=kwargs.get("max_retries", 10),
        )


class AzureProvider(BaseLLMProvider):
    """Azure OpenAI provider implementation."""

    def validate_config(self) -> None:
        """Validate Azure configuration."""
        required_vars = ["AZURE_ENDPOINT", "AZURE_API_VERSION", "AZURE_API_KEY"]
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise LLMProviderError(f"Missing Azure environment variables: {missing}")

    def create_chat_model(self, **kwargs) -> BaseChatModel:
        """Create Azure OpenAI chat model."""
        from langchain_openai import AzureChatOpenAI

        self.validate_config()

        # Set timeout if provided
        timeout = kwargs.get("timeout", 120)

        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version=os.environ["AZURE_API_VERSION"],
            api_key=SecretStr(os.environ["AZURE_API_KEY"]),
            azure_deployment=kwargs.get("model", "gpt-35-turbo"),
            max_tokens=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 0.7),
            model_kwargs={"top_p": kwargs.get("top_p", 1.0)},
            max_retries=kwargs.get("max_retries", 10),
            timeout=timeout,
        )

    def create_embedding_model(self, **kwargs) -> Embeddings:
        """Azure does not support embedding models in this implementation."""
        raise LLMProviderError("Azure provider does not support embedding models")


class OllamaProvider(BaseLLMProvider):
    """Ollama provider implementation."""

    def validate_config(self) -> None:
        """Validate Ollama configuration."""
        if not os.environ.get("OLLAMA_BASE_URL"):
            raise LLMProviderError("OLLAMA_BASE_URL environment variable is required")

    def create_chat_model(self, **kwargs) -> BaseChatModel:
        """Create Ollama chat model."""
        from langchain_ollama import ChatOllama

        self.validate_config()

        return ChatOllama(
            model=kwargs.get("model", "llama2"),
            num_predict=kwargs.get("max_tokens", 1000),
            num_ctx=kwargs.get("num_ctx"),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 1.0),
            base_url=kwargs.get("base_url") or os.environ["OLLAMA_BASE_URL"],
        )

    def create_embedding_model(self, **kwargs) -> Embeddings:
        """Create Ollama embedding model."""
        from langchain_ollama.embeddings import OllamaEmbeddings

        self.validate_config()

        return OllamaEmbeddings(
            model=kwargs.get("model", "llama2"),
            base_url=kwargs.get("base_url") or os.environ["OLLAMA_BASE_URL"],
        )


class OpenWebUIProvider(BaseLLMProvider):
    """Open WebUI provider implementation."""

    def validate_config(self) -> None:
        """Validate Open WebUI configuration."""
        required_vars = ["OPEN_WEBUI_BASE_URL", "OPEN_WEBUI_API_KEY"]
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise LLMProviderError(
                f"Missing Open WebUI environment variables: {missing}"
            )

    def create_chat_model(self, **kwargs) -> BaseChatModel:
        """Create Open WebUI chat model."""
        from langchain_ollama import ChatOllama

        self.validate_config()

        return ChatOllama(
            model=kwargs.get("model", "llama2"),
            num_predict=kwargs.get("max_tokens", 1000),
            num_ctx=kwargs.get("num_ctx"),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 1.0),
            base_url=os.environ["OPEN_WEBUI_BASE_URL"],
            headers={
                "Authorization": f"Bearer {os.environ['OPEN_WEBUI_API_KEY']}",
                "Content-Type": "application/json",
            },
        )

    def create_embedding_model(self, **kwargs) -> Embeddings:
        """Create Open WebUI embedding model."""
        from langchain_ollama.embeddings import OllamaEmbeddings

        self.validate_config()

        return OllamaEmbeddings(
            model=kwargs.get("model", "llama2"),
            base_url=os.environ["OPEN_WEBUI_BASE_URL"],
            headers={
                "Authorization": f"Bearer {os.environ['OPEN_WEBUI_API_KEY']}",
            },
        )


class VLLMProvider(BaseLLMProvider):
    """VLLM provider implementation."""

    def validate_config(self) -> None:
        """Validate VLLM configuration."""
        if not os.environ.get("VLLM_BASE_URL"):
            raise LLMProviderError("VLLM_BASE_URL environment variable is required")

    def create_chat_model(self, **kwargs) -> BaseChatModel:
        """Create VLLM chat model."""
        from langchain_openai import ChatOpenAI

        self.validate_config()

        # Set timeout if provided
        timeout = kwargs.get("timeout", 120)

        return ChatOpenAI(
            openai_api_key=os.environ.get("VLLM_API_KEY", "dummy-key"),
            openai_api_base=os.environ["VLLM_BASE_URL"],
            model=kwargs.get("model", "llama2"),
            max_tokens=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 1.0),
            max_retries=kwargs.get("max_retries", 10),
            timeout=timeout,
        )

    def create_embedding_model(self, **kwargs) -> Embeddings:
        """VLLM does not support embedding models in this implementation."""
        raise LLMProviderError("VLLM provider does not support embedding models")


class SnowflakeProvider(BaseLLMProvider):
    """Snowflake Cortex provider implementation."""

    def validate_config(self) -> None:
        """Validate Snowflake configuration."""
        required_vars = [
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_USERNAME",
            "SNOWFLAKE_PASSWORD",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_SCHEMA",
            "SNOWFLAKE_ROLE",
            "SNOWFLAKE_WAREHOUSE",
        ]
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise LLMProviderError(
                f"Missing Snowflake environment variables: {missing}"
            )

    def create_chat_model(self, **kwargs) -> BaseChatModel:
        """Create Snowflake Cortex chat model."""
        from expand_langchain.chain.model.custom_api.snowflake import (
            ChatSnowflakeCortex,
        )

        self.validate_config()

        return ChatSnowflakeCortex(
            model=kwargs.get("model", "snowflake-arctic"),
            cortex_function="complete",
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
            top_p=kwargs.get("top_p", 1.0),
            account=os.environ.get("SNOWFLAKE_ACCOUNT"),
            username=os.environ.get("SNOWFLAKE_USERNAME"),
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
            database=os.environ.get("SNOWFLAKE_DATABASE"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA"),
            role=os.environ.get("SNOWFLAKE_ROLE"),
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        )

    def create_embedding_model(self, **kwargs) -> Embeddings:
        """Snowflake does not support embedding models in this implementation."""
        raise LLMProviderError("Snowflake provider does not support embedding models")


class LLMProviderFactory:
    """Factory class for creating LLM providers."""

    _providers = {
        "openai": OpenAIProvider,
        "azure": AzureProvider,
        "ollama": OllamaProvider,
        "open_webui": OpenWebUIProvider,
        "vllm": VLLMProvider,
        "snowflake": SnowflakeProvider,
    }

    @classmethod
    def create_chat_model(cls, platform: str, **kwargs) -> BaseChatModel:
        """
        Create a chat model for the specified platform.

        Args:
            platform: Platform name (openai, azure, ollama, etc.)
            **kwargs: Model configuration parameters (including timeout)

        Returns:
            Configured chat model instance

        Raises:
            LLMProviderError: If platform is not supported or configuration is invalid
        """
        if platform not in cls._providers:
            available = list(cls._providers.keys())
            raise LLMProviderError(
                f"Unsupported platform: {platform}. Available: {available}"
            )

        # Handle timeout parameter with platform-specific warnings
        timeout = kwargs.get("timeout")
        if timeout is not None:
            import logging

            timeout_supported_platforms = ["openai", "azure", "vllm"]

            if platform in timeout_supported_platforms:
                logging.info(f"Timeout set to {timeout}s for {platform} platform")
            else:
                logging.warning(
                    f"Timeout parameter is not directly supported for {platform} platform. "
                    f"Timeout setting will be ignored. "
                    f"Supported platforms: {', '.join(timeout_supported_platforms)}"
                )
                # Remove timeout from kwargs for unsupported platforms
                kwargs = {k: v for k, v in kwargs.items() if k != "timeout"}

        provider = cls._providers[platform]()
        return provider.create_chat_model(**kwargs)

    @classmethod
    def create_embedding_model(cls, platform: str, **kwargs) -> Embeddings:
        """
        Create an embedding model for the specified platform.

        Args:
            platform: Platform name (openai, azure, ollama, etc.)
            **kwargs: Model configuration parameters

        Returns:
            Configured embedding model instance

        Raises:
            LLMProviderError: If platform is not supported or configuration is invalid
        """
        if platform not in cls._providers:
            available = list(cls._providers.keys())
            raise LLMProviderError(
                f"Unsupported platform: {platform}. Available: {available}"
            )

        provider = cls._providers[platform]()
        return provider.create_embedding_model(**kwargs)

    @classmethod
    def list_supported_platforms(cls) -> list:
        """List all supported platforms."""
        return list(cls._providers.keys())

    @classmethod
    def validate_platform_config(cls, platform: str) -> None:
        """
        Validate configuration for a specific platform.

        Args:
            platform: Platform name to validate

        Raises:
            LLMProviderError: If platform is not supported or configuration is invalid
        """
        if platform not in cls._providers:
            available = list(cls._providers.keys())
            raise LLMProviderError(
                f"Unsupported platform: {platform}. Available: {available}"
            )

        provider = cls._providers[platform]()
        provider.validate_config()
