"""
Custom exceptions for expand_langchain package.
Provides a clear hierarchy of exceptions for better error handling.
"""


class ExpandLangChainError(Exception):
    """Base exception for all expand_langchain related errors."""

    pass


class ConfigurationError(ExpandLangChainError):
    """Raised when there are configuration-related issues."""

    pass


class DataLoadError(ExpandLangChainError):
    """Raised when dataset loading fails."""

    pass


class ModelError(ExpandLangChainError):
    """Raised when LLM model operations fail."""

    pass


class RegistryError(ExpandLangChainError):
    """Raised when registry operations fail."""

    pass


class CacheError(ExpandLangChainError):
    """Raised when cache operations fail."""

    pass
