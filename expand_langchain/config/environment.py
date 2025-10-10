"""
Centralized environment configuration management.
Handles all environment variable access and validation.
"""

import logging
import os
from typing import Optional

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class EnvironmentConfig:
    """
    Centralized environment variable management with validation.

    This class provides a single point of access for all environment variables
    used throughout the application, with proper validation and error handling.
    """

    # Required environment variables
    REQUIRED_VARS = []

    # Optional environment variables with defaults
    OPTIONAL_VARS = {
        "OPENAI_API_BASE": None,
        "VLLM_API_KEY": "dummy-key",
        "LARK_WEBHOOK": None,
    }

    def __init__(self, validate_on_init: bool = True):
        """
        Initialize environment configuration.

        Args:
            validate_on_init: Whether to validate required variables on initialization
        """
        if validate_on_init:
            self._validate_required_vars()

    # OpenAI Configuration
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key."""
        return self._get_required("OPENAI_API_KEY")

    @property
    def openai_api_base(self) -> Optional[str]:
        """Get OpenAI API base URL."""
        return os.environ.get("OPENAI_API_BASE")

    # Azure Configuration
    @property
    def azure_endpoint(self) -> Optional[str]:
        """Get Azure endpoint URL."""
        return os.environ.get("AZURE_ENDPOINT")

    @property
    def azure_api_version(self) -> Optional[str]:
        """Get Azure API version."""
        return os.environ.get("AZURE_API_VERSION")

    @property
    def azure_api_key(self) -> Optional[str]:
        """Get Azure API key."""
        return os.environ.get("AZURE_API_KEY")

    # Ollama Configuration
    @property
    def ollama_base_url(self) -> Optional[str]:
        """Get Ollama base URL."""
        return os.environ.get("OLLAMA_BASE_URL")

    # Open WebUI Configuration
    @property
    def open_webui_base_url(self) -> Optional[str]:
        """Get Open WebUI base URL."""
        return os.environ.get("OPEN_WEBUI_BASE_URL")

    @property
    def open_webui_api_key(self) -> Optional[str]:
        """Get Open WebUI API key."""
        return os.environ.get("OPEN_WEBUI_API_KEY")

    # VLLM Configuration
    @property
    def vllm_base_url(self) -> Optional[str]:
        """Get VLLM base URL."""
        return os.environ.get("VLLM_BASE_URL")

    @property
    def vllm_api_key(self) -> str:
        """Get VLLM API key."""
        return os.environ.get("VLLM_API_KEY", "dummy-key")

    # Snowflake Configuration
    @property
    def snowflake_account(self) -> Optional[str]:
        """Get Snowflake account."""
        return os.environ.get("SNOWFLAKE_ACCOUNT")

    @property
    def snowflake_username(self) -> Optional[str]:
        """Get Snowflake username."""
        return os.environ.get("SNOWFLAKE_USERNAME")

    @property
    def snowflake_password(self) -> Optional[str]:
        """Get Snowflake password."""
        return os.environ.get("SNOWFLAKE_PASSWORD")

    @property
    def snowflake_database(self) -> Optional[str]:
        """Get Snowflake database."""
        return os.environ.get("SNOWFLAKE_DATABASE")

    @property
    def snowflake_schema(self) -> Optional[str]:
        """Get Snowflake schema."""
        return os.environ.get("SNOWFLAKE_SCHEMA")

    @property
    def snowflake_warehouse(self) -> Optional[str]:
        """Get Snowflake warehouse."""
        return os.environ.get("SNOWFLAKE_WAREHOUSE")

    @property
    def snowflake_role(self) -> Optional[str]:
        """Get Snowflake role."""
        return os.environ.get("SNOWFLAKE_ROLE")

    # Notification Configuration
    @property
    def lark_webhook(self) -> Optional[str]:
        """Get Lark webhook URL for notifications."""
        return os.environ.get("LARK_WEBHOOK")

    # Helper methods
    def _get_required(self, key: str) -> str:
        """
        Get a required environment variable.

        Args:
            key: Environment variable name

        Returns:
            Environment variable value

        Raises:
            ConfigurationError: If the environment variable is not set
        """
        value = os.environ.get(key)
        if not value:
            raise ConfigurationError(f"Required environment variable {key} not set")
        return value

    def _get_optional(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an optional environment variable.

        Args:
            key: Environment variable name
            default: Default value if not set

        Returns:
            Environment variable value or default
        """
        return os.environ.get(key, default)

    def _validate_required_vars(self) -> None:
        """
        Validate that all required environment variables are set.

        Raises:
            ConfigurationError: If any required variables are missing
        """
        missing_vars = []

        for var in self.REQUIRED_VARS:
            if not os.environ.get(var):
                missing_vars.append(var)

        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    def set_from_api_keys_file(self, api_keys_path: str) -> None:
        """
        Load API keys from a JSON file and set as environment variables.

        Args:
            api_keys_path: Path to the API keys JSON file
        """
        import json
        from pathlib import Path

        api_keys_file = Path(api_keys_path)

        if not api_keys_file.exists():
            logger.warning(f"API keys file not found: {api_keys_path}")
            return

        try:
            with open(api_keys_file, "r") as f:
                api_keys = json.load(f)

            for key, value in api_keys.items():
                if not os.environ.get(key):
                    logger.info(f"Setting {key} from API keys file")
                    os.environ[key] = value

        except (json.JSONDecodeError, OSError) as e:
            raise ConfigurationError(
                f"Failed to load API keys from {api_keys_path}: {e}"
            )

    def validate_platform_config(self, platform: str) -> None:
        """
        Validate that required environment variables for a platform are set.

        Args:
            platform: Platform name (azure, openai, ollama, etc.)

        Raises:
            ConfigurationError: If required variables for the platform are missing
        """
        platform_requirements = {
            "azure": ["AZURE_ENDPOINT", "AZURE_API_VERSION", "AZURE_API_KEY"],
            "openai": ["OPENAI_API_KEY"],
            "ollama": ["OLLAMA_BASE_URL"],
            "open_webui": ["OPEN_WEBUI_BASE_URL", "OPEN_WEBUI_API_KEY"],
            "vllm": ["VLLM_BASE_URL"],
            "snowflake": [
                "SNOWFLAKE_ACCOUNT",
                "SNOWFLAKE_USERNAME",
                "SNOWFLAKE_PASSWORD",
                "SNOWFLAKE_DATABASE",
                "SNOWFLAKE_SCHEMA",
                "SNOWFLAKE_WAREHOUSE",
                "SNOWFLAKE_ROLE",
            ],
        }

        required_vars = platform_requirements.get(platform, [])
        missing_vars = [var for var in required_vars if not os.environ.get(var)]

        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables for platform '{platform}': {', '.join(missing_vars)}"
            )

    def get_all_set_vars(self) -> dict:
        """
        Get all environment variables that are currently set.

        Returns:
            Dictionary of set environment variables (values masked for security)
        """
        sensitive_keys = ["API_KEY", "PASSWORD", "SECRET", "TOKEN", "WEBHOOK"]

        result = {}
        for key, value in os.environ.items():
            if any(sensitive in key.upper() for sensitive in sensitive_keys):
                result[key] = "***MASKED***"
            else:
                result[key] = value

        return result


# Global instance for easy access
env_config = EnvironmentConfig(validate_on_init=False)
