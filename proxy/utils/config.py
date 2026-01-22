"""Configuration management module."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for the proxy server."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str = None):
        """Load configuration from YAML file."""
        if config_path is None:
            # Default config path
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        # Resolve environment variables
        self._resolve_env_vars(self._config)

        return self

    def _resolve_env_vars(self, obj: Any) -> Any:
        """Recursively resolve environment variables in config values."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._resolve_env_vars(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._resolve_env_vars(item)
        elif isinstance(obj, str):
            # Match ${VAR_NAME} pattern
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, obj)
            for match in matches:
                env_value = os.environ.get(match, '')
                obj = obj.replace(f'${{{match}}}', env_value)
        return obj

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def provider_name(self) -> str:
        """Get current provider name."""
        return self.get('provider.name', 'gemini')

    @property
    def server_host(self) -> str:
        """Get server host."""
        return self.get('server.host', '0.0.0.0')

    @property
    def server_port(self) -> int:
        """Get server port."""
        return self.get('server.port', 8080)

    def get_provider_config(self, provider: str = None) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        if provider is None:
            provider = self.provider_name
        return self.get(provider, {})


# Global config instance
config = Config()
