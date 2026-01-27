"""Configuration management module."""

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for the proxy server."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    _config_path: Optional[Path] = None
    _config_mtime: float = 0.0
    _lock: threading.RLock
    _last_reload_time: float = 0.0
    _reload_cooldown: float = 1.0  # Minimum seconds between reloads
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration instance (only once)."""
        if not self._initialized:
            self._config_path = None
            self._config_mtime = 0.0
            self._lock = threading.RLock()
            self._last_reload_time = 0.0
            self._reload_cooldown = 1.0
            self._initialized = True

    def load(self, config_path: str = None):
        """Load configuration from YAML file."""
        with self._lock:
            if config_path is None:
                # Default config path
                base_dir = Path(__file__).parent.parent.parent
                config_path = base_dir / "config" / "config.yaml"

            config_path_obj = Path(config_path)

            try:
                # Load configuration
                with open(config_path_obj, 'r', encoding='utf-8') as f:
                    new_config = yaml.safe_load(f)

                # Resolve environment variables
                self._resolve_env_vars(new_config)

                # Update configuration
                self._config = new_config

                # Store metadata for hot reload
                self._config_path = config_path_obj
                self._config_mtime = config_path_obj.stat().st_mtime
                self._last_reload_time = time.time()

                return self
            except (OSError, IOError, yaml.YAMLError):
                # If this is the initial load, re-raise the exception
                if not self._config:  # No existing configuration
                    raise
                # For hot reload, log error but keep existing configuration
                # We'll log this error if we have a logger, but config module doesn't have one
                # Silently fail to avoid breaking the service
                return self

    def reload(self) -> None:
        """Reload configuration from the same config file."""
        if self._config_path is None:
            # If config hasn't been loaded yet, load with default path
            self.load()
        else:
            self.load(str(self._config_path))

    def should_reload(self) -> bool:
        """Check if configuration should be reloaded based on file modification time."""
        with self._lock:
            # If config path is not set, no need to reload
            if self._config_path is None:
                return False

            # Check cooldown period to avoid excessive reloads
            current_time = time.time()
            if current_time - self._last_reload_time < self._reload_cooldown:
                return False

            # Check if file modification time has changed
            try:
                current_mtime = self._config_path.stat().st_mtime
                return current_mtime > self._config_mtime
            except (OSError, IOError):
                # File may not exist or be inaccessible
                return False

    def check_and_reload(self) -> bool:
        """Check and reload configuration if needed.

        Returns:
            True if configuration was reloaded, False otherwise.
        """
        with self._lock:
            if self.should_reload():
                self.reload()
                return True
            return False

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
        self.check_and_reload()
        return self.get('provider.name', 'gemini')

    @property
    def server_host(self) -> str:
        """Get server host."""
        return self.get('server.host', '0.0.0.0')

    @property
    def server_port(self) -> int:
        """Get server port."""
        return self.get('server.port', 8080)

    @property
    def chatlog_enabled(self) -> bool:
        """Get chatlog enabled status (hot-reloadable)."""
        self.check_and_reload()
        return self.get('logging.chatlog_enabled', False)

    def get_provider_config(self, provider: str = None) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        self.check_and_reload()
        if provider is None:
            provider = self.provider_name
        return self.get(provider, {})


# Global config instance
config = Config()
