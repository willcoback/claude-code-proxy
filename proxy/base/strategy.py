"""Base strategy pattern for model converters."""

import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Type


@dataclass
class TokenUsage:
    """Token usage information."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class ProxyResponse:
    """Standard response from proxy."""
    data: Dict[str, Any]
    usage: TokenUsage
    is_stream: bool = False


class BaseModelStrategy(ABC):
    """
    Abstract base class for model conversion strategies.

    Each model provider (Gemini, Grok, etc.) should implement this interface
    to handle request/response conversions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with provider-specific configuration.

        Args:
            config: Provider configuration dictionary
        """
        self.config = config
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', '')
        self.base_url = config.get('base_url', '')
        self.timeout = config.get('timeout', 300)
        self.provider_proxy = config.get('proxy', '')

    def _get_http_client(self) -> httpx.AsyncClient:
        """
        Returns an httpx.AsyncClient instance, configured with a proxy if specified.
        """
        if self.provider_proxy:
            return httpx.AsyncClient(proxy=self.provider_proxy, timeout=self.timeout)
        else:
            return httpx.AsyncClient(timeout=self.timeout)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'gemini', 'grok')."""
        pass

    @abstractmethod
    def convert_request(self, claude_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Claude Code request format to target model request format.

        Args:
            claude_request: Request in Claude format

        Returns:
            Request in target model format
        """
        pass

    @abstractmethod
    def convert_response(self, model_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert target model response format to Claude Code response format.

        Args:
            model_response: Response from target model

        Returns:
            Response in Claude format
        """
        pass

    @abstractmethod
    async def send_request(self, request: Dict[str, Any]) -> ProxyResponse:
        """
        Send request to target model API.

        Args:
            request: Request in target model format

        Returns:
            ProxyResponse with data and token usage
        """
        pass

    @abstractmethod
    async def stream_request(
            self,
            request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send streaming request to target model API.

        Args:
            request: Request in target model format

        Yields:
            Response chunks in Claude format
        """
        pass

    def get_token_usage(self, response: Dict[str, Any]) -> TokenUsage:
        """
        Extract token usage from response.

        Args:
            response: Model response

        Returns:
            TokenUsage object
        """
        return TokenUsage()

    async def proxy(
            self,
            claude_request: Dict[str, Any],
            stream: bool = False
    ):
        """
        Main proxy method that handles the full conversion cycle.

        Args:
            claude_request: Request in Claude format
            stream: Whether to use streaming

        Returns:
            Response in Claude format or async generator for streaming
        """
        # Convert request
        target_request = self.convert_request(claude_request)

        if stream:
            return self.stream_request(target_request)
        else:
            # Send request
            response = await self.send_request(target_request)
            return response


class StrategyFactory:
    """Factory for creating model strategy instances."""

    _strategies: Dict[str, Type[BaseModelStrategy]] = {}

    @classmethod
    def register(cls, provider_name: str, strategy_class: Type[BaseModelStrategy]):
        """
        Register a strategy class for a provider.

        Args:
            provider_name: Name of the provider (e.g., 'gemini')
            strategy_class: Strategy class to register
        """
        cls._strategies[provider_name.lower()] = strategy_class

    @classmethod
    def get_strategy(cls, provider_name: str, config: Dict[str, Any]) -> BaseModelStrategy:
        """
        Get a strategy instance for the specified provider.

        Args:
            provider_name: Name of the provider
            config: Provider configuration

        Returns:
            Strategy instance

        Raises:
            ValueError: If provider is not registered
        """
        provider_key = provider_name.lower()
        if provider_key not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. Available: {available}"
            )

        strategy_class = cls._strategies[provider_key]
        return strategy_class(config)

    @classmethod
    def list_providers(cls) -> list:
        """List all registered providers."""
        return list(cls._strategies.keys())
