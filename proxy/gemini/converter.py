"""Gemini OpenAI compatible model strategy implementation."""

import json
import ssl
import uuid
from typing import Any, AsyncGenerator, Dict

import aiohttp
import certifi
import httpx
from openai import AsyncOpenAI

from ..base.strategy import BaseModelStrategy, StrategyFactory, TokenUsage, ProxyResponse
from ..utils import get_logger, get_chatlog_logger, config
from ..utils.thought_cache import ThoughtSignatureCache

logger = get_logger()
chatlog = get_chatlog_logger()

def truncate_value(value: Any, max_str_length: int = 500) -> Any:
    """
    Recursively truncate long string values while preserving JSON structure.
    """
    if isinstance(value, str):
        if len(value) > max_str_length:
            return value[:max_str_length] + f"... [truncated, {len(value)} chars total]"
        return value
    elif isinstance(value, dict):
        return {k: truncate_value(v, max_str_length) for k, v in value.items()}
    elif isinstance(value, list):
        if len(value) > 20:
            truncated_list = [truncate_value(item, max_str_length) for item in value[:20]]
            truncated_list.append(f"... [{len(value) - 20} more items]")
            return truncated_list
        return [truncate_value(item, max_str_length) for item in value]
    else:
        return value

def format_json_for_log(data: Any, max_str_length: int = 500) -> str:
    """
    Format JSON data for logging.
    """
    try:
        truncated_data = truncate_value(data, max_str_length)
        return json.dumps(truncated_data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)[:5000]

class GeminiStrategy(BaseModelStrategy):
    """
    Strategy for converting between Claude and Gemini OpenAI compatible API formats.
    Gemini OpenAI API is compatible with OpenAI format.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.http_proxy = config.get('proxy', '')  # HTTP/HTTPS proxy URL
        self.thought_cache = ThoughtSignatureCache()  # Initialize thought signature cache
        # Cleanup old cache entries on startup
        self.thought_cache.cleanup_old_entries(max_age_seconds=3600, max_entries=1000)
        logger.info(f"Thought signature cache initialized: {self.thought_cache.get_stats()}")

    @property
    def provider_name(self) -> str:
        return "gemini"

    def convert_request(self, claude_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Claude API request to Gemini OpenAI (OpenAI-compatible) API request.
        """
        gemini_request = {
            "model": self.model,
            "stream": claude_request.get("stream", False),
        }

        # Convert messages
        messages = []

        # Add system message if present
        system = claude_request.get("system")
        if system:
            if isinstance(system, str):
                messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                system_text = []
                for block in system:
                    if isinstance(block, str):
                        system_text.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        system_text.append(block.get("text", ""))
                messages.append({"role": "system", "content": "\n".join(system_text) + "\n\n请使用简体中文回复。确保所有输出内容均清晰、准确，并完全以中文呈现，除非另有明确指示。"})

        for msg in claude_request.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                text_parts = []
                tool_uses = []
                tool_results = []

                for block in content:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "text":
                            text_parts.append(block.get("text", ""))
                        elif block_type == "tool_use":
                            tool_call_id = block.get("id", "")
                            tool_call = {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {}))
                                }
                            }

                            # Try to get thought_signature from cache
                            thought_signature = self.thought_cache.get_signature(tool_call_id)
                            if thought_signature:
                                logger.info(f"Found cached thought_signature for tool_call {tool_call_id}")
                                tool_call["extra_content"] = {
                                    "google": {
                                        "thought_signature": thought_signature
                                    }
                                }

                            tool_uses.append(tool_call)
                        elif block_type == "tool_result":
                            tool_result_content = block.get("content", "")
                            if isinstance(tool_result_content, list):
                                parts = []
                                for item in tool_result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        parts.append(item.get("text", ""))
                                    elif isinstance(item, str):
                                        parts.append(item)
                                tool_result_content = "\n".join(parts)
                            tool_results.append({
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": tool_result_content
                            })

                # If assistant message with tool_use
                if role == "assistant" and tool_uses:
                    assistant_msg = {"role": "assistant"}
                    if text_parts:
                        assistant_msg["content"] = "\n".join(text_parts)
                    else:
                        assistant_msg["content"] = None
                    assistant_msg["tool_calls"] = tool_uses
                    messages.append(assistant_msg)
                # If user message with tool_result
                elif tool_results:
                    for tr in tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_call_id"],
                            "content": tr["content"]
                        })
                else:
                    messages.append({
                        "role": role,
                        "content": "\n".join(text_parts) if text_parts else ""
                    })
            else:
                messages.append({
                    "role": role,
                    "content": content
                })

        gemini_request["messages"] = messages

        # Add common parameters
        if "max_tokens" in claude_request:
            gemini_request["max_tokens"] = claude_request["max_tokens"]

        if "temperature" in claude_request:
            gemini_request["temperature"] = claude_request["temperature"]

        if "top_p" in claude_request:
            gemini_request["top_p"] = claude_request["top_p"]

        if "stop_sequences" in claude_request:
            gemini_request["stop"] = claude_request["stop_sequences"]

        # Convert tools (Claude format -> OpenAI/Gemini OpenAI format)
        tools = claude_request.get("tools", [])
        if tools:
            gemini_tools = []
            for tool in tools:
                # Clean parameters - remove unsupported JSON Schema fields
                parameters = tool.get("input_schema", {})
                cleaned_params = self._clean_json_schema(parameters)

                gemini_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": cleaned_params
                    }
                }
                gemini_tools.append(gemini_tool)
            gemini_request["tools"] = gemini_tools
            # Enable tool choice - let model decide when to use tools
            gemini_request["tool_choice"] = "auto"

        return gemini_request

    def _clean_json_schema(self, schema: Any) -> Any:
        """Remove unsupported JSON Schema fields for Gemini OpenAI API."""
        if not isinstance(schema, dict):
            return schema

        # Fields not supported by OpenAI/Gemini OpenAI API
        unsupported = {
            "$schema", "additionalProperties", "exclusiveMinimum", "exclusiveMaximum",
            "$id", "$ref", "$defs", "definitions", "if", "then", "else",
            "allOf", "anyOf", "oneOf", "not", "propertyNames", "patternProperties",
            "unevaluatedProperties", "unevaluatedItems", "const", "contentEncoding",
            "contentMediaType", "dependentRequired", "dependentSchemas"
        }

        cleaned = {}
        for key, value in schema.items():
            if key in unsupported:
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {k: self._clean_json_schema(v) for k, v in value.items()}
            elif key == "items":
                cleaned[key] = self._clean_json_schema(value) if isinstance(value, dict) else value
            elif isinstance(value, dict):
                cleaned[key] = self._clean_json_schema(value)
            elif isinstance(value, list):
                cleaned[key] = [self._clean_json_schema(item) if isinstance(item, dict) else item for item in value]
            else:
                cleaned[key] = value

        return cleaned

    def convert_response(self, gemini_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Gemini OpenAI (OpenAI-compatible) response to Claude format.
        """
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        choices = gemini_response.get("choices", [])
        content = []
        stop_reason = "end_turn"

        if choices:
            choice = choices[0]
            message = choice.get("message", {})

            # Handle text content
            content_text = message.get("content", "")
            if content_text:
                content.append({"type": "text", "text": content_text})

            # Handle tool calls
            tool_calls = message.get("tool_calls") or []
            for tc in tool_calls:
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                tool_call_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")

                # Extract and cache thought_signature if present
                extra_content = tc.get("extra_content", {})
                google_data = extra_content.get("google", {})
                thought_signature = google_data.get("thought_signature")

                if thought_signature:
                    logger.info(f"Storing thought_signature for tool_call {tool_call_id}")
                    self.thought_cache.store_signature(tool_call_id, thought_signature, msg_id)

                content.append({
                    "type": "tool_use",
                    "id": tool_call_id,
                    "name": func.get("name", ""),
                    "input": args
                })

            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                stop_reason = "max_tokens"
            elif finish_reason == "stop":
                stop_reason = "end_turn"
            elif finish_reason == "tool_calls":
                stop_reason = "tool_use"

        # Ensure content is not empty
        if not content:
            content = [{"type": "text", "text": ""}]

        usage_data = gemini_response.get("usage", {})
        usage = {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0)
        }

        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": self.model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage
        }

    def get_token_usage(self, response: Dict[str, Any]) -> TokenUsage:
        usage_data = response.get("usage", {})
        return TokenUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )

    async def send_request(self, request: Dict[str, Any]) -> ProxyResponse:
        logger.info(f"Sending request to Gemini OpenAI: {self.base_url}", extra={'provider': f"{self.provider_name}:{self.model}"})

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            http_client=self._get_http_client()
        )

        try:
            response = await client.chat.completions.create(**request)
            gemini_response = response.model_dump()

            if config.chatlog_enabled:
                chatlog.info(
                    f"=== RAW GEMINI OPENAI RESPONSE ===\n{format_json_for_log(gemini_response, max_str_length=1000)}",
                    extra={'provider': f"{self.provider_name}:{self.model}"}
                )

            usage = self.get_token_usage(gemini_response)
            claude_response = self.convert_response(gemini_response)

            if config.chatlog_enabled:
                chatlog.info(
                    f"=== CONVERTED CLAUDE RESPONSE ===\n{format_json_for_log(claude_response, max_str_length=1000)}",
                    extra={'provider': f"{self.provider_name}:{self.model}"}
                )

            return ProxyResponse(
                data=claude_response,
                usage=usage,
                is_stream=False
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini OpenAI API error: {error_msg}")

            # Provide helpful hints for connection errors
            if "Connection" in error_msg or "ConnectError" in error_msg:
                if not self.http_proxy:
                    logger.warning(
                        "Connection failed to Gemini API. "
                        "If you are in China, you may need to configure a proxy in config.yaml (gemini.proxy)"
                    )
                else:
                    logger.warning(f"Connection failed even with proxy: {self.http_proxy}")
            raise

    async def stream_request(
            self,
            request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"Sending streaming request to Gemini OpenAI: {self.base_url}", extra={'provider': f"{self.provider_name}:{self.model}"})
        logger.info(f"Tools count: {len(request.get('tools', []))}", extra={'provider': f"{self.provider_name}:{self.model}"})

        if config.chatlog_enabled:
            chatlog.info(
                f"=== STREAMING REQUEST TO GEMINI ===\n{format_json_for_log(request, max_str_length=1000)}",
                extra={'provider': f"{self.provider_name}:{self.model}"}
            )

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        yield {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        }

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        client = self._get_http_client()

        try:
            text_block_started = False
            content_block_index = 0
            total_output_tokens = 0
            tool_calls_in_progress = {}  # Track tool calls being streamed

            async with client.stream("POST", url, json=request, headers=headers, timeout=self.timeout) as response:
                response.raise_for_status()  # Raise an exception for bad status codes

                async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith('data: '):
                            continue

                        json_str = line[6:]
                        if json_str == '[DONE]':
                            break

                        try:
                            chunk = json.loads(json_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})

                        # Handle text content
                        if "content" in delta and delta["content"]:
                            content = delta["content"]
                            if not text_block_started:
                                yield {
                                    "type": "content_block_start",
                                    "index": content_block_index,
                                    "content_block": {"type": "text", "text": ""}
                                }
                                text_block_started = True

                            yield {
                                "type": "content_block_delta",
                                "index": content_block_index,
                                "delta": {"type": "text_delta", "text": content}
                            }

                        # Handle tool calls (OpenAI/Gemini OpenAI format)
                        if "tool_calls" in delta:
                            # Close text block if open
                            if text_block_started:
                                yield {
                                    "type": "content_block_stop",
                                    "index": content_block_index
                                }
                                content_block_index += 1
                                text_block_started = False

                            for tool_call in delta["tool_calls"]:
                                tc_index = tool_call.get("index", 0)

                                # New tool call starting
                                if "id" in tool_call:
                                    tool_id = tool_call["id"]
                                    func = tool_call.get("function", {})
                                    tool_name = func.get("name", "")

                                    # Extract thought_signature if present
                                    extra_content = tool_call.get("extra_content", {})
                                    google_data = extra_content.get("google", {})
                                    thought_signature = google_data.get("thought_signature")

                                    tool_calls_in_progress[tc_index] = {
                                        "id": tool_id,
                                        "name": tool_name,
                                        "arguments": "",
                                        "thought_signature": thought_signature
                                    }

                                    # Store thought_signature immediately if present
                                    if thought_signature:
                                        logger.info(f"Storing thought_signature for streaming tool_call {tool_id}")
                                        self.thought_cache.store_signature(tool_id, thought_signature, msg_id)

                                    yield {
                                        "type": "content_block_start",
                                        "index": content_block_index + tc_index,
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": tool_id,
                                            "name": tool_name,
                                            "input": {}
                                        }
                                    }

                                # Streaming arguments
                                if "function" in tool_call:
                                    args_delta = tool_call["function"].get("arguments", "")
                                    if args_delta and tc_index in tool_calls_in_progress:
                                        tool_calls_in_progress[tc_index]["arguments"] += args_delta
                                        yield {
                                            "type": "content_block_delta",
                                            "index": content_block_index + tc_index,
                                            "delta": {
                                                "type": "input_json_delta",
                                                "partial_json": args_delta
                                            }
                                        }

                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason:
                            # Close text block if still open
                            if text_block_started:
                                yield {
                                    "type": "content_block_stop",
                                    "index": content_block_index
                                }
                                content_block_index += 1

                            # Close all tool call blocks
                            for tc_index in tool_calls_in_progress:
                                yield {
                                    "type": "content_block_stop",
                                    "index": content_block_index + tc_index
                                }

                            stop_reason = "end_turn"
                            if finish_reason == "length":
                                stop_reason = "max_tokens"
                            elif finish_reason == "tool_calls":
                                stop_reason = "tool_use"

                            yield {
                                "type": "message_delta",
                                "delta": {
                                    "stop_reason": stop_reason,
                                    "stop_sequence": None
                                },
                                "usage": {"output_tokens": total_output_tokens}
                            }

                        yield {"type": "message_stop"}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini OpenAI streaming API error: {error_msg}")

            # Provide helpful hints for connection errors
            if "Connection" in error_msg or "connect" in error_msg.lower():
                if not self.http_proxy:
                    logger.warning(
                        "Connection failed to Gemini API. "
                        "If you are in China, you may need to configure a proxy in config.yaml (gemini.proxy)"
                    )
                else:
                    logger.warning(f"Connection failed even with proxy: {self.http_proxy}")
            raise
        finally:
            await client.aclose()

# Register strategy with factory
StrategyFactory.register("gemini", GeminiStrategy)
