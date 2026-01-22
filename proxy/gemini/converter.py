"""Gemini model strategy implementation."""

import asyncio
import json
import random
import uuid
from typing import Any, AsyncGenerator, Dict, List

import aiohttp

from ..base.strategy import BaseModelStrategy, StrategyFactory, TokenUsage, ProxyResponse
from ..utils import get_logger

logger = get_logger()


def truncate_value(value: Any, max_str_length: int = 500) -> Any:
    """
    Recursively truncate long string values while preserving JSON structure.

    Args:
        value: The value to truncate
        max_str_length: Maximum length for string values

    Returns:
        Truncated value with structure preserved
    """
    if isinstance(value, str):
        if len(value) > max_str_length:
            return value[:max_str_length] + f"... [truncated, {len(value)} chars total]"
        return value
    elif isinstance(value, dict):
        return {k: truncate_value(v, max_str_length) for k, v in value.items()}
    elif isinstance(value, list):
        # For very long lists, truncate the list itself
        if len(value) > 20:
            truncated_list = [truncate_value(item, max_str_length) for item in value[:20]]
            truncated_list.append(f"... [{len(value) - 20} more items]")
            return truncated_list
        return [truncate_value(item, max_str_length) for item in value]
    else:
        return value


def format_json_for_log(data: Any, max_str_length: int = 500) -> str:
    """
    Format JSON data for logging, truncating long field values while preserving structure.

    Args:
        data: The data to format
        max_str_length: Maximum length for individual string values

    Returns:
        Formatted JSON string with truncated values
    """
    try:
        truncated_data = truncate_value(data, max_str_length)
        return json.dumps(truncated_data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)[:5000]


# JSON Schema fields not supported by Gemini API
UNSUPPORTED_SCHEMA_FIELDS = {
    "$schema",
    "additionalProperties",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "propertyNames",
    "patternProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "const",
    "contentEncoding",
    "contentMediaType",
    "$id",
    "$ref",
    "$defs",
    "definitions",
    "if",
    "then",
    "else",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
}


def sanitize_schema_for_gemini(schema: Any) -> Any:
    """
    Recursively remove JSON Schema fields not supported by Gemini API.

    Gemini only supports a subset of OpenAPI/JSON Schema:
    - type, description, enum, properties, required, items, format
    - minimum, maximum, minItems, maxItems, minLength, maxLength
    """
    if not isinstance(schema, dict):
        return schema

    cleaned = {}
    for key, value in schema.items():
        # Skip unsupported fields
        if key in UNSUPPORTED_SCHEMA_FIELDS:
            continue

        # Recursively clean nested structures
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                prop_name: sanitize_schema_for_gemini(prop_schema)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            cleaned[key] = sanitize_schema_for_gemini(value)
        elif key == "items" and isinstance(value, list):
            cleaned[key] = [sanitize_schema_for_gemini(item) for item in value]
        elif isinstance(value, dict):
            cleaned[key] = sanitize_schema_for_gemini(value)
        elif isinstance(value, list):
            cleaned[key] = [
                sanitize_schema_for_gemini(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value

    return cleaned


class GeminiStrategy(BaseModelStrategy):
    """
    Strategy for converting between Claude and Gemini API formats.

    Handles:
    - Request format conversion (Claude -> Gemini)
    - Response format conversion (Gemini -> Claude)
    - Streaming response handling
    """

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _convert_content_to_parts(self, content: Any) -> List[Dict[str, Any]]:
        """
        Convert Claude content format to Gemini parts format.

        Claude content can be:
        - string: simple text
        - list: array of content blocks (text, image, tool_use, tool_result)
        """
        parts = []

        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    parts.append({"text": block})
                elif isinstance(block, dict):
                    block_type = block.get("type", "")

                    if block_type == "text":
                        parts.append({"text": block.get("text", "")})

                    elif block_type == "image":
                        # Handle base64 image
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            parts.append({
                                "inline_data": {
                                    "mime_type": source.get("media_type", "image/png"),
                                    "data": source.get("data", "")
                                }
                            })

                    elif block_type == "tool_use":
                        # Claude tool_use -> Gemini functionCall
                        # Gemini 3 requires thoughtSignature for function calls
                        parts.append({
                            "functionCall": {
                                "name": block.get("name", ""),
                                "args": block.get("input", {})
                            },
                            "thoughtSignature": "skip_thought_signature_validator"
                        })

                    elif block_type == "tool_result":
                        # Claude tool_result -> Gemini functionResponse
                        tool_result_content = block.get("content", "")
                        if isinstance(tool_result_content, list):
                            # Extract text from content blocks
                            text_parts = []
                            for item in tool_result_content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            tool_result_content = "\n".join(text_parts)

                        parts.append({
                            "functionResponse": {
                                "name": block.get("tool_use_id", ""),
                                "response": {"result": tool_result_content}
                            }
                        })

        return parts if parts else [{"text": ""}]

    def _convert_tools_to_gemini(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Claude tools format to Gemini function declarations.

        Claude tool schemas may include JSON Schema fields not supported by Gemini,
        such as $schema, additionalProperties, exclusiveMinimum, propertyNames, etc.
        These are sanitized before sending to Gemini.
        """
        if not tools:
            return []

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                parameters = sanitize_schema_for_gemini(func.get("parameters", {}))
                function_declarations.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": parameters
                })
            elif "name" in tool:
                # Claude native tool format
                parameters = sanitize_schema_for_gemini(tool.get("input_schema", {}))
                function_declarations.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": parameters
                })

        return [{"functionDeclarations": function_declarations}] if function_declarations else []

    def convert_request(self, claude_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Claude API request to Gemini API request.

        Claude format:
        {
            "model": "claude-3-5-sonnet-...",
            "max_tokens": 1024,
            "messages": [...],
            "system": "...",
            "tools": [...]
        }

        Gemini format:
        {
            "contents": [...],
            "systemInstruction": {...},
            "generationConfig": {...},
            "tools": [...]
        }
        """
        gemini_request = {}

        # Convert messages to contents
        contents = []
        messages = claude_request.get("messages", [])

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map Claude roles to Gemini roles
            gemini_role = "model" if role == "assistant" else "user"

            parts = self._convert_content_to_parts(content)
            contents.append({
                "role": gemini_role,
                "parts": parts
            })

        gemini_request["contents"] = contents

        # Convert system prompt
        system = claude_request.get("system")
        if system:
            if isinstance(system, str):
                gemini_request["systemInstruction"] = {
                    "parts": [{"text": system}]
                }
            elif isinstance(system, list):
                # Handle system as array of content blocks
                system_text = []
                for block in system:
                    if isinstance(block, str):
                        system_text.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        system_text.append(block.get("text", ""))
                gemini_request["systemInstruction"] = {
                    "parts": [{"text": "\n".join(system_text)}]
                }

        # Generation config
        generation_config = {}

        if "max_tokens" in claude_request:
            generation_config["maxOutputTokens"] = claude_request["max_tokens"]

        if "temperature" in claude_request:
            generation_config["temperature"] = claude_request["temperature"]

        if "top_p" in claude_request:
            generation_config["topP"] = claude_request["top_p"]

        if "top_k" in claude_request:
            generation_config["topK"] = claude_request["top_k"]

        if "stop_sequences" in claude_request:
            generation_config["stopSequences"] = claude_request["stop_sequences"]

        if generation_config:
            gemini_request["generationConfig"] = generation_config

        # Convert tools
        tools = claude_request.get("tools", [])
        if tools:
            gemini_tools = self._convert_tools_to_gemini(tools)
            if gemini_tools:
                gemini_request["tools"] = gemini_tools

        return gemini_request

    def _convert_parts_to_content(self, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Gemini parts to Claude content blocks."""
        content_blocks = []

        for part in parts:
            if "text" in part:
                content_blocks.append({
                    "type": "text",
                    "text": part["text"]
                })

            elif "functionCall" in part:
                func_call = part["functionCall"]
                content_blocks.append({
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": func_call.get("name", ""),
                    "input": func_call.get("args", {})
                })

        return content_blocks

    def convert_response(self, gemini_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Gemini API response to Claude API response.

        Gemini format:
        {
            "candidates": [{
                "content": {"role": "model", "parts": [...]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {...}
        }

        Claude format:
        {
            "id": "msg_...",
            "type": "message",
            "role": "assistant",
            "content": [...],
            "model": "...",
            "stop_reason": "...",
            "usage": {...}
        }
        """
        # Generate unique message ID
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Extract content from candidates
        content_blocks = []
        stop_reason = "end_turn"

        candidates = gemini_response.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            content_blocks = self._convert_parts_to_content(parts)

            # Map finish reason
            finish_reason = candidate.get("finishReason", "STOP")
            stop_reason_map = {
                "STOP": "end_turn",
                "MAX_TOKENS": "max_tokens",
                "SAFETY": "stop_sequence",
                "RECITATION": "stop_sequence",
                "OTHER": "end_turn"
            }
            stop_reason = stop_reason_map.get(finish_reason, "end_turn")

            # Check if there's a tool use, adjust stop_reason
            for block in content_blocks:
                if block.get("type") == "tool_use":
                    stop_reason = "tool_use"
                    break

        # Extract usage
        usage_metadata = gemini_response.get("usageMetadata", {})
        usage = {
            "input_tokens": usage_metadata.get("promptTokenCount", 0),
            "output_tokens": usage_metadata.get("candidatesTokenCount", 0)
        }

        claude_response = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
            "model": self.model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage
        }

        return claude_response

    def get_token_usage(self, response: Dict[str, Any]) -> TokenUsage:
        """Extract token usage from Gemini response."""
        usage_metadata = response.get("usageMetadata", {})
        return TokenUsage(
            input_tokens=usage_metadata.get("promptTokenCount", 0),
            output_tokens=usage_metadata.get("candidatesTokenCount", 0),
            total_tokens=usage_metadata.get("totalTokenCount", 0)
        )

    def _get_api_url(self, stream: bool = False) -> str:
        """Get the API URL for Gemini."""
        action = "streamGenerateContent" if stream else "generateContent"
        url = f"{self.base_url}/models/{self.model}:{action}"
        if stream:
            url += "?alt=sse"
        return url

    async def _make_post_request(self, url: str, request: Dict[str, Any], headers: Dict[str, str], max_retries: int = 5) -> aiohttp.ClientResponse:
        """Helper for POST requests with retry on transient errors (429, 5xx)."""
        transient_codes = {429, 502, 503, 504}
        session = None
        try:
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            for attempt in range(max_retries + 1):
                async with session.post(url, json=request, headers=headers) as response:
                    if response.status not in transient_codes:
                        return response

                    response_text = await response.text()
                    delay = min(5 * (2 ** attempt) * random.uniform(0.8, 1.2), 120)
                    logger.warning(f"Gemini transient ({response.status}): attempt {attempt + 1}/{max_retries + 1}, backoff {delay:.1f}s - {response_text[:200]}...")

                    if attempt == max_retries:
                        logger.error(f"Gemini max retries exceeded: {response.status}")
                        raise Exception(f"Gemini API failed after {max_retries + 1} attempts: {response.status}")

                    await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Request failed (final): {str(e)}")
            raise
        finally:
            if session:
                await session.close()

    async def send_request(self, request: Dict[str, Any]) -> ProxyResponse:
        """Send non-streaming request to Gemini API."""
        url = self._get_api_url(stream=False)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        logger.info(f"Sending request to Gemini: {url}")
        logger.info(f"=== CONVERTED GEMINI REQUEST ===\n{format_json_for_log(request)}")

        response = await self._make_post_request(url, request, headers)
        response_text = await response.text()

        logger.info(f"=== RAW GEMINI RESPONSE ===\n{format_json_for_log(response_text, max_str_length=15000)}")

        gemini_response = json.loads(response_text)
        usage = self.get_token_usage(gemini_response)
        claude_response = self.convert_response(gemini_response)

        logger.info(f"=== CONVERTED CLAUDE RESPONSE ===\n{format_json_for_log(claude_response)}")

        return ProxyResponse(
            data=claude_response,
            usage=usage,
            is_stream=False
        )

    async def stream_request(
            self,
            request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send streaming request to Gemini API and yield Claude-formatted chunks."""
        url = self._get_api_url(stream=True)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        logger.info(f"Sending streaming request to Gemini: {url}")
        logger.info(f"=== CONVERTED GEMINI REQUEST (STREAM) ===\n{format_json_for_log(request)}")

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        content_block_index = 0
        accumulated_text = ""
        tool_calls = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Send message_start event
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

        # In stream_request:
        response = await self._make_post_request(url, request, headers)
        if response.status != 200:
            error_text = await response.text()
            logger.error(f"Gemini streaming API error: {response.status} - {error_text}")
            raise Exception(f"Gemini API error: {response.status} - {error_text}")

        text_block_started = False

        async for line in response.content:
            line = line.decode('utf-8').strip()

            if not line or not line.startswith('data: '):
                continue

            json_str = line[6:]  # Remove 'data: ' prefix
            if not json_str or json_str == '[DONE]':
                continue

            try:
                chunk = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            # Log streaming chunk
            logger.debug(f"=== GEMINI STREAM CHUNK ===\n{format_json_for_log(chunk, max_str_length=2000)}")

            # Extract usage metadata
            usage_metadata = chunk.get("usageMetadata", {})
            if usage_metadata:
                total_input_tokens = usage_metadata.get("promptTokenCount", total_input_tokens)
                total_output_tokens = usage_metadata.get("candidatesTokenCount", total_output_tokens)

            candidates = chunk.get("candidates", [])
            if not candidates:
                continue

            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                if "text" in part:
                    text = part["text"]

                    if not text_block_started:
                        # Start content block
                        yield {
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": {"type": "text", "text": ""}
                        }
                        text_block_started = True

                    # Send text delta
                    yield {
                        "type": "content_block_delta",
                        "index": content_block_index,
                        "delta": {"type": "text_delta", "text": text}
                    }
                    accumulated_text += text

                elif "functionCall" in part:
                    # End previous text block if any
                    if text_block_started:
                        yield {
                            "type": "content_block_stop",
                            "index": content_block_index
                        }
                        content_block_index += 1
                        text_block_started = False

                    # Start tool use block
                    func_call = part["functionCall"]
                    tool_id = f"toolu_{uuid.uuid4().hex[:24]}"

                    yield {
                        "type": "content_block_start",
                        "index": content_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": func_call.get("name", ""),
                            "input": {}
                        }
                    }

                    # Send input as delta
                    input_json = json.dumps(func_call.get("args", {}))
                    yield {
                        "type": "content_block_delta",
                        "index": content_block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": input_json
                        }
                    }

                    yield {
                        "type": "content_block_stop",
                        "index": content_block_index
                    }

                    tool_calls.append({
                        "id": tool_id,
                        "name": func_call.get("name", ""),
                        "input": func_call.get("args", {})
                    })
                    content_block_index += 1

            # Check for finish
            finish_reason = candidate.get("finishReason")
            if finish_reason:
                if text_block_started:
                    yield {
                        "type": "content_block_stop",
                        "index": content_block_index
                    }

                # Map stop reason
                stop_reason_map = {
                    "STOP": "end_turn",
                    "MAX_TOKENS": "max_tokens",
                    "SAFETY": "stop_sequence"
                }
                stop_reason = stop_reason_map.get(finish_reason, "end_turn")
                if tool_calls:
                    stop_reason = "tool_use"

                # Send message_delta with stop_reason
                yield {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": None
                    },
                    "usage": {"output_tokens": total_output_tokens}
                }

        # Send message_stop
        yield {
            "type": "message_stop"
        }

        # Store final usage for logging
        self._last_usage = TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens
        )


# Register strategy with factory
StrategyFactory.register("gemini", GeminiStrategy)
