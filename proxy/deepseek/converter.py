"""DeepSeek Anthropic-compatible model strategy implementation."""

import copy
import json
import uuid
from typing import Any, AsyncGenerator, Dict

import aiohttp

from ..base.strategy import BaseModelStrategy, StrategyFactory, TokenUsage, ProxyResponse
from ..utils import get_logger

logger = get_logger()


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


class DeepSeekStrategy(BaseModelStrategy):
    """
    Strategy for DeepSeek Anthropic-compatible API.
    Since DeepSeek supports Anthropic Messages API directly, conversion is minimal.
    """

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @staticmethod
    def _merge_content(content1: Any, content2: Any) -> Any:
        """
        Merge two content values according to Anthropic API format rules.

        Args:
            content1: First content (string or list of content blocks)
            content2: Second content (string or list of content blocks)

        Returns:
            Merged content in valid Anthropic format
        """
        # Convert both to normalized form (list of content blocks)
        def normalize(content):
            if isinstance(content, str):
                return [{"type": "text", "text": content}]
            elif isinstance(content, list):
                return content
            else:
                return []

        normalized1 = normalize(content1)
        normalized2 = normalize(content2)

        # Merge the two lists
        return normalized1 + normalized2

    def convert_request(self, claude_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimal conversion for DeepSeek Anthropic API.
        Directly forward Claude request with model/base_url adjustments.
        Fix consecutive assistant messages for DeepSeek compatibility.
        """
        # Log original request messages structure
        original_messages = claude_request.get("messages", [])
        logger.info(
            f"=== ORIGINAL REQUEST ({len(original_messages)} messages) ===\n"
            f"{format_json_for_log(original_messages, max_str_length=200)}",
            extra={'provider': f"{self.provider_name}:{self.model}"}
        )

        # Log system prompt if present
        system_prompt = claude_request.get("system")
        if system_prompt:
            logger.info(
                f"=== SYSTEM PROMPT ===\n{system_prompt}",
                extra={'provider': f"{self.provider_name}:{self.model}"}
            )

        # Deep copy to avoid polluting original request
        req = copy.deepcopy(claude_request)

        # Override model
        req["model"] = self.model  # e.g., "deepseek-chat"

        # Fix consecutive assistant messages and add thinking block for tool_use
        messages = req.get("messages", [])

        # Remove prefill assistant messages (partial responses for format guidance)
        # Claude Code uses prefill to guide output format, but DeepSeek doesn't support it well
        if messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant":
                content = last_msg.get("content", [])
                # Check if this is a prefill (very short text content, typically just "{" or similar)
                if isinstance(content, list) and len(content) == 1:
                    block = content[0]
                    if (block.get("type") == "text" and
                        isinstance(block.get("text"), str) and
                        len(block["text"].strip()) <= 5):  # Very short prefill
                        logger.info(
                            f"Detected and removing prefill assistant message: {block['text']!r}",
                            extra={'provider': f"{self.provider_name}:{self.model}"}
                        )
                        messages.pop()  # Remove the prefill message

        i = 0
        while i < len(messages):
            curr_msg = messages[i]

            # If current message is assistant and has tool_use, ensure thinking block
            if curr_msg.get("role") == "assistant":
                content = curr_msg.get("content", [])
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]

                has_tool_use = any(block.get("type") == "tool_use" for block in content)
                has_thinking = any(block.get("type") == "thinking" for block in content)

                if has_tool_use and not has_thinking:
                    # Insert a thinking block at the beginning of content
                    content.insert(0, {"type": "thinking", "thinking": "Thinking..."})
                    curr_msg["content"] = content

            # Handle consecutive assistant messages for merging
            if i > 0:
                prev_msg = messages[i-1]
                if prev_msg.get("role") == "assistant" and curr_msg.get("role") == "assistant":
                    # Merge content blocks safely using helper method
                    prev_content = prev_msg.get("content", [])
                    curr_content = curr_msg.get("content", [])
                    prev_msg["content"] = self._merge_content(prev_content, curr_content)
                    # Remove duplicate assistant message
                    del messages[i]
                    # Do not increment i, as the next message is now at current i
                    continue
            i += 1
        req["messages"] = messages

        # Log converted request messages structure
        logger.info(
            f"=== CONVERTED REQUEST ({len(messages)} messages) ===\n"
            f"{format_json_for_log(messages, max_str_length=200)}",
            extra={'provider': f"{self.provider_name}:{self.model}"}
        )

        # DeepSeek ignores some fields but supports core: messages, system, tools, max_tokens, etc.
        # No major conversion needed (content blocks are compatible)

        return req

    def convert_response(self, deepseek_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimal conversion from DeepSeek response to Claude format.
        Format is compatible, so mostly identity mapping.
        """
        msg_id = deepseek_response.get("id", f"msg_{uuid.uuid4().hex[:24]}")

        content = deepseek_response.get("content", [])
        if not content:
            content = [{"type": "text", "text": ""}]

        stop_reason = deepseek_response.get("stop_reason", "end_turn")
        stop_sequence = deepseek_response.get("stop_sequence")

        usage_data = deepseek_response.get("usage", {})
        usage = {
            "input_tokens": usage_data.get("input_tokens", 0),
            "output_tokens": usage_data.get("output_tokens", 0)
        }

        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": self.model,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "usage": usage
        }

    def get_token_usage(self, response: Dict[str, Any]) -> TokenUsage:
        usage_data = response.get("usage", {})
        return TokenUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )

    async def send_request(self, request: Dict[str, Any]) -> ProxyResponse:
        url = f"{self.base_url}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,  # Preferred for DeepSeek
            "anthropic-version": "2023-06-01"  # Compatibility
        }

        logger.info(f"Sending request to DeepSeek: {url}", extra={'provider': f"{self.provider_name}:{self.model}"})
        logger.info(
            f"=== FULL REQUEST TO DEEPSEEK ===\n{format_json_for_log(request, max_str_length=300)}",
            extra={'provider': f"{self.provider_name}:{self.model}"}
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response_text = await response.text()

                if response.status != 200:
                    logger.error(f"DeepSeek API error: {response.status} - {response_text}")
                    raise Exception(f"DeepSeek API error: {response.status} - {response_text}")

                logger.info(f"=== RAW DEEPSEEK RESPONSE ===\n{format_json_for_log(response_text)}")

                deepseek_response = json.loads(response_text)
                usage = self.get_token_usage(deepseek_response)
                claude_response = self.convert_response(deepseek_response)

                return ProxyResponse(
                    data=claude_response,
                    usage=usage,
                    is_stream=False
                )

    async def stream_request(
        self,
        request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        url = f"{self.base_url}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        logger.info(f"Sending streaming request to DeepSeek: {url}", extra={'provider': f"{self.provider_name}:{self.model}"})
        logger.info(f"Tools count: {len(request.get('tools', []))}", extra={'provider': f"{self.provider_name}:{self.model}"})
        logger.info(
            f"=== FULL STREAMING REQUEST TO DEEPSEEK ===\n{format_json_for_log(request, max_str_length=300)}",
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

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"DeepSeek streaming API error: {response.status} - {error_text}")
                    raise Exception(f"DeepSeek API error: {response.status} - {error_text}")


                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line.startswith('data: '):
                        continue

                    json_str = line[6:]
                    if json_str == '[DONE]':
                        break

                    try:
                        chunk = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    # DeepSeek stream format is Anthropic-compatible SSE
                    # Handle message_start, content_block_start/delta/stop, message_delta, message_stop
                    event_type = chunk.get("type")

                    if event_type == "message_start":
                        usage = chunk.get("message", {}).get("usage", {})
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
                                "usage": usage
                            }
                        }

                    elif event_type == "content_block_start":
                        yield chunk  # Direct forward

                    elif event_type == "content_block_delta":
                        yield chunk

                    elif event_type == "content_block_stop":
                        yield chunk

                    elif event_type == "message_delta":
                        yield chunk

                    elif event_type == "message_stop":
                        yield chunk

                yield {"type": "message_stop"}


# Register strategy
StrategyFactory.register("deepseek", DeepSeekStrategy)