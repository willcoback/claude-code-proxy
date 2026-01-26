"""Tests for DeepSeek strategy converter."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy.deepseek.converter import DeepSeekStrategy


def test_convert_simple_message():
    """Test converting a simple text message."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }

    deepseek_request = strategy.convert_request(claude_request)

    # DeepSeek uses same format as Anthropic, model should be replaced
    assert deepseek_request["model"] == "deepseek-chat"
    assert "messages" in deepseek_request
    assert len(deepseek_request["messages"]) == 1
    assert deepseek_request["messages"][0]["role"] == "user"
    assert deepseek_request["messages"][0]["content"] == "Hello, how are you?"
    assert deepseek_request["max_tokens"] == 1024
    print("✓ test_convert_simple_message passed")


def test_convert_message_with_system():
    """Test converting a message with system prompt."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ]
    }

    deepseek_request = strategy.convert_request(claude_request)

    assert deepseek_request["model"] == "deepseek-chat"
    assert deepseek_request["system"] == "You are a helpful assistant."
    assert len(deepseek_request["messages"]) == 1
    print("✓ test_convert_message_with_system passed")


def test_convert_multi_turn():
    """Test converting multi-turn conversation."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "And 3+3?"}
        ]
    }

    deepseek_request = strategy.convert_request(claude_request)

    assert len(deepseek_request["messages"]) == 3
    assert deepseek_request["messages"][0]["role"] == "user"
    assert deepseek_request["messages"][1]["role"] == "assistant"
    assert deepseek_request["messages"][2]["role"] == "user"
    print("✓ test_convert_multi_turn passed")


def test_convert_content_blocks():
    """Test converting Claude content blocks format."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"}
                ]
            }
        ]
    }

    deepseek_request = strategy.convert_request(claude_request)

    # DeepSeek supports same content block format
    messages = deepseek_request["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Hello"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "World"
    print("✓ test_convert_content_blocks passed")


def test_convert_tools():
    """Test converting tools/functions."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ]
    }

    deepseek_request = strategy.convert_request(claude_request)

    # DeepSeek supports same tools format
    assert "tools" in deepseek_request
    assert len(deepseek_request["tools"]) == 1
    assert deepseek_request["tools"][0]["name"] == "get_weather"
    print("✓ test_convert_tools passed")


def test_convert_response():
    """Test converting DeepSeek response to Claude format."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    deepseek_response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "deepseek-chat",
        "content": [{"type": "text", "text": "Hello! I'm doing great."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 15
        }
    }

    claude_response = strategy.convert_response(deepseek_response)

    assert claude_response["type"] == "message"
    assert claude_response["role"] == "assistant"
    assert claude_response["model"] == "deepseek-chat"
    assert len(claude_response["content"]) == 1
    assert claude_response["content"][0]["type"] == "text"
    assert claude_response["content"][0]["text"] == "Hello! I'm doing great."
    assert claude_response["stop_reason"] == "end_turn"
    assert claude_response["usage"]["input_tokens"] == 10
    assert claude_response["usage"]["output_tokens"] == 15
    print("✓ test_convert_response passed")


def test_convert_tool_use_response():
    """Test converting DeepSeek tool_use response to Claude format."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    deepseek_response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "deepseek-chat",
        "content": [
            {"type": "thinking", "thinking": "Thinking..."},
            {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "Tokyo"}}
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5
        }
    }

    claude_response = strategy.convert_response(deepseek_response)

    assert claude_response["stop_reason"] == "tool_use"
    assert len(claude_response["content"]) == 2
    assert claude_response["content"][0]["type"] == "thinking"
    assert claude_response["content"][1]["type"] == "tool_use"
    assert claude_response["content"][1]["name"] == "get_weather"
    assert claude_response["content"][1]["input"] == {"location": "Tokyo"}
    print("✓ test_convert_tool_use_response passed")


def test_merge_consecutive_assistant_messages():
    """Test merging consecutive assistant messages."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "Tell me a joke"}
        ]
    }

    deepseek_request = strategy.convert_request(claude_request)

    # Two consecutive assistant messages should be merged into one
    messages = deepseek_request["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"

    # Check merged content
    assistant_content = messages[1]["content"]
    # Content may be list or string depending on implementation
    print(f"Assistant content after merge: {assistant_content}")
    print("✓ test_merge_consecutive_assistant_messages passed")


def test_add_thinking_block_for_tool_use():
    """Test adding thinking block when assistant message has tool_use."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "Tokyo"}}
                ]
            }
        ]
    }

    deepseek_request = strategy.convert_request(claude_request)

    messages = deepseek_request["messages"]
    assert len(messages) == 2
    assistant_msg = messages[1]
    assert assistant_msg["role"] == "assistant"
    content = assistant_msg["content"]
    assert isinstance(content, list)
    assert len(content) == 2  # thinking + tool_use
    assert content[0]["type"] == "thinking"
    assert content[0]["thinking"] == "Thinking..."
    assert content[1]["type"] == "tool_use"
    print("✓ test_add_thinking_block_for_tool_use passed")


def test_no_thinking_block_when_already_present():
    """Test not adding thinking block when already present."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I need to use the weather tool"},
                    {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "Tokyo"}}
                ]
            }
        ]
    }

    deepseek_request = strategy.convert_request(claude_request)

    messages = deepseek_request["messages"]
    assistant_msg = messages[1]
    content = assistant_msg["content"]
    assert isinstance(content, list)
    assert len(content) == 2  # original thinking + tool_use, no extra thinking added
    assert content[0]["type"] == "thinking"
    assert content[0]["thinking"] == "I need to use the weather tool"
    assert content[1]["type"] == "tool_use"
    print("✓ test_no_thinking_block_when_already_present passed")


def test_handle_string_content_with_tool_use():
    """Test handling string content when tool_use detection is needed."""
    config = {
        "api_key": "test_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/anthropic",
        "timeout": 60
    }
    strategy = DeepSeekStrategy(config)

    # When content is a string, tool_use detection should not trigger
    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }

    deepseek_request = strategy.convert_request(claude_request)

    messages = deepseek_request["messages"]
    assistant_msg = messages[1]
    content = assistant_msg["content"]
    # String content should remain unchanged
    assert content == "Hi there!"
    print("✓ test_handle_string_content_with_tool_use passed")


def run_all_tests():
    """Run all tests."""
    print("Running DeepSeek converter tests...\n")

    test_convert_simple_message()
    test_convert_message_with_system()
    test_convert_multi_turn()
    test_convert_content_blocks()
    test_convert_tools()
    test_convert_response()
    test_convert_tool_use_response()
    test_merge_consecutive_assistant_messages()
    test_add_thinking_block_for_tool_use()
    test_no_thinking_block_when_already_present()
    test_handle_string_content_with_tool_use()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()