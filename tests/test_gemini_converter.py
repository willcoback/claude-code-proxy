"""Tests for Gemini strategy converter."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy.gemini.converter import GeminiStrategy


def test_convert_simple_message():
    """Test converting a simple text message."""
    config = {
        "api_key": "test_key",
        "model": "gemini-2.0-flash",
        "base_url": "https://example.com",
        "timeout": 60
    }
    strategy = GeminiStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }

    gemini_request = strategy.convert_request(claude_request)

    assert "contents" in gemini_request
    assert len(gemini_request["contents"]) == 1
    assert gemini_request["contents"][0]["role"] == "user"
    assert gemini_request["contents"][0]["parts"][0]["text"] == "Hello, how are you?"
    assert gemini_request["generationConfig"]["maxOutputTokens"] == 1024
    print("✓ test_convert_simple_message passed")


def test_convert_message_with_system():
    """Test converting a message with system prompt."""
    config = {
        "api_key": "test_key",
        "model": "gemini-2.0-flash",
        "base_url": "https://example.com",
        "timeout": 60
    }
    strategy = GeminiStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ]
    }

    gemini_request = strategy.convert_request(claude_request)

    assert "systemInstruction" in gemini_request
    assert gemini_request["systemInstruction"]["parts"][0]["text"] == "You are a helpful assistant."
    print("✓ test_convert_message_with_system passed")


def test_convert_multi_turn():
    """Test converting multi-turn conversation."""
    config = {
        "api_key": "test_key",
        "model": "gemini-2.0-flash",
        "base_url": "https://example.com",
        "timeout": 60
    }
    strategy = GeminiStrategy(config)

    claude_request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "And 3+3?"}
        ]
    }

    gemini_request = strategy.convert_request(claude_request)

    assert len(gemini_request["contents"]) == 3
    assert gemini_request["contents"][0]["role"] == "user"
    assert gemini_request["contents"][1]["role"] == "model"  # Gemini uses 'model' instead of 'assistant'
    assert gemini_request["contents"][2]["role"] == "user"
    print("✓ test_convert_multi_turn passed")


def test_convert_content_blocks():
    """Test converting Claude content blocks format."""
    config = {
        "api_key": "test_key",
        "model": "gemini-2.0-flash",
        "base_url": "https://example.com",
        "timeout": 60
    }
    strategy = GeminiStrategy(config)

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

    gemini_request = strategy.convert_request(claude_request)

    parts = gemini_request["contents"][0]["parts"]
    assert len(parts) == 2
    assert parts[0]["text"] == "Hello"
    assert parts[1]["text"] == "World"
    print("✓ test_convert_content_blocks passed")


def test_convert_tools():
    """Test converting tools/functions."""
    config = {
        "api_key": "test_key",
        "model": "gemini-2.0-flash",
        "base_url": "https://example.com",
        "timeout": 60
    }
    strategy = GeminiStrategy(config)

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

    gemini_request = strategy.convert_request(claude_request)

    assert "tools" in gemini_request
    assert len(gemini_request["tools"]) == 1
    func_decls = gemini_request["tools"][0]["functionDeclarations"]
    assert len(func_decls) == 1
    assert func_decls[0]["name"] == "get_weather"
    print("✓ test_convert_tools passed")


def test_convert_response():
    """Test converting Gemini response to Claude format."""
    config = {
        "api_key": "test_key",
        "model": "gemini-2.0-flash",
        "base_url": "https://example.com",
        "timeout": 60
    }
    strategy = GeminiStrategy(config)

    gemini_response = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "Hello! I'm doing great."}]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 15,
            "totalTokenCount": 25
        }
    }

    claude_response = strategy.convert_response(gemini_response)

    assert claude_response["type"] == "message"
    assert claude_response["role"] == "assistant"
    assert len(claude_response["content"]) == 1
    assert claude_response["content"][0]["type"] == "text"
    assert claude_response["content"][0]["text"] == "Hello! I'm doing great."
    assert claude_response["stop_reason"] == "end_turn"
    assert claude_response["usage"]["input_tokens"] == 10
    assert claude_response["usage"]["output_tokens"] == 15
    print("✓ test_convert_response passed")


def test_convert_function_call_response():
    """Test converting Gemini function call response to Claude tool_use format."""
    config = {
        "api_key": "test_key",
        "model": "gemini-2.0-flash",
        "base_url": "https://example.com",
        "timeout": 60
    }
    strategy = GeminiStrategy(config)

    gemini_response = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "Tokyo"}
                    }
                }]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
    }

    claude_response = strategy.convert_response(gemini_response)

    assert claude_response["stop_reason"] == "tool_use"
    assert len(claude_response["content"]) == 1
    assert claude_response["content"][0]["type"] == "tool_use"
    assert claude_response["content"][0]["name"] == "get_weather"
    assert claude_response["content"][0]["input"] == {"location": "Tokyo"}
    print("✓ test_convert_function_call_response passed")


def run_all_tests():
    """Run all tests."""
    print("Running Gemini converter tests...\n")

    test_convert_simple_message()
    test_convert_message_with_system()
    test_convert_multi_turn()
    test_convert_content_blocks()
    test_convert_tools()
    test_convert_response()
    test_convert_function_call_response()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
