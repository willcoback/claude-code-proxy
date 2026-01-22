"""End-to-end tests for the proxy server with mocked Gemini API."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiohttp import web
import aiohttp


class MockGeminiServer:
    """Mock Gemini API server for testing."""

    def __init__(self, port=19999):
        self.port = port
        self.app = web.Application()
        self.app.router.add_post('/v1beta/models/{model}:generateContent', self.handle_generate)
        self.app.router.add_post('/v1beta/models/{model}:streamGenerateContent', self.handle_stream)
        self.runner = None

    async def handle_generate(self, request):
        """Handle non-streaming generate request."""
        body = await request.json()

        # Extract the user message
        contents = body.get('contents', [])
        user_text = ""
        if contents:
            parts = contents[-1].get('parts', [])
            if parts:
                user_text = parts[0].get('text', '')

        # Generate mock response
        response = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": f"Mock response to: {user_text[:50]}"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30
            }
        }

        return web.json_response(response)

    async def handle_stream(self, request):
        """Handle streaming generate request."""
        body = await request.json()

        # Generate mock streaming response
        async def stream_response():
            # First chunk - partial text
            chunk1 = {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello "}]
                    }
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15
                }
            }
            yield f"data: {json.dumps(chunk1)}\n\n"

            # Second chunk - more text with finish reason
            chunk2 = {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "World!"}]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 20
                }
            }
            yield f"data: {json.dumps(chunk2)}\n\n"

        response = web.StreamResponse()
        response.content_type = 'text/event-stream'
        await response.prepare(request)

        async for chunk in stream_response():
            await response.write(chunk.encode())

        await response.write_eof()
        return response

    async def start(self):
        """Start the mock server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', self.port)
        await site.start()
        print(f"Mock Gemini server started on port {self.port}")

    async def stop(self):
        """Stop the mock server."""
        if self.runner:
            await self.runner.cleanup()


def test_proxy_with_mock():
    """Test the full proxy flow with a mocked Gemini API."""
    import uvicorn

    # Update config to use mock server
    from proxy.utils import config
    config.load()
    config._config['gemini']['base_url'] = 'http://127.0.0.1:19999/v1beta'
    config._config['gemini']['api_key'] = 'mock_key'

    mock_server = MockGeminiServer(port=19999)

    async def run_test():
        # Start mock Gemini server
        await mock_server.start()

        try:
            # Start proxy server
            from main import app
            proxy_config = uvicorn.Config(app, host='127.0.0.1', port=18081, log_level='warning')
            proxy_server = uvicorn.Server(proxy_config)

            # Run proxy in background
            proxy_task = asyncio.create_task(proxy_server.serve())

            # Wait for server to start
            await asyncio.sleep(1)

            # Test non-streaming request
            async with aiohttp.ClientSession() as session:
                claude_request = {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": "Test message"}
                    ]
                }

                async with session.post(
                        'http://127.0.0.1:18081/v1/messages',
                        json=claude_request
                ) as response:
                    assert response.status == 200, f"Expected 200, got {response.status}"
                    data = await response.json()

                    # Verify Claude response format
                    assert data.get('type') == 'message'
                    assert data.get('role') == 'assistant'
                    assert 'content' in data
                    assert len(data['content']) > 0
                    assert data['content'][0].get('type') == 'text'
                    assert 'usage' in data

                    print(f"✓ Non-streaming test passed")
                    print(f"  Response: {data['content'][0]['text'][:50]}")
                    print(f"  Tokens: {data['usage']}")

            # Test streaming request
            async with aiohttp.ClientSession() as session:
                claude_request = {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1024,
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Stream test"}
                    ]
                }

                async with session.post(
                        'http://127.0.0.1:18081/v1/messages',
                        json=claude_request
                ) as response:
                    assert response.status == 200, f"Expected 200, got {response.status}"

                    events = []
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            event_data = json.loads(line[6:])
                            events.append(event_data)

                    # Verify streaming events
                    event_types = [e.get('type') for e in events]
                    assert 'message_start' in event_types
                    assert 'content_block_start' in event_types or 'content_block_delta' in event_types
                    assert 'message_stop' in event_types

                    print(f"✓ Streaming test passed")
                    print(f"  Events received: {len(events)}")
                    print(f"  Event types: {event_types}")

            # Graceful shutdown
            proxy_server.should_exit = True
            await proxy_task

        finally:
            await mock_server.stop()

    asyncio.run(run_test())
    print("\n✓ All E2E tests passed!")


if __name__ == "__main__":
    print("Running E2E tests with mock Gemini API...\n")
    test_proxy_with_mock()
