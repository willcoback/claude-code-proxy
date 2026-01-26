"""
Claude Code Proxy - Main Entry Point

This proxy server receives requests from Claude Code, converts them to
target model format (e.g., Gemini), and returns responses in Claude format.
"""

import json
import uuid
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from proxy.utils import setup_logger, log_request, config


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


from proxy.base import StrategyFactory
# Strategies are automatically discovered and registered via proxy/__init__.py

# Load configuration at module load time
config.load()

# Initialize logger
logger = setup_logger(
    name="claude-code-proxy",
    log_dir=config.get("logging.dir", "./logs"),
    level=config.get("logging.level", "INFO"),
    provider=config.get_provider_config().get('model', 'system')
)

# Create FastAPI app
app = FastAPI(
    title="Claude Code Proxy",
    description="Proxy server for converting Claude Code requests to other LLM providers",
    version="1.0.0"
)


def get_strategy():
    """Get the configured model strategy, ensuring it's always up-to-date."""
    # Ensure config is reloaded if file has changed
    config.check_and_reload()
    provider_name = config.provider_name
    provider_config = config.get_provider_config(provider_name)
    return StrategyFactory.get_strategy(provider_name, provider_config)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("=" * 60)
    logger.info("Claude Code Proxy Starting")
    logger.info(f"Provider: {config.provider_name}")
    logger.info(f"Model: {config.get_provider_config().get('model', 'unknown')}")
    logger.info(f"Available providers: {StrategyFactory.list_providers()}")
    logger.info("=" * 60)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "provider": config.provider_name}


@app.post("/v1/messages")
async def messages(request: Request):
    """
    Main proxy endpoint for Claude API messages.

    Receives Claude-formatted requests, converts to target model format,
    sends to target API, and converts response back to Claude format.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()

    try:
        # Parse request body
        body = await request.json()
        stream = body.get("stream", False)

        logger.info(f"[{request_id}] Received request | stream={stream} | model={body.get('model', 'unknown')}", extra={'provider': 'proxy'})

        # Check and reload configuration if changed
        if config.check_and_reload():
            logger.info(f"[{request_id}] Configuration reloaded from file", extra={'provider': 'config'})

        providers = [config.provider_name] + config.get("provider.fallback_providers", [])
        used_provider = None
        used_model = None

        for provider_name in providers:
            try:
                provider_config = config.get_provider_config(provider_name)
                strategy = StrategyFactory.get_strategy(provider_name, provider_config)
                model_name = strategy.model
                used_provider = provider_name
                used_model = model_name
                logger.info(f"[{request_id}] Using provider: {provider_name} ({model_name})", extra={'provider': f"{provider_name}:{model_name}"})

                if stream:
                    # Handle streaming response
                    async def generate_stream():
                        total_input_tokens = 0
                        total_output_tokens = 0

                        try:
                            # Convert request and get streaming response
                            target_request = strategy.convert_request(body)
                            async for chunk in strategy.stream_request(target_request):
                                # Track tokens from message_start
                                if chunk.get("type") == "message_start":
                                    usage = chunk.get("message", {}).get("usage", {})
                                    total_input_tokens = usage.get("input_tokens", 0)

                                # Track tokens from message_delta
                                if chunk.get("type") == "message_delta":
                                    usage = chunk.get("usage", {})
                                    total_output_tokens = usage.get("output_tokens", total_output_tokens)

                                # Send chunk as SSE with correct event type
                                chunk_json = json.dumps(chunk, ensure_ascii=False)
                                event_type = chunk.get("type", "message")
                                yield f"event: {event_type}\ndata: {chunk_json}\n\n"

                            # Get final usage if available
                            if hasattr(strategy, '_last_usage'):
                                total_input_tokens = strategy._last_usage.input_tokens
                                total_output_tokens = strategy._last_usage.output_tokens

                            # Log request
                            log_request(
                                logger=logger,
                                model_name=f"{used_provider}:{used_model}",
                                input_tokens=total_input_tokens,
                                output_tokens=total_output_tokens,
                                request_id=request_id,
                                status="success"
                            )

                        except Exception as e:
                            logger.error(f"[{request_id}] Streaming error with {provider_name}: {str(e)}")
                            error_chunk = {
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": str(e)
                                }
                            }
                            yield f"event: error\ndata: {json.dumps(error_chunk)}\n\n"

                            log_request(
                                logger=logger,
                                model_name=f"{used_provider}:{used_model}",
                                input_tokens=0,
                                output_tokens=0,
                                request_id=request_id,
                                status="error"
                            )

                    return StreamingResponse(
                        generate_stream(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Request-Id": request_id
                        }
                    )

                else:
                    # Handle non-streaming response
                    response = await strategy.proxy(body, stream=False)

                    # Log request
                    log_request(
                        logger=logger,
                        model_name=f"{used_provider}:{used_model}",
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        request_id=request_id,
                        status="success"
                    )

                    logger.info(
                        f"[{request_id}] Request completed | "
                        f"provider: {used_provider} | "
                        f"tokens: {response.usage.total_tokens}",
                        extra={'provider': f"{used_provider}:{used_model}"}
                    )

                    return JSONResponse(
                        content=response.data,
                        headers={"X-Request-Id": request_id}
                    )

            except Exception as e:
                logger.warning(f"[{request_id}] Provider {provider_name} failed: {str(e)} | Trying next...")
                continue

        # All providers failed
        raise HTTPException(
            status_code=500,
            detail=f"All providers failed: {', '.join(providers)}"
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error processing request: {str(e)}", exc_info=True)

        log_request(
            logger=logger,
            model_name=config.get_provider_config().get('model', 'unknown'),
            input_tokens=0,
            output_tokens=0,
            request_id=request_id,
            status="error"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "type": "api_error",
                "message": str(e),
                "request_id": request_id
            }
        )


@app.get("/v1/models")
async def list_models():
    """List available models (returns the configured model as Claude model)."""
    model_id = config.get_provider_config().get('model', 'claude-3-5-sonnet-20241022')
    return {
        "object": "list",
        "data": [
            {
                "id": "claude-3-5-sonnet-20241022",
                "object": "model",
                "created": 1698959748,
                "owned_by": "anthropic",
                "proxied_to": model_id
            },
            {
                "id": "claude-3-opus-20240229",
                "object": "model",
                "created": 1698959748,
                "owned_by": "anthropic",
                "proxied_to": model_id
            }
        ]
    }


@app.post("/api/event_logging/batch")
async def event_logging_batch():
    """Handle Claude Code telemetry events (no-op, just acknowledge)."""
    return {"status": "ok"}


def main():
    """Main entry point."""
    # Run server
    uvicorn.run(
        "main:app",
        host=config.server_host,
        port=config.server_port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
