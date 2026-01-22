"""Logging module with daily rotating file handler."""

import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class ProxyLogger:
    """Custom logger for claude-code-proxy with daily rotation."""

    _loggers = {}

    @classmethod
    def setup_logger(
            cls,
            name: str = "claude-code-proxy",
            log_dir: str = "./logs",
            level: str = "INFO"
    ) -> logging.Logger:
        """
        Setup and return a logger with console and daily rotating file handlers.

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level

        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]

        # Create logs directory if not exists
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers = []  # Clear existing handlers

        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # Daily rotating file handler
        log_file = log_path / f"proxy_{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.suffix = "%Y-%m-%d.log"
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def get_logger(cls, name: str = "claude-code-proxy") -> logging.Logger:
        """Get an existing logger or create a new one."""
        if name not in cls._loggers:
            return cls.setup_logger(name)
        return cls._loggers[name]


def setup_logger(
        name: str = "claude-code-proxy",
        log_dir: str = "./logs",
        level: str = "INFO"
) -> logging.Logger:
    """Convenience function to setup logger."""
    return ProxyLogger.setup_logger(name, log_dir, level)


def get_logger(name: str = "claude-code-proxy") -> logging.Logger:
    """Convenience function to get logger."""
    return ProxyLogger.get_logger(name)


def log_request(
        logger: logging.Logger,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        request_id: str = "",
        status: str = "success"
):
    """
    Log a proxy request with standard format.

    Args:
        logger: Logger instance
        model_name: Name of the target model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens consumed
        request_id: Unique request identifier
        status: Request status (success/error)
    """
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    logger.info(
        f"REQUEST | model={model_name} | "
        f"input_tokens={input_tokens} | output_tokens={output_tokens} | "
        f"total_tokens={total_tokens} | status={status} | request_id={request_id}"
    )
