"""Logging module with daily rotating file handler."""

import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class ProviderFilter(logging.Filter):
    def __init__(self, provider: str = "system"):
        super().__init__()
        self.provider = provider

    def filter(self, record):
        # Ensure provider field always exists
        if not hasattr(record, 'provider'):
            record.provider = self.provider
        return True


class HourlyRotatingFileHandler(TimedRotatingFileHandler):
    """Custom hourly rotating file handler with timestamp in filename."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set suffix format for rotation
        self.suffix = "%Y-%m-%d_%H"
        self.extMatch = None

    def rotation_filename(self, source_filename, date_str):
        """Generate rotated filename with timestamp."""
        log_path = Path(source_filename).parent
        basename = Path(source_filename).stem.split('-')[0]  # Get 'proxy' from 'proxy-2026-01-23_13'
        return str(log_path / f"{basename}-{date_str}.log")

    def emit(self, record):
        """
        Emit a record, checking if log file still exists before writing.

        If the log file was manually deleted, recreate it.
        """
        try:
            # Check if the log file still exists
            if self.stream and not Path(self.baseFilename).exists():
                # File was deleted, close current stream and reopen
                self.stream.close()
                self.stream = None

            # If stream is None, open the file
            if self.stream is None:
                self.stream = self._open()

            # Call parent's emit to actually write the log
            super().emit(record)
        except Exception:
            # If anything goes wrong, call parent's handleError
            self.handleError(record)

    def shouldRollover(self, record):
        """
        Check if rollover needed.
        Override parent's shouldRollover to check filename changes.

        Args:
            record: Log record (required by parent class, unused here)

        Returns:
            True if rollover needed, False otherwise
        """
        return self.get_current_filename() != self.baseFilename

    def doRollover(self):
        """Override to handle rollover with custom filename format."""
        if self.stream:
            self.stream.close()
            self.stream = None

        # Get current time for new filename
        current_time_filename = self.get_current_filename()

        # If the filename has changed, update baseFilename
        if current_time_filename != self.baseFilename:
            self.baseFilename = current_time_filename

        # Open new log file
        self.mode = 'a'
        self.stream = self._open()

        # Calculate next rollover time
        current_time_sec = datetime.now().timestamp()
        new_rollover_at = self.computeRollover(current_time_sec)
        while new_rollover_at <= current_time_sec:
            new_rollover_at = new_rollover_at + self.interval
        self.rolloverAt = new_rollover_at

    def get_current_filename(self):
        """Get filename for current hour."""
        log_path = Path(self.baseFilename).parent
        basename = Path(self.baseFilename).stem.split('-')[0]  # Get 'proxy' part
        current_hour = datetime.now().strftime('%Y-%m-%d_%H')
        return str(log_path / f"{basename}-{current_hour}.log")


class ProxyLogger:
    """Custom logger for claude-code-proxy with hourly rotation."""

    _loggers = {}
    _chatlog_logger = None

    @classmethod
    def setup_logger(
            cls,
            name: str = "claude-code-proxy",
            log_dir: str = "./logs",
            level: str = "INFO",
            provider: str = None
    ) -> logging.Logger:
        """
        Setup and return a logger with console and daily rotating file handlers.

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            provider: Provider name for log format

        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            logger = cls._loggers[name]
            # Update provider filter if it exists
            for f in logger.filters:
                if isinstance(f, ProviderFilter):
                    f.provider = provider if provider else "system"
                    break
            else:
                # No filter found, add one
                logger.addFilter(ProviderFilter(provider if provider else "system"))
            return logger

        # Create logs directory if not exists
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers = []  # Clear existing handlers

        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s|%(provider)s| | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s|%(provider)s| | %(message)s',
            datefmt='%H:%M:%S'
        )

        # Hourly rotating file handler with timestamp in filename
        current_hour = datetime.now().strftime('%Y-%m-%d_%H')
        log_file = log_path / f"proxy-{current_hour}.log"
        file_handler = HourlyRotatingFileHandler(
            filename=str(log_file),
            when='H',
            interval=1,
            backupCount=1680,  # ~70 days
            encoding='utf-8',
            utc=False
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Always add provider filter with default value if not specified
        logger.addFilter(ProviderFilter(provider if provider else "system"))

        cls._loggers[name] = logger
        return logger

    @classmethod
    def get_logger(cls, name: str = "claude-code-proxy") -> logging.Logger:
        """Get an existing logger or create a new one."""
        if name not in cls._loggers:
            return cls.setup_logger(name)
        return cls._loggers[name]

    @classmethod
    def setup_chatlog_logger(
            cls,
            log_dir: str = "./logs",
            level: str = "INFO"
    ) -> logging.Logger:
        """
        Setup and return a chatlog logger for detailed request/response logging.

        This logger writes detailed information including:
        - Original request body
        - Converted request body
        - Model response
        - Converted response

        Args:
            log_dir: Directory for log files
            level: Logging level

        Returns:
            Configured chatlog logger instance
        """
        if cls._chatlog_logger is not None:
            return cls._chatlog_logger

        # Create logs directory if not exists
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create logger with unique name to avoid conflicts
        logger = logging.getLogger("claude-code-proxy-chatlog")
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers = []  # Clear existing handlers
        logger.propagate = False  # Don't propagate to parent loggers

        # Create formatter (simpler format for chatlog)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s|%(provider)s| | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Hourly rotating file handler with timestamp in filename
        current_hour = datetime.now().strftime('%Y-%m-%d_%H')
        log_file = log_path / f"chatlog-{current_hour}.log"
        file_handler = HourlyRotatingFileHandler(
            filename=str(log_file),
            when='H',
            interval=1,
            backupCount=1680,  # ~70 days
            encoding='utf-8',
            utc=False
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Add provider filter
        logger.addFilter(ProviderFilter("chatlog"))

        cls._chatlog_logger = logger
        return logger

    @classmethod
    def get_chatlog_logger(cls) -> logging.Logger:
        """Get the chatlog logger, creating it if necessary."""
        if cls._chatlog_logger is None:
            return cls.setup_chatlog_logger()
        return cls._chatlog_logger


def setup_logger(
        name: str = "claude-code-proxy",
        log_dir: str = "./logs",
        level: str = "INFO",
        provider: str = None
) -> logging.Logger:
    """Convenience function to setup logger."""
    return ProxyLogger.setup_logger(name, log_dir, level, provider)


def get_logger(name: str = "claude-code-proxy") -> logging.Logger:
    """Convenience function to get logger."""
    return ProxyLogger.get_logger(name)


def get_chatlog_logger() -> logging.Logger:
    """Convenience function to get chatlog logger."""
    return ProxyLogger.get_chatlog_logger()


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
        f"total_tokens={total_tokens} | status={status} | request_id={request_id}",
        extra={'provider': model_name}
    )
