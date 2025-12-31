"""
Logging configuration for the RAG application.
Provides structured logging with different levels and formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Useful for log aggregation and analysis tools.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


def setup_logging(
    name: str = "rag",
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    rotation: str = "size"  # "size" or "time"
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs to console only.
        json_format: Whether to use JSON format for logs
        rotation: Log rotation strategy - "size" or "time"
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging("rag.embeddings", level="DEBUG")
        >>> logger.info("Embedding document", extra={"doc_id": "123"})
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if rotation == "size":
            # Rotate when file reaches 10MB, keep 5 backups
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
        else:
            # Rotate daily, keep 30 days
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=30
            )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class RAGLogger:
    """
    Context-aware logger for RAG operations.
    Automatically adds context like query_id, user_id, etc.
    """
    
    def __init__(self, name: str, context: Optional[dict] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal log method that adds context"""
        extra = {**self.context, **kwargs}
        self.logger.log(level, message, extra={'extra': extra})
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def set_context(self, **kwargs):
        """Update logging context"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        self.context.clear()


# Create default logger
default_logger = setup_logging()
