"""
Logging Configuration
Centralized logging setup for the application
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import os


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    audit_log:  bool = True
) -> logging.Logger:
    """
    Configure logging for the application
    
    Args:
        log_level:  Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        log_to_file: Whether to log to files
        log_to_console:  Whether to log to console
        audit_log: Whether to enable separate audit logging
    
    Returns:
        Configured root logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create log directory
    if log_dir is None:
        log_dir = Path("./logs")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler. setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler (rotating)
    if log_to_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler (only errors and above)
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "error.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler. setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
    
    # Audit logger (separate logger for security events)
    if audit_log:
        audit_logger = logging.getLogger("guardrails.audit")
        audit_logger.setLevel(logging. INFO)
        audit_logger.propagate = False  # Don't propagate to root
        
        # Audit file handler
        audit_handler = logging.handlers.RotatingFileHandler(
            log_dir / "audit" / "security_audit.log",
            maxBytes=50 * 1024 * 1024,  # 50MB for audit logs
            backupCount=10,
            encoding='utf-8'
        )
        
        # JSON-like format for audit logs
        audit_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger. addHandler(audit_handler)
        
        # Also create audit directory
        (log_dir / "audit").mkdir(exist_ok=True)
    
    # Set levels for noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Log initial message
    root_logger.info(f"Logging configured:  level={log_level}, dir={log_dir}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)
