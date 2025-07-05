"""
Logging configuration for LangGraph Lab.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from ..config.settings import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    rich_console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration with Rich console support.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        rich_console: Whether to use Rich console for colorized output
        
    Returns:
        Configured logger instance
    """
    settings = get_settings()
    
    # Use provided log level or fall back to settings
    level = log_level or settings.log_level
    
    # Create logger
    logger = logging.getLogger("langgraph_lab")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Configure formatters
    if rich_console:
        # Rich handler for colorized console output
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True
        )
        rich_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(rich_handler)
    else:
        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str = "langgraph_lab") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Default logger instance
logger = setup_logging()