import logging
import sys
import os

def setup_logging(level=logging.INFO):
    """
    Sets up the global logging configuration for the application.
    Logs to stdout with a clean, professional format.
    """
    # Create logger
    logger = logging.getLogger("darkroom")
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

def get_logger(name=None):
    """
    Helper to get a sub-logger for a specific module.
    """
    if name:
        return logging.getLogger(f"darkroom.{name}")
    return logging.getLogger("darkroom")
