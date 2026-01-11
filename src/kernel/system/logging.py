import logging
import sys
import io


class _DummyStream(io.TextIOBase):
    """
    A file-like object that discards all input.
    Prevents AttributeError when sys.stdout/stderr are None in GUI apps.
    """

    def write(self, x: str) -> int:
        return len(x)

    def flush(self) -> None:
        pass


def init_streams() -> None:
    """
    Ensures sys.stdout and sys.stderr are not None.
    Required for Windows bundled applications without a console.
    """
    if sys.stdout is None:
        sys.stdout = _DummyStream()
    if sys.stderr is None:
        sys.stderr = _DummyStream()


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Sets up the global logging configuration for the application.
    """
    init_streams()

    # Create logger
    logger = logging.getLogger("negativepy")
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Create console handler using the now-guaranteed sys.stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Helper to get a sub-logger for a specific module.
    """
    if name:
        return logging.getLogger(f"negativepy.{name}")
    return logging.getLogger("negativepy")
