import logging
import os
import sys
import io


class _DummyStream(io.TextIOBase):
    """
    DevNull replacement for GUI apps.
    """

    def write(self, x: str) -> int:
        return len(x)

    def flush(self) -> None:
        pass


def init_streams() -> None:
    """
    Fixes None stdout/stderr in Windows GUI bundles.
    """
    if sys.stdout is None:
        sys.stdout = _DummyStream()
    if sys.stderr is None:
        sys.stderr = _DummyStream()


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Global log config.
    """
    init_streams()

    # Create logger
    logger = logging.getLogger("negpy")
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

    # File handler too: the console log is invisible in a packaged .app (no terminal), so also write
    # to a rotating file under the user data dir. Users can attach it to bug reports, and the global
    # exception hook (desktop/main.py) records unhandled slot exceptions here that would otherwise
    # only abort with a native crash report that hides the Python traceback. Best-effort — a
    # file-logging failure (read-only home, etc.) must never stop the app from starting.
    try:
        from logging.handlers import RotatingFileHandler

        from negpy.kernel.system.paths import get_default_user_dir

        log_dir = get_default_user_dir()
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(os.path.join(log_dir, "negpy.log"), maxBytes=2_000_000, backupCount=2, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        logger.warning("could not set up file logging", exc_info=True)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Helper to get a sub-logger for a specific module.
    """
    if name:
        return logging.getLogger(f"negpy.{name}")
    return logging.getLogger("negpy")
