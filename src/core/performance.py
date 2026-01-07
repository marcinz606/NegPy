import time
import functools
import os
import csv
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec
from src.logging_config import get_logger
from src.config import APP_CONFIG

logger = get_logger("perf")

P = ParamSpec("P")
R = TypeVar("R")


def get_perf_log_path() -> str:
    return os.path.join(APP_CONFIG.cache_dir, "perf_stats.csv")


def init_perf_log() -> None:
    log_path = get_perf_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "function", "duration_ms", "image_shape"])


def clear_perf_log() -> None:
    log_path = get_perf_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "function", "duration_ms", "image_shape"])


def log_to_csv(function_name: str, duration_ms: float, shape: Any) -> None:
    try:
        init_perf_log()
        log_path = get_perf_log_path()
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    function_name,
                    f"{duration_ms:.3f}",
                    str(shape),
                ]
            )
    except Exception as e:
        logger.error(f"Failed to log perf stats: {e}")


def time_function(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration_ms = (end - start) * 1000

        # Try to find image shape for context
        shape = "N/A"
        # Check all args for something with a .shape
        for arg in args:
            if hasattr(arg, "shape"):
                shape = getattr(arg, "shape")
                break
        if shape == "N/A":
            for val in kwargs.values():
                if hasattr(val, "shape"):
                    shape = getattr(val, "shape")
                    break

        logger.info(f"PERF: {func.__name__} took {duration_ms:.3f}ms (shape: {shape})")
        log_to_csv(func.__name__, duration_ms, shape)
        return result

    return wrapper
