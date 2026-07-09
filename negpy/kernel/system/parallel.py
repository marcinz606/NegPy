"""
Runtime gate for Numba parallel (multi-core) CPU kernels.

The app pins NUMBA_THREADING_LAYER=workqueue (desktop/main.py), and workqueue
is not thread-safe under concurrent access: if two threads invoke
parallel=True kernels at the same time, Numba hard-terminates the process.
The render worker, export worker and the CPU display path (ICC LUT) can all
overlap, which surfaced as crashes during rendering/export (reported on macOS
first, but the race is cross-platform). Two safeguards:

- every parallel kernel is compiled in BOTH serial and parallel variants via
  `parallel_njit` and dispatched per call, so parallelism can be switched off
  at runtime without recompiling or restarting — the serial variant is the
  plain @njit path that has always been stable;
- parallel invocations are serialized behind a process-wide lock, so two
  threads can never be inside the workqueue scheduler at once. The kernels
  are themselves multi-core, so cross-thread serialization costs ~nothing.

Default policy: parallel on, except macOS until the lock fix is verified on
real hardware there. Override with `cpu_parallel = true/false` under
[performance] in override.toml.
"""

import sys
import threading
from typing import Any, Callable, Optional

from numba import njit

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# workqueue threading layer: concurrent parallel-kernel entry aborts the process.
_invocation_gate = threading.Lock()


def default_cpu_parallel(platform: str = sys.platform) -> bool:
    """Platform policy: parallel kernels on, except macOS (pending verification)."""
    return platform != "darwin"


_parallel_enabled: bool = default_cpu_parallel()


def parallel_enabled() -> bool:
    return _parallel_enabled


def set_parallel_enabled(enabled: bool) -> None:
    global _parallel_enabled
    _parallel_enabled = bool(enabled)


def configure_cpu_parallel(override: Optional[bool]) -> None:
    """Apply the startup policy: explicit override wins, else the platform default."""
    set_parallel_enabled(default_cpu_parallel() if override is None else override)
    logger.info(
        "CPU parallel kernels: %s (override=%s, platform=%s)",
        "enabled" if _parallel_enabled else "disabled",
        override,
        sys.platform,
    )


def parallel_njit(**jit_kwargs: Any) -> Callable:
    """
    @njit that compiles serial AND parallel variants and picks one per call.

    The kernel body may use `prange`; with parallel=False it degrades to plain
    `range`, so both variants compile from the same source. Each variant JITs
    lazily on first use, so the unused one costs nothing. Parallel calls are
    serialized behind the module lock (see module docstring).

    The serial variant is always compiled with cache=False: numba's disk cache
    is keyed by the function's source location, so both variants of the same
    function share one cache slot — whichever compiles first, the other loads
    its binary (verified: a "serial" call can silently execute the cached
    parallel object, defeating the failsafe). Only the parallel variant may
    honour a caller-supplied cache=True; the serial path re-JITs once per
    process on first use.
    """
    jit_kwargs.pop("parallel", None)

    def wrap(py_func: Callable) -> "_DualDispatcher":
        return _DualDispatcher(py_func, jit_kwargs)

    return wrap


class _DualDispatcher:
    """Callable pairing the serial and parallel compilations of one kernel."""

    def __init__(self, py_func: Callable, jit_kwargs: dict):
        self.serial = njit(**{**jit_kwargs, "cache": False}, parallel=False)(py_func)
        self.parallel = njit(**jit_kwargs, parallel=True)(py_func)
        self.__wrapped__ = py_func
        self.__name__ = getattr(py_func, "__name__", "kernel")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _parallel_enabled:
            with _invocation_gate:
                return self.parallel(*args, **kwargs)
        return self.serial(*args, **kwargs)
