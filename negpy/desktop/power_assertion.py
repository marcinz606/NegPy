"""Best-effort idle-sleep protection for unattended desktop operations.

The scanner process remains responsible for its own USB lifecycle.  This module
only asks the operating system to keep the computer awake while a user-requested
operation is running.  macOS and Windows receive a real assertion; other platforms
receive an inert one.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any, Protocol

from negpy.kernel.system.logging import get_logger


logger = get_logger(__name__)

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class UnattendedPowerAssertion(Protocol):
    """One idempotently releasable operating-system power assertion."""

    def release(self) -> None: ...


class _CaffeinateProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class _NoOpPowerAssertion:
    def release(self) -> None:
        return


class _NSProcessInfoPowerAssertion:
    def __init__(self, process_info: Any, token: object) -> None:
        self._process_info = process_info
        self._token: object | None = token

    def release(self) -> None:
        token = self._token
        if token is None:
            return
        self._token = None
        try:
            self._process_info.endActivity_(token)
        except Exception:
            # Releasing an OS hint must never hide a scanner result or stop the controller
            # forwarding its terminal signal.
            logger.warning("could not end the macOS NSProcessInfo activity", exc_info=True)


class _WindowsPowerAssertion:
    """Windows keeps the execution state per thread: release on the thread that acquired it."""

    def __init__(self, kernel32: Any) -> None:
        self._kernel32: Any | None = kernel32

    def release(self) -> None:
        kernel32 = self._kernel32
        if kernel32 is None:
            return
        self._kernel32 = None
        try:
            kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        except Exception:
            logger.warning("could not clear the Windows execution state", exc_info=True)


class _CaffeinatePowerAssertion:
    _WAIT_SECONDS = 2.0

    def __init__(self, process: _CaffeinateProcess) -> None:
        self._process: _CaffeinateProcess | None = process

    def release(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        try:
            if process.poll() is not None:
                return
            process.terminate()
            try:
                process.wait(timeout=self._WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self._WAIT_SECONDS)
        except ProcessLookupError:
            return
        except Exception:
            logger.warning("could not stop the macOS caffeinate helper", exc_info=True)


def _load_foundation() -> Any:
    """Load PyObjC lazily so it remains an optional packaged dependency."""

    return importlib.import_module("Foundation")


def _load_kernel32() -> Any:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
    return kernel32


def _acquire_windows(kernel32_loader: Callable[[], Any] | None) -> UnattendedPowerAssertion:
    """Block idle system sleep for the calling thread; the display may still turn off."""

    try:
        kernel32 = (_load_kernel32 if kernel32_loader is None else kernel32_loader)()
        if not kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED):
            raise OSError("SetThreadExecutionState refused the request")
        return _WindowsPowerAssertion(kernel32)
    except Exception:
        logger.warning("could not create a Windows idle-sleep assertion", exc_info=True)
        return _NoOpPowerAssertion()


def acquire_unattended_power_assertion(
    reason: str,
    *,
    _platform: str | None = None,
    _foundation_loader: Callable[[], Any] | None = None,
    _kernel32_loader: Callable[[], Any] | None = None,
    _process_factory: Callable[..., _CaffeinateProcess] | None = None,
    _parent_pid: int | None = None,
) -> UnattendedPowerAssertion:
    """Prevent idle sleep until the returned assertion is released.

    On macOS, ``NSProcessInfo`` is preferred because its activity is owned by
    this process.  PyObjC is optional, so packaged builds fall back to
    ``/usr/bin/caffeinate -i -w <parent pid>``.  The ``-w`` binding makes the
    helper release its assertion even if NegPy exits without normal cleanup.
    On Windows the assertion belongs to the calling thread and ends with it.
    Backend failures are logged and degrade to a no-op instead of blocking a
    scan.
    """

    reason = str(reason).strip()
    if not reason:
        raise ValueError("power assertion reason must not be empty")
    platform = sys.platform if _platform is None else _platform
    if platform == "win32":
        return _acquire_windows(_kernel32_loader)
    if platform != "darwin":
        return _NoOpPowerAssertion()

    foundation_loader = _load_foundation if _foundation_loader is None else _foundation_loader
    try:
        foundation = foundation_loader()
        process_info = foundation.NSProcessInfo.processInfo()
        options = int(foundation.NSActivityUserInitiated) | int(foundation.NSActivityIdleSystemSleepDisabled)
        token = process_info.beginActivityWithOptions_reason_(options, reason)
        if token is None:
            raise RuntimeError("NSProcessInfo returned no activity token")
        return _NSProcessInfoPowerAssertion(process_info, token)
    except Exception:
        # Missing PyObjC is normal in a minimal wheel. Keep this at debug level, because the
        # supported caffeinate fallback follows immediately.
        logger.debug("macOS NSProcessInfo activity is unavailable", exc_info=True)

    process_factory = subprocess.Popen if _process_factory is None else _process_factory
    parent_pid = os.getpid() if _parent_pid is None else _parent_pid
    try:
        process = process_factory(
            ["/usr/bin/caffeinate", "-i", "-w", str(parent_pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"caffeinate exited immediately with status {returncode}")
        return _CaffeinatePowerAssertion(process)
    except Exception:
        logger.warning("could not create a macOS idle-sleep assertion", exc_info=True)
        return _NoOpPowerAssertion()


__all__ = ["UnattendedPowerAssertion", "acquire_unattended_power_assertion"]
