"""OS-boundary tests for unattended desktop power assertions."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from negpy.desktop.power_assertion import acquire_unattended_power_assertion


class _ProcessInfo:
    def __init__(self) -> None:
        self.begin_calls: list[tuple[int, str]] = []
        self.end_calls: list[object] = []
        self.token = object()

    def beginActivityWithOptions_reason_(self, options: int, reason: str) -> object:
        self.begin_calls.append((options, reason))
        return self.token

    def endActivity_(self, token: object) -> None:
        self.end_calls.append(token)


class _CaffeinateProcess:
    def __init__(self, *, wait_times_out: bool = False) -> None:
        self.returncode: int | None = None
        self.wait_times_out = wait_times_out
        self.poll_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        self.poll_calls += 1
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.wait_times_out and self.kill_calls == 0:
            assert timeout is not None
            raise subprocess.TimeoutExpired("caffeinate", timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


class _Kernel32:
    def __init__(self, *, result: int = 0x80000000, clear_fails: bool = False) -> None:
        self.calls: list[int] = []
        self.result = result
        self.clear_fails = clear_fails

    def SetThreadExecutionState(self, flags: int) -> int:  # noqa: N802
        self.calls.append(flags)
        if self.clear_fails and flags == 0x80000000:
            raise OSError("clear failed")
        return self.result


def _foundation(process_info: _ProcessInfo) -> Any:
    return SimpleNamespace(
        NSActivityUserInitiated=0x01,
        NSActivityIdleSystemSleepDisabled=0x04,
        NSProcessInfo=SimpleNamespace(processInfo=lambda: process_info),
    )


def test_unsupported_platform_assertion_is_an_idempotent_noop() -> None:
    def fail_loader() -> Any:
        raise AssertionError("an unsupported platform must not load a backend")

    assertion = acquire_unattended_power_assertion(
        "test scan",
        _platform="linux",
        _foundation_loader=fail_loader,
        _kernel32_loader=fail_loader,
        _process_factory=lambda *_args, **_kwargs: pytest.fail("an unsupported platform must not run a helper"),
    )

    assertion.release()
    assertion.release()


def test_windows_blocks_idle_system_sleep_and_clears_it_once() -> None:
    kernel32 = _Kernel32()

    assertion = acquire_unattended_power_assertion(
        "NegPy scan batch",
        _platform="win32",
        _foundation_loader=lambda: pytest.fail("Windows must not load Foundation"),
        _kernel32_loader=lambda: kernel32,
        _process_factory=lambda *_args, **_kwargs: pytest.fail("Windows must not run caffeinate"),
    )

    assert kernel32.calls == [0x80000001]
    assertion.release()
    assertion.release()
    assert kernel32.calls == [0x80000001, 0x80000000]


def test_windows_refused_request_degrades_to_noop() -> None:
    kernel32 = _Kernel32(result=0)

    assertion = acquire_unattended_power_assertion("NegPy scan batch", _platform="win32", _kernel32_loader=lambda: kernel32)

    assertion.release()
    assert kernel32.calls == [0x80000001]


def test_windows_backend_failures_degrade_to_noop() -> None:
    assertion = acquire_unattended_power_assertion(
        "NegPy scan batch",
        _platform="win32",
        _kernel32_loader=lambda: (_ for _ in ()).throw(OSError("no kernel32")),
    )

    assertion.release()


def test_windows_release_failure_is_swallowed() -> None:
    kernel32 = _Kernel32(clear_fails=True)
    assertion = acquire_unattended_power_assertion("NegPy scan batch", _platform="win32", _kernel32_loader=lambda: kernel32)

    assertion.release()
    assertion.release()

    assert kernel32.calls == [0x80000001, 0x80000000]


def test_macos_prefers_nsprocessinfo_and_releases_once() -> None:
    process_info = _ProcessInfo()

    assertion = acquire_unattended_power_assertion(
        "NegPy scan batch",
        _platform="darwin",
        _foundation_loader=lambda: _foundation(process_info),
        _process_factory=lambda *_args, **_kwargs: pytest.fail("Foundation success must not run caffeinate"),
    )

    assert process_info.begin_calls == [(0x05, "NegPy scan batch")]
    assertion.release()
    assertion.release()
    assert process_info.end_calls == [process_info.token]


def test_macos_falls_back_to_parent_bound_caffeinate() -> None:
    process = _CaffeinateProcess()
    launches: list[tuple[list[str], dict[str, object]]] = []

    def spawn(argv: list[str], **kwargs: object) -> _CaffeinateProcess:
        launches.append((argv, kwargs))
        return process

    assertion = acquire_unattended_power_assertion(
        "NegPy scan batch",
        _platform="darwin",
        _foundation_loader=lambda: (_ for _ in ()).throw(ImportError("no PyObjC")),
        _process_factory=spawn,
        _parent_pid=4321,
    )

    assert launches == [
        (
            ["/usr/bin/caffeinate", "-i", "-w", "4321"],
            {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": True,
                "start_new_session": True,
            },
        )
    ]
    assertion.release()
    assertion.release()
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_calls == [2.0]


def test_caffeinate_release_kills_a_helper_that_ignores_terminate() -> None:
    process = _CaffeinateProcess(wait_times_out=True)
    assertion = acquire_unattended_power_assertion(
        "NegPy scan batch",
        _platform="darwin",
        _foundation_loader=lambda: (_ for _ in ()).throw(ImportError("no PyObjC")),
        _process_factory=lambda *_args, **_kwargs: process,
        _parent_pid=4321,
    )

    assertion.release()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == [2.0, 2.0]


def test_macos_backend_failures_degrade_to_noop() -> None:
    assertion = acquire_unattended_power_assertion(
        "NegPy scan batch",
        _platform="darwin",
        _foundation_loader=lambda: (_ for _ in ()).throw(ImportError("no PyObjC")),
        _process_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("caffeinate unavailable")),
    )

    assertion.release()


def test_power_assertion_rejects_an_empty_reason() -> None:
    with pytest.raises(ValueError, match="reason must not be empty"):
        acquire_unattended_power_assertion("  ", _platform="linux")
