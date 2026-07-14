"""Controller lifecycle tests for LS-5000 unattended-run protection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from negpy.desktop import controller as controller_module
from negpy.desktop.controller import AppController
from negpy.desktop.workers.ls5000_roll_worker import (
    RollFrameChoice,
    RollOperation,
    RollPreviewRequest,
    RollScanCompletion,
    RollScanRequest,
)
from negpy.services.scanning.roll_preview_controls import ScanMaterial


class _Signal:
    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.error = error

    def emit(self, *args: object) -> None:
        self.calls.append(args)
        if self.error is not None:
            raise self.error


class _PowerAssertion:
    def __init__(self, *, release_error: Exception | None = None) -> None:
        self.release_calls = 0
        self.release_error = release_error

    def release(self) -> None:
        self.release_calls += 1
        if self.release_error is not None:
            raise self.release_error


class _ControllerHarness:
    _begin_ls5000_roll_power_assertion = AppController._begin_ls5000_roll_power_assertion
    _release_ls5000_roll_power_assertion = AppController._release_ls5000_roll_power_assertion
    start_ls5000_roll_preview = AppController.start_ls5000_roll_preview
    start_ls5000_roll_scan = AppController.start_ls5000_roll_scan
    cancel_scan = AppController.cancel_scan
    _on_ls5000_roll_preview_ready = AppController._on_ls5000_roll_preview_ready
    _on_ls5000_roll_finished = AppController._on_ls5000_roll_finished
    _on_ls5000_roll_error = AppController._on_ls5000_roll_error

    def __init__(self) -> None:
        self._ls5000_roll_power_assertion = None
        self.scan_started = _Signal()
        self.ls5000_roll_preview_requested = _Signal()
        self.ls5000_roll_scan_requested = _Signal()
        self.ls5000_roll_preview_ready = _Signal()
        self.ls5000_roll_finished = _Signal()
        self.ls5000_roll_error = _Signal()
        self.ls5000_roll_thumbnail_error = _Signal()
        self.scan_worker = SimpleNamespace(cancel_calls=0)
        self.scan_worker.cancel = lambda: setattr(
            self.scan_worker,
            "cancel_calls",
            self.scan_worker.cancel_calls + 1,
        )
        self.ls5000_roll_worker = SimpleNamespace(request_stop_calls=0)
        self.ls5000_roll_worker.request_stop = lambda: setattr(
            self.ls5000_roll_worker,
            "request_stop_calls",
            self.ls5000_roll_worker.request_stop_calls + 1,
        )
        self.imports: list[tuple[list[str], bool]] = []

    def import_negative_roll_scans(self, paths: list[str], *, black_and_white: bool) -> None:
        self.imports.append((paths, black_and_white))


def _preview_request() -> RollPreviewRequest:
    return RollPreviewRequest(
        attempts_root="/tmp/negpy-attempts",
        device_id="coolscan3:test",
        adapter_frame_capacity=40,
    )


def _scan_request() -> RollScanRequest:
    return RollScanRequest(
        device_id="coolscan3:test",
        adapter_frame_capacity=40,
        preview_token="preview-token",
        attempts_root="/tmp/negpy-attempts",
        output_folder="/tmp/negpy-output",
        filename_pattern='scan-{{ "%03d" % seq }}',
        material=ScanMaterial.COLOR_NEGATIVE,
        frames=(RollFrameChoice(2),),
    )


@pytest.fixture
def assertion_factory(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], _PowerAssertion]:
    reasons: list[str] = []
    assertion = _PowerAssertion()

    def acquire(reason: str) -> _PowerAssertion:
        reasons.append(reason)
        return assertion

    monkeypatch.setattr(controller_module, "acquire_unattended_power_assertion", acquire)
    return reasons, assertion


def test_preview_assertion_spans_request_until_preview_ready(
    assertion_factory: tuple[list[str], _PowerAssertion],
) -> None:
    reasons, assertion = assertion_factory
    controller = _ControllerHarness()
    request = _preview_request()
    session = object()

    controller.start_ls5000_roll_preview(request)

    assert reasons == ["NegPy LS-5000 roll preview"]
    assert assertion.release_calls == 0
    assert controller.scan_started.calls == [()]
    assert controller.ls5000_roll_preview_requested.calls == [(request,)]

    controller._on_ls5000_roll_preview_ready("preview-token", session)

    assert assertion.release_calls == 1
    assert controller._ls5000_roll_power_assertion is None
    assert controller.ls5000_roll_preview_ready.calls == [("preview-token", session)]


def test_full_scan_assertion_releases_before_completion_is_forwarded_and_imported(
    assertion_factory: tuple[list[str], _PowerAssertion],
) -> None:
    reasons, assertion = assertion_factory
    controller = _ControllerHarness()
    request = _scan_request()
    completion = RollScanCompletion(
        rgb_paths=("/tmp/frame001.tif",),
        black_and_white=False,
        stopped=False,
    )

    controller.start_ls5000_roll_scan(request)
    controller._on_ls5000_roll_finished(completion)

    assert reasons == ["NegPy LS-5000 full-roll scan"]
    assert assertion.release_calls == 1
    assert controller.ls5000_roll_finished.calls == [(completion,)]
    assert controller.imports == [(["/tmp/frame001.tif"], False)]


def test_stop_keeps_assertion_until_worker_reports_terminal_completion(
    assertion_factory: tuple[list[str], _PowerAssertion],
) -> None:
    _reasons, assertion = assertion_factory
    controller = _ControllerHarness()
    completion = RollScanCompletion(
        rgb_paths=(),
        black_and_white=False,
        stopped=True,
        operation=RollOperation.FULL_SCAN,
    )

    controller.start_ls5000_roll_scan(_scan_request())
    controller.cancel_scan()

    assert assertion.release_calls == 0
    assert controller.scan_worker.cancel_calls == 1
    assert controller.ls5000_roll_worker.request_stop_calls == 1

    controller._on_ls5000_roll_finished(completion)
    assert assertion.release_calls == 1


def test_worker_error_releases_assertion_before_forwarding(
    assertion_factory: tuple[list[str], _PowerAssertion],
) -> None:
    _reasons, assertion = assertion_factory
    controller = _ControllerHarness()
    failure = object()

    controller.start_ls5000_roll_preview(_preview_request())
    controller._on_ls5000_roll_error(failure)

    assert assertion.release_calls == 1
    assert controller.ls5000_roll_error.calls == [(failure,)]


def test_thumbnail_reload_error_does_not_release_active_roll_assertion(
    assertion_factory: tuple[list[str], _PowerAssertion],
) -> None:
    _reasons, assertion = assertion_factory
    controller = _ControllerHarness()
    failure = object()

    controller.start_ls5000_roll_scan(_scan_request())
    AppController._on_ls5000_roll_thumbnail_error(controller, failure)

    assert assertion.release_calls == 0
    assert controller._ls5000_roll_power_assertion is assertion
    assert controller.ls5000_roll_thumbnail_error.calls == [(failure,)]


@pytest.mark.parametrize("operation", ["preview", "scan"])
def test_start_signal_exception_releases_assertion(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    assertion = _PowerAssertion()
    monkeypatch.setattr(
        controller_module,
        "acquire_unattended_power_assertion",
        lambda _reason: assertion,
    )
    controller = _ControllerHarness()
    error = RuntimeError("signal failed")
    if operation == "preview":
        controller.ls5000_roll_preview_requested = _Signal(error)
        with pytest.raises(RuntimeError, match="signal failed"):
            controller.start_ls5000_roll_preview(_preview_request())
    else:
        controller.ls5000_roll_scan_requested = _Signal(error)
        with pytest.raises(RuntimeError, match="signal failed"):
            controller.start_ls5000_roll_scan(_scan_request())

    assert assertion.release_calls == 1
    assert controller._ls5000_roll_power_assertion is None


def test_release_failure_does_not_hide_terminal_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assertion = _PowerAssertion(release_error=RuntimeError("release failed"))
    monkeypatch.setattr(
        controller_module,
        "acquire_unattended_power_assertion",
        lambda _reason: assertion,
    )
    controller = _ControllerHarness()
    failure = object()

    controller.start_ls5000_roll_preview(_preview_request())
    controller._on_ls5000_roll_error(failure)

    assert assertion.release_calls == 1
    assert controller.ls5000_roll_error.calls == [(failure,)]
    assert controller._ls5000_roll_power_assertion is None


def test_second_roll_operation_is_rejected_without_releasing_first(
    assertion_factory: tuple[list[str], _PowerAssertion],
) -> None:
    reasons, assertion = assertion_factory
    controller = _ControllerHarness()

    controller.start_ls5000_roll_preview(_preview_request())
    with pytest.raises(RuntimeError, match="already owns"):
        controller.start_ls5000_roll_scan(_scan_request())

    assert reasons == ["NegPy LS-5000 roll preview"]
    assert assertion.release_calls == 0
