from __future__ import annotations

import threading

from negpy.desktop.workers.scan_worker import ScanRequest, ScanWorker
from negpy.infrastructure.scanners.params import ScanParams


class _EjectService:
    def __init__(self, result: bool = True, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def eject(self, device_id: str) -> bool:
        self.calls.append(device_id)
        if self.error is not None:
            raise self.error
        return self.result


class _ScanService:
    def __init__(
        self,
        *,
        cancel_during_acquisition: bool = False,
        acquisition_error: Exception | None = None,
        cancel_during_write: bool = False,
        write_error: Exception | None = None,
    ) -> None:
        self.cancel_during_acquisition = cancel_during_acquisition
        self.acquisition_error = acquisition_error
        self.cancel_during_write = cancel_during_write
        self.write_error = write_error
        self.cancel_at_acquisition_start: bool | None = None
        self.cancel_event: threading.Event | None = None
        self.run_calls = 0
        self.write_calls = 0

    def run_scan(self, *, device_id, params, progress, cancel):
        self.run_calls += 1
        self.cancel_event = cancel
        self.cancel_at_acquisition_start = cancel.is_set()
        if self.cancel_during_acquisition:
            cancel.set()
        if self.acquisition_error is not None:
            raise self.acquisition_error
        return object()

    def write_result(self, **_kwargs) -> str:
        self.write_calls += 1
        if self.cancel_during_write:
            assert self.cancel_event is not None
            self.cancel_event.set()
        if self.write_error is not None:
            raise self.write_error
        return "/tmp/scan.tif"


def _scan_request() -> ScanRequest:
    return ScanRequest(
        device_id="coolscan3:test",
        params=ScanParams(dpi=4_000, depth=16, capture_ir=False),
        output_folder="/tmp",
        filename_pattern='scan-{{ "%03d" % seq }}',
        output_format="TIFF",
    )


def _terminal_outcomes(worker: ScanWorker) -> tuple[list[str], list[None], list[str]]:
    finished: list[str] = []
    cancelled: list[None] = []
    errors: list[str] = []
    worker.finished.connect(finished.append)
    worker.cancelled.connect(lambda: cancelled.append(None))
    worker.error.connect(errors.append)
    return finished, cancelled, errors


def test_scan_worker_emits_eject_result() -> None:
    worker = ScanWorker()
    service = _EjectService()
    worker._service = service  # type: ignore[assignment]
    results: list[bool] = []
    worker.ejected.connect(results.append)

    worker.eject("coolscan3:test")

    assert service.calls == ["coolscan3:test"]
    assert results == [True]


def test_scan_worker_reports_eject_failure() -> None:
    worker = ScanWorker()
    service = _EjectService(error=RuntimeError("transport refused"))
    worker._service = service  # type: ignore[assignment]
    errors: list[str] = []
    worker.eject_error.connect(errors.append)

    worker.eject("coolscan3:test")

    assert service.calls == ["coolscan3:test"]
    assert errors == ["transport refused"]


def test_scan_worker_emits_cancelled_when_acquisition_returns_after_cancel() -> None:
    worker = ScanWorker()
    service = _ScanService(cancel_during_acquisition=True)
    worker._service = service  # type: ignore[assignment]
    finished, cancelled, errors = _terminal_outcomes(worker)

    worker.run_scan(_scan_request())

    assert finished == []
    assert cancelled == [None]
    assert errors == []
    assert service.write_calls == 0
    assert worker._scanning is False


def test_scan_worker_emits_cancelled_when_acquisition_raises_after_cancel() -> None:
    worker = ScanWorker()
    service = _ScanService(
        cancel_during_acquisition=True,
        acquisition_error=RuntimeError("Scan cancelled"),
    )
    worker._service = service  # type: ignore[assignment]
    finished, cancelled, errors = _terminal_outcomes(worker)

    worker.run_scan(_scan_request())

    assert finished == []
    assert cancelled == [None]
    assert errors == []
    assert service.write_calls == 0
    assert worker._scanning is False


def test_prepare_scan_preserves_cancel_pressed_before_queued_run_starts() -> None:
    worker = ScanWorker()
    service = _ScanService()
    worker._service = service  # type: ignore[assignment]
    finished, cancelled, errors = _terminal_outcomes(worker)
    ensure_calls = 0
    original_ensure_service = worker._ensure_service

    def ensure_service():
        nonlocal ensure_calls
        ensure_calls += 1
        return original_ensure_service()

    worker._ensure_service = ensure_service  # type: ignore[method-assign]

    worker.prepare_scan()
    worker.cancel()
    worker.run_scan(_scan_request())

    assert ensure_calls == 0
    assert service.run_calls == 0
    assert service.cancel_at_acquisition_start is None
    assert finished == []
    assert cancelled == [None]
    assert errors == []


def test_cancel_during_write_does_not_hide_write_failure() -> None:
    worker = ScanWorker()
    service = _ScanService(
        cancel_during_write=True,
        write_error=OSError("scratch disk full"),
    )
    worker._service = service  # type: ignore[assignment]
    finished, cancelled, errors = _terminal_outcomes(worker)

    worker.run_scan(_scan_request())

    assert service.write_calls == 1
    assert finished == []
    assert cancelled == []
    assert errors == ["scratch disk full"]
    assert worker._scanning is False
