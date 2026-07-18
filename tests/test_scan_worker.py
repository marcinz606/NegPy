from __future__ import annotations

import threading

import pytest

from negpy.desktop.workers.scan_worker import BatchRequest, ScanRequest, ScanWorker
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


class _BatchService:
    """Records per-frame scans; returns a frame-numbered path from write_result."""

    def __init__(self, *, fail_on: int | None = None, cancel_before: int | None = None) -> None:
        self.fail_on = fail_on
        self.cancel_before = cancel_before
        self.frames: list[int] = []
        self.written_seqs: list[int] = []
        self.windows: list = []
        self.offsets: list[float] = []
        self.eject_calls: list[str] = []

    def eject(self, device_id: str) -> bool:
        self.eject_calls.append(device_id)
        return True

    def run_scan(self, device_id, params, progress, cancel):
        if self.cancel_before is not None and params.frame == self.cancel_before:
            cancel.set()
        self.frames.append(params.frame)
        self.windows.append(params.window)
        self.offsets.append(params.frame_offset_mm)
        if self.fail_on is not None and params.frame == self.fail_on:
            raise RuntimeError("frame failed")
        if progress:
            progress(1.0)
        return object()

    def write_result(self, *, result, output_folder, filename_pattern, output_format, seq) -> str:
        self.written_seqs.append(seq)
        return f"/tmp/frame_{seq:03d}.tif"


def _scan_request() -> ScanRequest:
    return ScanRequest(
        device_id="coolscan3:test",
        params=ScanParams(dpi=4_000, depth=16, capture_ir=False),
        output_folder="/tmp",
        filename_pattern='scan-{{ "%03d" % seq }}',
        output_format="TIFF",
    )


def _batch_request(frames=(2, 3, 4), frame_windows=None) -> BatchRequest:
    return BatchRequest(
        device_id="coolscan3:test",
        params=ScanParams(dpi=4_000, depth=16, capture_ir=False),
        output_folder="/tmp",
        filename_pattern='scan-{{ "%03d" % seq }}',
        output_format="TIFF",
        frames=tuple(frames),
        frame_windows=frame_windows or {},
    )


def _terminal_outcomes(worker: ScanWorker) -> tuple[list[str], list[None], list[str]]:
    finished: list[str] = []
    cancelled: list[None] = []
    errors: list[str] = []
    worker.finished.connect(finished.append)
    worker.cancelled.connect(lambda: cancelled.append(None))
    worker.error.connect(errors.append)
    return finished, cancelled, errors


def test_batch_applies_progressive_offset_per_frame_position() -> None:
    worker = ScanWorker()
    service = _BatchService()
    worker._service = service  # type: ignore[assignment]
    req = BatchRequest(
        device_id="coolscan3:test",
        params=ScanParams(dpi=4_000, depth=16, capture_ir=False, frame_offset_mm=1.0),
        output_folder="/tmp",
        filename_pattern='scan-{{ "%03d" % seq }}',
        output_format="TIFF",
        frames=(2, 3, 4),
        frame_offset_modifier_mm=0.2,
    )

    worker.run_batch(req)

    # Drift follows the physical frame position (N-1), not the enumeration order.
    assert service.offsets == pytest.approx([1.2, 1.4, 1.6])


def test_batch_negative_drift_floors_the_offset_at_zero() -> None:
    worker = ScanWorker()
    service = _BatchService()
    worker._service = service  # type: ignore[assignment]
    req = BatchRequest(
        device_id="coolscan3:test",
        params=ScanParams(dpi=4_000, depth=16, capture_ir=False, frame_offset_mm=0.3),
        output_folder="/tmp",
        filename_pattern='scan-{{ "%03d" % seq }}',
        output_format="TIFF",
        frames=(1, 2, 3),
        frame_offset_modifier_mm=-0.25,
    )

    worker.run_batch(req)

    assert service.offsets == pytest.approx([0.3, 0.05, 0.0])


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


def test_run_batch_scans_each_frame_and_reports_paths() -> None:
    worker = ScanWorker()
    service = _BatchService()
    worker._service = service  # type: ignore[assignment]
    frame_done: list[int] = []
    batch_finished: list[list[str]] = []
    worker.frame_done.connect(lambda frame, _path: frame_done.append(frame))
    worker.batch_finished.connect(batch_finished.append)

    worker.run_batch(_batch_request((2, 3, 4)))

    assert service.frames == [2, 3, 4]
    assert service.written_seqs == [2, 3, 4]  # frame-numbered, not 1,2,3
    assert frame_done == [2, 3, 4]
    assert batch_finished == [["/tmp/frame_002.tif", "/tmp/frame_003.tif", "/tmp/frame_004.tif"]]
    assert worker._scanning is False


def test_run_batch_stop_between_frames_keeps_completed_frames() -> None:
    worker = ScanWorker()
    service = _BatchService(cancel_before=3)
    worker._service = service  # type: ignore[assignment]
    frame_done: list[int] = []
    batch_finished: list[list[str]] = []
    cancelled: list[None] = []
    worker.frame_done.connect(lambda frame, _path: frame_done.append(frame))
    worker.batch_finished.connect(batch_finished.append)
    worker.cancelled.connect(lambda: cancelled.append(None))

    worker.run_batch(_batch_request((2, 3, 4)))

    assert service.written_seqs == [2]
    assert frame_done == [2]
    assert batch_finished == [["/tmp/frame_002.tif"]]
    assert cancelled == [None]


def test_run_batch_error_reports_error_and_keeps_prior_frames() -> None:
    worker = ScanWorker()
    service = _BatchService(fail_on=3)
    worker._service = service  # type: ignore[assignment]
    frame_done: list[int] = []
    batch_finished: list[list[str]] = []
    errors: list[str] = []
    worker.frame_done.connect(lambda frame, _path: frame_done.append(frame))
    worker.batch_finished.connect(batch_finished.append)
    worker.error.connect(errors.append)

    worker.run_batch(_batch_request((2, 3, 4)))

    assert frame_done == [2]
    assert batch_finished == [["/tmp/frame_002.tif"]]
    assert errors == ["frame failed"]


def test_run_batch_auto_returns_film_after_a_clean_batch() -> None:
    worker = ScanWorker()
    service = _BatchService()
    worker._service = service  # type: ignore[assignment]

    worker.run_batch(_batch_request((2, 3, 4)))

    assert service.eject_calls == ["coolscan3:test"]


def test_run_batch_does_not_eject_when_stopped() -> None:
    worker = ScanWorker()
    service = _BatchService(cancel_before=3)
    worker._service = service  # type: ignore[assignment]

    worker.run_batch(_batch_request((2, 3, 4)))

    assert service.eject_calls == []


def test_run_batch_does_not_eject_on_error() -> None:
    worker = ScanWorker()
    service = _BatchService(fail_on=3)
    worker._service = service  # type: ignore[assignment]

    worker.run_batch(_batch_request((2, 3, 4)))

    assert service.eject_calls == []


def test_run_batch_scans_an_explicit_frame_subset() -> None:
    worker = ScanWorker()
    service = _BatchService()
    worker._service = service  # type: ignore[assignment]

    worker.run_batch(_batch_request((1, 2, 4, 6)))

    assert service.frames == [1, 2, 4, 6]
    assert service.written_seqs == [1, 2, 4, 6]


def test_run_batch_applies_per_frame_windows_falling_back_to_base() -> None:
    worker = ScanWorker()
    service = _BatchService()
    worker._service = service  # type: ignore[assignment]
    w2 = (0.1, 0.1, 0.5, 0.5)
    w4 = (0.2, 0.2, 0.6, 0.6)

    worker.run_batch(_batch_request((2, 3, 4), frame_windows={2: w2, 4: w4}))

    assert service.windows == [w2, None, w4]  # frame 3 has no window → base (None)


class _Result:
    def __init__(self, rgb) -> None:
        self.rgb = rgb


class _PreviewService:
    def __init__(self, *, rgb=None, acquisition_error: Exception | None = None, cancel_during_acquisition: bool = False) -> None:
        self._rgb = rgb if rgb is not None else object()
        self.acquisition_error = acquisition_error
        self.cancel_during_acquisition = cancel_during_acquisition
        self.run_calls = 0
        self.write_calls = 0

    def run_scan(self, *, device_id, params, progress, cancel):
        self.run_calls += 1
        if self.cancel_during_acquisition:
            cancel.set()
        if self.acquisition_error is not None:
            raise self.acquisition_error
        return _Result(self._rgb)

    def write_result(self, **_kwargs) -> str:
        self.write_calls += 1
        return "/tmp/x.tif"


def test_run_preview_emits_rgb_and_skips_write() -> None:
    worker = ScanWorker()
    rgb = object()
    service = _PreviewService(rgb=rgb)
    worker._service = service  # type: ignore[assignment]
    previews: list[object] = []
    finished, cancelled, errors = _terminal_outcomes(worker)
    worker.preview_ready.connect(previews.append)

    worker.run_preview(_scan_request())

    assert previews == [rgb]
    assert service.write_calls == 0
    assert finished == [] and cancelled == [] and errors == []
    assert worker._scanning is False


def test_run_preview_cancelled_emits_cancelled_not_preview() -> None:
    worker = ScanWorker()
    service = _PreviewService(cancel_during_acquisition=True)
    worker._service = service  # type: ignore[assignment]
    previews: list[object] = []
    finished, cancelled, errors = _terminal_outcomes(worker)
    worker.preview_ready.connect(previews.append)

    worker.run_preview(_scan_request())

    assert previews == []
    assert cancelled == [None]
    assert errors == []


def test_run_preview_error_emits_error() -> None:
    worker = ScanWorker()
    service = _PreviewService(acquisition_error=RuntimeError("io error"))
    worker._service = service  # type: ignore[assignment]
    previews: list[object] = []
    finished, cancelled, errors = _terminal_outcomes(worker)
    worker.preview_ready.connect(previews.append)

    worker.run_preview(_scan_request())

    assert previews == []
    assert errors == ["io error"]
    assert cancelled == []
