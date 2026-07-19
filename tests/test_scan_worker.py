from __future__ import annotations

import threading

import pytest

from negpy.desktop.workers.scan_worker import BatchRequest, RollPreviewRequest, ScanRequest, ScanWorker
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice, ScannerUnavailable
from negpy.infrastructure.scanners.params import ScanMode, ScanParams
from negpy.infrastructure.scanners.roll import RollPreview


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


def test_list_devices_failure_emits_the_empty_list_before_the_error() -> None:
    """Order matters: the sidebar's devices_ready handler rewrites the status label,
    so an error emitted first would lose the backend's install hint."""

    class _FailingService:
        def refresh_devices(self):
            raise ScannerUnavailable("python-sane not importable. pip install python-sane")

    worker = ScanWorker()
    worker._service = _FailingService()  # type: ignore[assignment]
    order: list[str] = []
    worker.devices_ready.connect(lambda devices: order.append(f"devices:{len(devices)}"))
    worker.error.connect(lambda msg: order.append(f"error:{msg}"))

    worker.list_devices()

    assert order == ["devices:0", "error:python-sane not importable. pip install python-sane"]


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


class _FakeRollSession:
    def __init__(self, *, slots_seen: list, open_error: Exception | None = None, cancel_at: int | None = None) -> None:
        self.slots_seen = slots_seen
        self.offsets: dict[int, float] = {}
        self.closed = False
        self._cancel_at = cancel_at
        self._open_error = open_error

    def set_offset(self, slot: int, offset: float) -> None:
        self.offsets[slot] = offset

    def preview(self, slots, *, cancel):
        for slot in slots:
            if self._cancel_at == slot:
                cancel.set()
                return
            self.slots_seen.append(slot)
            yield RollPreview(slot=slot, rgb=object(), offset=self.offsets.get(slot, 0.0))

    def close(self) -> None:
        self.closed = True


class _RollService:
    def __init__(self, *, session: _FakeRollSession | None = None, open_error: Exception | None = None) -> None:
        self.session = session or _FakeRollSession(slots_seen=[])
        self._open_error = open_error

    def open_roll(self, device, *, dpi):
        if self._open_error is not None:
            raise self._open_error
        self.dpi = dpi
        return self.session


def _roll_request(slots=(1, 2, 3), offsets=None) -> RollPreviewRequest:
    caps = ScannerCapabilities(
        ir_channel=False,
        supported_dpi=(1000,),
        supported_depths=(8,),
        sources=(ScanMode.NEGATIVE,),
        max_area_mm=(25.0, 38.0),
        adapter_frame_capacity=len(slots),
    )
    device = ScannerDevice(id="coolscan3:test", vendor="Nikon", model="LS-50", capabilities=caps)
    return RollPreviewRequest(device=device, slots=tuple(slots), dpi=1000, offsets=offsets or {})


def test_run_roll_preview_emits_one_result_per_slot_then_finishes() -> None:
    worker = ScanWorker()
    service = _RollService()
    worker._service = service  # type: ignore[assignment]
    previews: list = []
    done: list = []
    _finished, cancelled, errors = _terminal_outcomes(worker)
    worker.roll_preview_ready.connect(previews.append)
    worker.roll_preview_finished.connect(lambda: done.append(True))

    worker.run_roll_preview(_roll_request())

    assert [p.slot for p in previews] == [1, 2, 3]
    assert done == [True]
    assert cancelled == [] and errors == []
    assert service.session.closed is True
    assert worker._scanning is False


def test_run_roll_preview_applies_offsets_before_previewing() -> None:
    worker = ScanWorker()
    service = _RollService()
    worker._service = service  # type: ignore[assignment]

    worker.run_roll_preview(_roll_request(offsets={2: 0.25}))

    assert service.session.offsets == {2: 0.25}


def test_run_roll_preview_cancel_stops_the_strip() -> None:
    worker = ScanWorker()
    session = _FakeRollSession(slots_seen=[], cancel_at=3)
    service = _RollService(session=session)
    worker._service = service  # type: ignore[assignment]
    done: list = []
    _finished, cancelled, errors = _terminal_outcomes(worker)
    worker.roll_preview_finished.connect(lambda: done.append(True))

    worker.run_roll_preview(_roll_request())

    assert session.slots_seen == [1, 2]
    assert cancelled == [None]
    assert done == [] and errors == []


def test_run_roll_preview_reports_a_failure_to_open_the_strip() -> None:
    worker = ScanWorker()
    worker._service = _RollService(open_error=RuntimeError("no film"))  # type: ignore[assignment]
    done: list = []
    _finished, cancelled, errors = _terminal_outcomes(worker)
    worker.roll_preview_finished.connect(lambda: done.append(True))

    worker.run_roll_preview(_roll_request())

    assert errors == ["no film"]
    assert done == [] and cancelled == []


def test_run_roll_preview_closes_the_session_when_a_slot_raises() -> None:
    class _Exploding(_FakeRollSession):
        def preview(self, slots, *, cancel):
            yield RollPreview(slot=1, rgb=object())
            raise RuntimeError("transport died")

    session = _Exploding(slots_seen=[])
    worker = ScanWorker()
    worker._service = _RollService(session=session)  # type: ignore[assignment]
    _finished, _cancelled, errors = _terminal_outcomes(worker)

    worker.run_roll_preview(_roll_request())

    assert errors == ["transport died"]
    assert session.closed is True
