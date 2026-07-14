"""Offline orchestration tests for the LS-5000 roll worker.

The worker sits directly above the process-isolated USB capture helper and the
SANE backend.  These tests replace both boundaries with fakes, so collecting
them can never enumerate or move a real scanner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from negpy.desktop.workers import ls5000_roll_worker as roll_worker_module
from negpy.desktop.workers.ls5000_roll_worker import (
    LS5000RollWorker,
    RollFrameChoice,
    RollOperation,
    RollPreviewRequest,
    RollScanCompletion,
    RollScanRequest,
)
from negpy.infrastructure.scanners.ls5000_single_pass.capture_process import (
    AttemptPaths,
    BatchAckAction,
    BatchSessionPaths,
    CaptureAttemptResult,
    CaptureBatchProcessError,
    CaptureBatchRequest,
    CaptureBatchResult,
    CaptureMode,
    CaptureOutcome,
    CaptureRequest,
    CaptureStopped,
)
from negpy.infrastructure.scanners.params import RegisteredScanGeometry
from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.ls5000_roll_outputs import PromotedFrame
from negpy.services.scanning.ls5000_sane_rgb import (
    LS5000_FRAME_PITCH_UNITS,
    LS5000_FULL_WINDOW_ROWS,
)
from negpy.services.scanning.roll_preview_controls import ScanMaterial


@dataclass(frozen=True)
class _Slot:
    slot_id: int


class _PreviewSession:
    """Small structural fake for RollPreviewSession.

    ``resolve_origin`` returns native integer positions, which exercises the
    real SANE-geometry conversion without constructing scanner artifacts.
    """

    def __init__(
        self,
        *,
        slot_count: int = 8,
        origins: dict[tuple[int, int], int] | None = None,
    ) -> None:
        self.preview = object()
        self.slots = tuple(_Slot(slot_id) for slot_id in range(1, slot_count + 1))
        self.origins = origins or {}
        self.resolve_calls: list[tuple[int, int]] = []

    def resolve_origin(self, slot_id: int, boundary_offset_rows: int = 0) -> int:
        self.resolve_calls.append((slot_id, boundary_offset_rows))
        key = (slot_id, boundary_offset_rows)
        if key in self.origins:
            return self.origins[key]
        return (slot_id - 1) * LS5000_FRAME_PITCH_UNITS + boundary_offset_rows


class _Adapter:
    def __init__(
        self,
        result_for: Callable[[CaptureRequest], CaptureAttemptResult],
        attempts_root: Path,
    ) -> None:
        self._result_for = result_for
        self._attempts_root = attempts_root
        self.requests: list[CaptureRequest] = []
        self.batch_requests: list[CaptureBatchRequest] = []
        self.batch_actions: list[BatchAckAction] = []
        self.scratch_exists_when_ack: list[bool] = []
        self.clear_stop_calls = 0
        self.request_stop_calls = 0
        self._stop_requested = False
        self.before_return: Callable[[CaptureRequest], None] | None = None
        self.batch_process_error_after_frames: int | None = None

    def clear_stop(self) -> None:
        self.clear_stop_calls += 1
        self._stop_requested = False

    def request_stop(self) -> None:
        self.request_stop_calls += 1
        self._stop_requested = True

    def run_attempt(self, request: CaptureRequest) -> CaptureAttemptResult:
        self.requests.append(request)
        result = self._result_for(request)
        if self.before_return is not None:
            self.before_return(request)
        return result

    def run_batch_session(
        self,
        request: CaptureBatchRequest,
        *,
        frame_handler: Callable[[CaptureAttemptResult], BatchAckAction],
    ) -> CaptureBatchResult:
        self.batch_requests.append(request)
        batch_number = len(self.batch_requests)
        session_id = f"batch-session-{batch_number}"
        batch_directory = self._attempts_root / session_id
        batch_directory.mkdir()
        paths = BatchSessionPaths(
            directory=batch_directory,
            job=batch_directory / "batch-job.json",
            first_plan=batch_directory / "first-plan.jsonl",
            continuation_plan=batch_directory / "continuation-plan.jsonl",
            manifest=batch_directory / "manifest.json",
            session_journal=batch_directory / "session-journal.json",
            stdout=batch_directory / "stdout.txt",
            stderr=batch_directory / "stderr.txt",
        )
        handled: list[CaptureAttemptResult] = []
        stopped = False
        outcome = CaptureOutcome.COMPLETE
        error: str | None = None
        selected_slots = request.selected_slots
        for frame_index, frame_request in enumerate(request.frames, start=1):
            result = self._result_for(frame_request)
            if self.before_return is not None:
                self.before_return(frame_request)
            if result.outcome is not CaptureOutcome.COMPLETE:
                outcome = result.outcome
                error = None if result.journal is None else result.journal.get("error")
                break
            result = replace(
                result,
                batch_session_id=session_id,
                batch_frame_index=frame_index,
                batch_frame_total=len(request.frames),
                batch_selected_slots=selected_slots,
            )
            action = frame_handler(result)
            if self._stop_requested:
                action = BatchAckAction.STOP
            self.batch_actions.append(action)
            self.scratch_exists_when_ack.append(result.paths.output.exists())
            handled.append(result)
            if action is BatchAckAction.STOP:
                stopped = True
                break
            if self.batch_process_error_after_frames == len(handled):
                receipt = {
                    "recovery_required": "power-cycle scanner before another attempt",
                    "session_id": session_id,
                    "status": "failed",
                }
                paths.session_journal.write_text(
                    json.dumps(receipt),
                    encoding="utf-8",
                )
                raise CaptureBatchProcessError(
                    "USB endpoint stalled after the child safely released",
                    outcome=CaptureOutcome.RECOVERY_REQUIRED,
                    paths=paths,
                    frames=handled,
                    returncode=1,
                    session_journal=receipt,
                )
        receipt = {
            "error": error,
            "recovery_required": (
                "power-cycle scanner before another attempt"
                if outcome is CaptureOutcome.RECOVERY_REQUIRED
                else "none"
            ),
            "session_id": session_id,
            "status": "complete" if outcome is CaptureOutcome.COMPLETE else "failed",
        }
        paths.session_journal.write_text(json.dumps(receipt), encoding="utf-8")
        return CaptureBatchResult(
            outcome=outcome,
            request=request,
            paths=paths,
            frames=tuple(handled),
            returncode=0 if outcome is CaptureOutcome.COMPLETE else 1,
            stopped=stopped,
            session_journal=receipt,
            stdout="",
            stderr="",
        )


class _Finalizer:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, bool]] = []

    def finalize_attempt(self, attempt: Any, *, delete_scratch: bool) -> Any:
        self.calls.append((attempt, delete_scratch))
        if delete_scratch:
            attempt.stream_path.unlink(missing_ok=True)
        return SimpleNamespace(
            manifest_path=attempt.directory / "final" / "manifest.json",
            output_paths={},
        )


class _ScannerService:
    def __init__(self, result: ScanResult | None = None) -> None:
        self.result = result or _scan_result()
        self.scan_calls: list[dict[str, Any]] = []
        self.write_calls: list[dict[str, Any]] = []
        self.output_path = "/scans/bw_001.tif"

    def run_scan(self, **kwargs: Any) -> ScanResult:
        self.scan_calls.append(kwargs)
        assert not kwargs["cancel"].is_set()
        kwargs["progress"](0.25)
        kwargs["progress"](1.0)
        return self.result

    def write_result(self, **kwargs: Any) -> str:
        self.write_calls.append(kwargs)
        return self.output_path


def _scan_result(rgb: np.ndarray | None = None) -> ScanResult:
    """Return a valid RGB-only result; replace ``rgb`` for future QC cases."""

    if rgb is None:
        rgb = np.arange(6 * 8 * 3, dtype=np.uint16).reshape(6, 8, 3)
    return ScanResult(
        rgb=rgb,
        ir=None,
        dpi=4_000,
        device_model="Nikon LS-5000 ED",
    )


def _attempt_result(
    root: Path,
    request: CaptureRequest,
    *,
    outcome: CaptureOutcome = CaptureOutcome.COMPLETE,
    error: str | None = None,
) -> CaptureAttemptResult:
    suffix = f"{request.mode.value}-{len(list(root.glob('attempt-*'))) + 1}"
    directory = root / f"attempt-{suffix}"
    directory.mkdir(parents=True)
    paths = AttemptPaths(
        directory=directory,
        output=directory / "capture.bin",
        journal=directory / "journal.json",
        plan=directory / "plan.jsonl",
        manifest=directory / "manifest.json",
        stdout=directory / "stdout.log",
        stderr=directory / "stderr.log",
    )
    paths.output.write_bytes(b"packed")
    paths.journal.write_text("{}\n", encoding="utf-8")
    journal = None if error is None else {"error": error}
    return CaptureAttemptResult(
        outcome=outcome,
        request=request,
        paths=paths,
        argv=("fake-capture-worker",),
        returncode=0 if outcome is CaptureOutcome.COMPLETE else 1,
        stdout="",
        stderr="",
        journal=journal,
    )


def _worker(
    tmp_path: Path,
    *,
    session: _PreviewSession | None = None,
    result_for: Callable[[CaptureRequest], CaptureAttemptResult] | None = None,
    scanner_service: _ScannerService | None = None,
    finalizer: _Finalizer | None = None,
    promoter: Callable[..., PromotedFrame] | None = None,
) -> tuple[LS5000RollWorker, _Adapter, _PreviewSession, list[Path]]:
    attempts_root = tmp_path / "attempts"
    attempts_root.mkdir()
    preview_session = session or _PreviewSession()
    if result_for is None:

        def default_result_for(request: CaptureRequest) -> CaptureAttemptResult:
            return _attempt_result(attempts_root, request)

        result_for = default_result_for
    adapter = _Adapter(result_for, attempts_root)
    factory_roots: list[Path] = []

    def adapter_factory(root: Path) -> _Adapter:
        factory_roots.append(root)
        return adapter

    worker = LS5000RollWorker(
        adapter_factory=adapter_factory,
        scanner_service_factory=(lambda: scanner_service or _ScannerService()),
        finalizer_factory=(lambda: finalizer or _Finalizer()),
        preview_builder=lambda _result: preview_session,
        promoter=promoter or _default_promoter,
        bw_result_validator=lambda _result: None,
        bw_tiff_validator=lambda _path: None,
    )
    return worker, adapter, preview_session, factory_roots


def _default_promoter(
    _finalization: Any,
    *,
    output_folder: Path,
    filename_pattern: str,
    minimum_sequence: int,
) -> PromotedFrame:
    stem = f"frame-{minimum_sequence:03d}"
    return PromotedFrame(
        rgb_path=output_folder / f"{stem}.tif",
        ir_path=output_folder / f"{stem}_IR.tif",
        ir_valid_mask_path=output_folder / f"{stem}_IR_VALID.tif",
        receipt_path=output_folder / f"{stem}_SCAN.json",
        sequence=minimum_sequence,
    )


def _preview_request(
    tmp_path: Path,
    *,
    device_id: str = "coolscan3:libusb:001:002",
) -> RollPreviewRequest:
    return RollPreviewRequest(
        attempts_root=str(tmp_path / "attempts"),
        device_id=device_id,
        adapter_frame_capacity=40,
    )


def _load_preview(worker: LS5000RollWorker, tmp_path: Path) -> str:
    ready: list[tuple[str, Any]] = []
    worker.preview_ready.connect(lambda token, session: ready.append((token, session)))
    worker.load_preview(_preview_request(tmp_path))
    assert len(ready) == 1
    return ready[0][0]


def _scan_request(
    tmp_path: Path,
    *,
    material: ScanMaterial,
    frames: tuple[RollFrameChoice, ...],
    preview_token: str,
    device_id: str = "coolscan3:libusb:001:002",
) -> RollScanRequest:
    return RollScanRequest(
        device_id=device_id,
        adapter_frame_capacity=40,
        preview_token=preview_token,
        attempts_root=str(tmp_path / "attempts"),
        output_folder=str(tmp_path / "scans"),
        filename_pattern='roll_{{ "%03d" % seq }}',
        material=material,
        frames=frames,
    )


def test_load_preview_runs_only_preview_capture_and_publishes_session(tmp_path: Path) -> None:
    worker, adapter, session, factory_roots = _worker(tmp_path)
    previews: list[Any] = []
    progress: list[Any] = []
    errors: list[Any] = []
    worker.preview_ready.connect(lambda token, preview_session: previews.append((token, preview_session)))
    worker.progress.connect(progress.append)
    worker.error.connect(errors.append)

    _load_preview(worker, tmp_path)

    assert adapter.requests == [CaptureRequest(mode=CaptureMode.PREVIEW)]
    assert adapter.clear_stop_calls == 1
    assert factory_roots == [(tmp_path / "attempts").resolve()]
    assert len(previews) == 1
    assert previews[0][0]
    assert previews[0][1] is session
    assert [(item.completed, item.total) for item in progress] == [(0, 1), (1, 1)]
    assert [item.operation for item in progress] == [
        RollOperation.PREVIEW,
        RollOperation.PREVIEW,
    ]
    assert [(item.slot_id, item.percent) for item in progress] == [
        (None, None),
        (None, 100),
    ]
    assert progress[0].message == "Making previews · Reading the whole roll"
    assert progress[-1].message == (f"Preview complete · {len(session.slots)} scanner slots loaded")
    assert errors == []


def test_preview_stop_has_typed_preview_completion(tmp_path: Path) -> None:
    def stopped_result(_request: CaptureRequest) -> CaptureAttemptResult:
        raise CaptureStopped("stopped before preview launch")

    worker, _adapter, _session, _factory_roots = _worker(
        tmp_path,
        result_for=stopped_result,
    )
    completed: list[RollScanCompletion] = []
    worker.finished.connect(completed.append)

    worker.load_preview(_preview_request(tmp_path))

    assert completed == [
        RollScanCompletion(
            rgb_paths=(),
            black_and_white=False,
            stopped=True,
            operation=RollOperation.PREVIEW,
        )
    ]


def test_reload_thumbnail_is_offline_and_checks_same_boundary_origin(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    worker, adapter, session, _factory_roots = _worker(tmp_path)
    _load_preview(worker, tmp_path)
    reloaded = np.full((4, 96, 3), 1234, dtype=np.uint16)
    reload_calls: list[tuple[Any, Any, int]] = []

    def fake_reload(preview: Any, slot: Any, offset: int) -> np.ndarray:
        reload_calls.append((preview, slot, offset))
        return reloaded

    monkeypatch.setattr(roll_worker_module, "reload_thumbnail", fake_reload)
    thumbnails: list[tuple[int, int, np.ndarray]] = []
    errors: list[Any] = []
    worker.thumbnail_ready.connect(lambda slot, offset, image: thumbnails.append((slot, offset, image)))
    worker.error.connect(errors.append)

    worker.reload_preview_thumbnail(2, -17)

    assert reload_calls == [(session.preview, session.slots[1], -17)]
    assert session.resolve_calls == [(2, -17)]
    assert len(adapter.requests) == 1
    assert adapter.requests[0].mode is CaptureMode.PREVIEW
    assert thumbnails[0][0] == 2
    assert thumbnails[0][1] == -17
    np.testing.assert_array_equal(thumbnails[0][2], reloaded)
    assert errors == []


def test_color_scan_carries_each_boundary_offset_through_finalize_and_promote(
    tmp_path: Path,
) -> None:
    finalizer = _Finalizer()
    promotion_calls: list[dict[str, Any]] = []

    def promoter(finalization: Any, **kwargs: Any) -> PromotedFrame:
        promotion_calls.append({"finalization": finalization, **kwargs})
        return _default_promoter(finalization, **kwargs)

    worker, adapter, _session, _factory_roots = _worker(
        tmp_path,
        finalizer=finalizer,
        promoter=promoter,
    )
    preview_token = _load_preview(worker, tmp_path)
    completed: list[RollScanCompletion] = []
    frame_finished: list[tuple[int, str]] = []
    progress: list[Any] = []
    errors: list[Any] = []
    worker.finished.connect(completed.append)
    worker.frame_finished.connect(lambda slot, path: frame_finished.append((slot, path)))
    worker.progress.connect(progress.append)
    worker.error.connect(errors.append)

    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(2, -11), RollFrameChoice(7, 23)),
            preview_token=preview_token,
        )
    )

    assert adapter.requests == [CaptureRequest(mode=CaptureMode.PREVIEW)]
    assert adapter.batch_requests == [
        CaptureBatchRequest(
            frames=(
                CaptureRequest(
                    mode=CaptureMode.FULL,
                    selected_slot=2,
                    boundary_offset_rows=-11,
                ),
                CaptureRequest(
                    mode=CaptureMode.FULL,
                    selected_slot=7,
                    boundary_offset_rows=23,
                ),
            )
        )
    ]
    assert adapter.batch_actions == [
        BatchAckAction.CONTINUE,
        BatchAckAction.CONTINUE,
    ]
    assert adapter.scratch_exists_when_ack == [False, False]
    attempts = [call[0] for call in finalizer.calls]
    assert [attempt.selected_slot for attempt in attempts] == [2, 7]
    assert [attempt.boundary_offset_rows for attempt in attempts] == [-11, 23]
    assert all(attempt.session.root == (tmp_path / "attempts").resolve() for attempt in attempts)
    assert len({attempt.session.session_id for attempt in attempts}) == 1
    assert [attempt.batch_frame_index for attempt in attempts] == [1, 2]
    assert all(attempt.batch_selected_slots == (2, 7) for attempt in attempts)
    assert [delete for _attempt, delete in finalizer.calls] == [True, True]
    assert [call["minimum_sequence"] for call in promotion_calls] == [1, 2]
    assert all(call["output_folder"] == (tmp_path / "scans").resolve() for call in promotion_calls)
    assert completed == [
        RollScanCompletion(
            rgb_paths=(
                str((tmp_path / "scans" / "frame-001.tif").resolve()),
                str((tmp_path / "scans" / "frame-002.tif").resolve()),
            ),
            black_and_white=False,
            stopped=False,
        )
    ]
    assert frame_finished == [(2, completed[0].rgb_paths[0]), (7, completed[0].rgb_paths[1])]
    full_scan_messages = [item.message for item in progress]
    assert full_scan_messages == [
        "Scanning frame 1 of 2 · Slot 02",
        "Finished frame 1 of 2 · Slot 02",
        "Scanning frame 2 of 2 · Slot 07",
        "Finished frame 2 of 2 · Slot 07",
    ]
    assert [item.operation for item in progress] == [RollOperation.FULL_SCAN] * 4
    assert [item.slot_id for item in progress] == [2, 2, 7, 7]
    assert [item.percent for item in progress] == [None, 100, None, 100]
    receipts = list((tmp_path / "scans").glob("negpy-ls5000-batch-*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["session_id"] == (
        attempts[0].batch_session_id
    )
    assert errors == []


def test_bw_scan_is_4000dpi_16bit_rgb4_without_ir_at_exact_registered_geometry(
    tmp_path: Path,
) -> None:
    native_origin = 5 * LS5000_FRAME_PITCH_UNITS + 1_234
    session = _PreviewSession(origins={(7, -13): native_origin})
    scanner = _ScannerService()
    scanner.output_path = str(tmp_path / "scans" / "bw-001.tif")
    worker, adapter, _session, _factory_roots = _worker(
        tmp_path,
        session=session,
        scanner_service=scanner,
    )
    preview_token = _load_preview(worker, tmp_path)
    completed: list[RollScanCompletion] = []
    progress: list[Any] = []
    errors: list[Any] = []
    worker.finished.connect(completed.append)
    worker.progress.connect(progress.append)
    worker.error.connect(errors.append)

    request = _scan_request(
        tmp_path,
        material=ScanMaterial.BLACK_AND_WHITE_NEGATIVE,
        frames=(RollFrameChoice(7, -13),),
        preview_token=preview_token,
    )
    worker.scan_selected(request)

    assert session.resolve_calls == [(7, -13)]
    assert adapter.requests == [CaptureRequest(mode=CaptureMode.PREVIEW)]
    assert len(scanner.scan_calls) == 1
    scan_call = scanner.scan_calls[0]
    assert scan_call["device_id"] == request.device_id
    params = scan_call["params"]
    assert params.dpi == 4_000
    assert params.depth == 16
    assert params.samples_per_scan == 4
    assert params.capture_ir is False
    assert params.autofocus is True
    assert params.auto_exposure is True
    assert params.frame is None
    assert params.registered_geometry == RegisteredScanGeometry(
        frame=6,
        subframe_mm=(1_234.25 * (25.4 / 4_000)),
        br_y_device_px=LS5000_FULL_WINDOW_ROWS - 1,
    )
    assert len(scanner.write_calls) == 1
    write_call = scanner.write_calls[0]
    np.testing.assert_array_equal(
        write_call["result"].rgb,
        np.rot90(scanner.result.rgb, k=1, axes=(0, 1)),
    )
    assert write_call["output_folder"] == request.output_folder
    assert write_call["filename_pattern"] == request.filename_pattern
    assert write_call["output_format"] == "TIFF"
    assert any(item.message == "Scanning frame 1 of 1 · Slot 07 · 25%" for item in progress)
    assert [item.operation for item in progress] == [RollOperation.FULL_SCAN] * 4
    assert [item.slot_id for item in progress] == [7, 7, 7, 7]
    assert [item.percent for item in progress] == [None, 25, 100, 100]
    assert completed == [
        RollScanCompletion(
            rgb_paths=(scanner.output_path,),
            black_and_white=True,
            stopped=False,
        )
    ]
    assert errors == []


def test_stop_during_active_frame_finishes_it_and_never_launches_next_frame(
    tmp_path: Path,
) -> None:
    finalizer = _Finalizer()
    worker, adapter, _session, _factory_roots = _worker(tmp_path, finalizer=finalizer)
    preview_token = _load_preview(worker, tmp_path)
    completed: list[RollScanCompletion] = []
    worker.finished.connect(completed.append)

    def request_stop_after_active_capture(request: CaptureRequest) -> None:
        if request.mode is CaptureMode.FULL:
            worker.request_stop()

    adapter.before_return = request_stop_after_active_capture
    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(2), RollFrameChoice(3)),
            preview_token=preview_token,
        )
    )

    assert adapter.batch_requests == [
        CaptureBatchRequest(
            frames=(
                CaptureRequest(mode=CaptureMode.FULL, selected_slot=2),
                CaptureRequest(mode=CaptureMode.FULL, selected_slot=3),
            )
        )
    ]
    assert adapter.batch_actions == [BatchAckAction.STOP]
    assert adapter.scratch_exists_when_ack == [False]
    assert len(finalizer.calls) == 1
    assert finalizer.calls[0][0].selected_slot == 2
    assert finalizer.calls[0][1] is True
    assert adapter.request_stop_calls == 1
    assert completed == [
        RollScanCompletion(
            rgb_paths=(str((tmp_path / "scans" / "frame-001.tif").resolve()),),
            black_and_white=False,
            stopped=True,
        )
    ]


def test_preview_recovery_required_error_is_reported_without_exposing_session(
    tmp_path: Path,
) -> None:
    attempts_root = tmp_path / "attempts"

    def result_for(request: CaptureRequest) -> CaptureAttemptResult:
        return _attempt_result(
            attempts_root,
            request,
            outcome=CaptureOutcome.RECOVERY_REQUIRED,
            error="USB endpoint stalled",
        )

    worker, _adapter, _session, _factory_roots = _worker(tmp_path, result_for=result_for)
    errors: list[Any] = []
    previews: list[Any] = []
    worker.error.connect(errors.append)
    worker.preview_ready.connect(lambda token, preview_session: previews.append((token, preview_session)))

    worker.load_preview(_preview_request(tmp_path))

    assert previews == []
    assert len(errors) == 1
    assert errors[0].recovery_required is True
    assert errors[0].message == "Roll preview failed: USB endpoint stalled"


def test_full_color_recovery_required_error_skips_finalize_and_completion(
    tmp_path: Path,
) -> None:
    attempts_root = tmp_path / "attempts"

    def result_for(request: CaptureRequest) -> CaptureAttemptResult:
        if request.mode is CaptureMode.PREVIEW:
            return _attempt_result(attempts_root, request)
        return _attempt_result(
            attempts_root,
            request,
            outcome=CaptureOutcome.RECOVERY_REQUIRED,
            error="power cycle scanner before another attempt",
        )

    finalizer = _Finalizer()
    worker, _adapter, _session, _factory_roots = _worker(
        tmp_path,
        result_for=result_for,
        finalizer=finalizer,
    )
    preview_token = _load_preview(worker, tmp_path)
    errors: list[Any] = []
    completed: list[Any] = []
    worker.error.connect(errors.append)
    worker.finished.connect(completed.append)

    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(4),),
            preview_token=preview_token,
        )
    )

    assert finalizer.calls == []
    assert completed == []
    assert len(errors) == 1
    assert errors[0].recovery_required is True
    assert errors[0].message == "Slot 04 failed: power cycle scanner before another attempt"


def test_failure_after_completed_frame_publishes_the_partial_queue(
    tmp_path: Path,
) -> None:
    attempts_root = tmp_path / "attempts"
    full_attempt = 0

    def result_for(request: CaptureRequest) -> CaptureAttemptResult:
        nonlocal full_attempt
        if request.mode is CaptureMode.PREVIEW:
            return _attempt_result(attempts_root, request)
        full_attempt += 1
        if full_attempt == 1:
            return _attempt_result(attempts_root, request)
        return _attempt_result(
            attempts_root,
            request,
            outcome=CaptureOutcome.RECOVERY_REQUIRED,
            error="USB endpoint stalled",
        )

    worker, adapter, _session, _factory_roots = _worker(
        tmp_path,
        result_for=result_for,
    )
    preview_token = _load_preview(worker, tmp_path)
    completed: list[RollScanCompletion] = []
    errors: list[Any] = []
    worker.finished.connect(completed.append)
    worker.error.connect(errors.append)

    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(2), RollFrameChoice(3)),
            preview_token=preview_token,
        )
    )

    assert completed == [
        RollScanCompletion(
            rgb_paths=(str((tmp_path / "scans" / "frame-001.tif").resolve()),),
            black_and_white=False,
            stopped=True,
        )
    ]
    assert adapter.batch_actions == [BatchAckAction.CONTINUE]
    assert adapter.scratch_exists_when_ack == [False]
    assert len(errors) == 1
    assert errors[0].recovery_required is True
    assert errors[0].message == "Slot 03 failed: USB endpoint stalled"


def test_post_launch_batch_error_keeps_partial_paths_and_recovery_state(
    tmp_path: Path,
) -> None:
    worker, adapter, _session, _factory_roots = _worker(tmp_path)
    preview_token = _load_preview(worker, tmp_path)
    adapter.batch_process_error_after_frames = 1
    completed: list[RollScanCompletion] = []
    errors: list[Any] = []
    worker.finished.connect(completed.append)
    worker.error.connect(errors.append)

    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(2), RollFrameChoice(3)),
            preview_token=preview_token,
        )
    )

    assert completed == [
        RollScanCompletion(
            rgb_paths=(str((tmp_path / "scans" / "frame-001.tif").resolve()),),
            black_and_white=False,
            stopped=True,
        )
    ]
    assert len(errors) == 1
    assert errors[0].recovery_required is True
    assert errors[0].message == (
        "Full-quality roll scan failed: USB endpoint stalled after the child "
        "safely released"
    )
    receipts = list((tmp_path / "scans").glob("negpy-ls5000-batch-*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.parametrize("capacity", [None, 0, 6, 36, 40.0, 41])
def test_roll_requests_require_the_sa30_40_frame_capacity(
    tmp_path: Path,
    capacity: object,
) -> None:
    with pytest.raises(ValueError, match="40-frame adapter"):
        RollPreviewRequest(
            attempts_root=str(tmp_path / "attempts"),
            device_id="coolscan3:libusb:001:002",
            adapter_frame_capacity=capacity,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="40-frame adapter"):
        RollScanRequest(
            device_id="coolscan3:libusb:001:002",
            adapter_frame_capacity=capacity,  # type: ignore[arg-type]
            preview_token="preview-token",
            attempts_root=str(tmp_path / "attempts"),
            output_folder=str(tmp_path / "scans"),
            filename_pattern='roll_{{ "%03d" % seq }}',
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(1),),
        )


def test_roll_scan_request_refuses_a_non_sequenced_filename_before_hardware(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="different basename when seq changes"):
        RollScanRequest(
            device_id="coolscan3:libusb:001:002",
            adapter_frame_capacity=40,
            preview_token="preview-token",
            attempts_root=str(tmp_path / "attempts"),
            output_folder=str(tmp_path / "scans"),
            filename_pattern="constant-name",
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(1),),
        )


@pytest.mark.parametrize(
    ("preview_token", "device_id"),
    [
        ("wrong-preview-token", "coolscan3:libusb:001:002"),
        ("use-current-token", "coolscan3:libusb:009:009"),
    ],
)
def test_full_scan_rejects_preview_token_or_device_mismatch_and_invalidates(
    tmp_path: Path,
    preview_token: str,
    device_id: str,
) -> None:
    worker, adapter, _session, _factory_roots = _worker(tmp_path)
    current_token = _load_preview(worker, tmp_path)
    invalidated: list[None] = []
    errors: list[Any] = []
    worker.preview_invalidated.connect(lambda: invalidated.append(None))
    worker.error.connect(errors.append)

    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(2),),
            preview_token=(current_token if preview_token == "use-current-token" else preview_token),
            device_id=device_id,
        )
    )

    assert adapter.requests == [CaptureRequest(mode=CaptureMode.PREVIEW)]
    assert invalidated == [None]
    assert len(errors) == 1
    assert "do not match this scanner" in errors[0].message


def test_explicit_preview_invalidation_discards_token_and_coordinates(
    tmp_path: Path,
) -> None:
    worker, adapter, _session, _factory_roots = _worker(tmp_path)
    preview_token = _load_preview(worker, tmp_path)
    invalidated: list[None] = []
    errors: list[Any] = []
    worker.preview_invalidated.connect(lambda: invalidated.append(None))
    worker.error.connect(errors.append)

    worker.invalidate_preview()
    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(2),),
            preview_token=preview_token,
        )
    )

    assert adapter.requests == [CaptureRequest(mode=CaptureMode.PREVIEW)]
    assert invalidated == [None]
    assert len(errors) == 1
    assert errors[0].message == "load roll thumbnails before scanning selected frames"


def test_recovery_latch_only_clears_after_a_successful_new_preview(
    tmp_path: Path,
) -> None:
    attempts_root = tmp_path / "attempts"
    fail_next_full = True

    def result_for(request: CaptureRequest) -> CaptureAttemptResult:
        nonlocal fail_next_full
        if request.mode is CaptureMode.FULL and fail_next_full:
            fail_next_full = False
            return _attempt_result(
                attempts_root,
                request,
                outcome=CaptureOutcome.RECOVERY_REQUIRED,
                error="USB endpoint stalled",
            )
        return _attempt_result(attempts_root, request)

    finalizer = _Finalizer()
    worker, adapter, _session, _factory_roots = _worker(
        tmp_path,
        result_for=result_for,
        finalizer=finalizer,
    )
    old_token = _load_preview(worker, tmp_path)
    errors: list[Any] = []
    worker.error.connect(errors.append)
    old_request = _scan_request(
        tmp_path,
        material=ScanMaterial.COLOR_NEGATIVE,
        frames=(RollFrameChoice(3),),
        preview_token=old_token,
    )

    worker.scan_selected(old_request)
    worker.scan_selected(old_request)

    assert [request.mode for request in adapter.requests] == [CaptureMode.PREVIEW]
    assert len(adapter.batch_requests) == 1
    assert errors[-1].recovery_required is True
    assert "load a new whole-roll preview" in errors[-1].message

    new_token = _load_preview(worker, tmp_path)
    assert new_token != old_token
    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(3),),
            preview_token=new_token,
        )
    )

    assert [request.mode for request in adapter.requests] == [
        CaptureMode.PREVIEW,
        CaptureMode.PREVIEW,
    ]
    assert len(adapter.batch_requests) == 2
    assert len(finalizer.calls) == 1


def test_any_full_scan_exception_invalidates_the_bound_preview(
    tmp_path: Path,
) -> None:
    def broken_promoter(*_args: Any, **_kwargs: Any) -> PromotedFrame:
        raise RuntimeError("output promotion failed")

    worker, adapter, _session, _factory_roots = _worker(
        tmp_path,
        promoter=broken_promoter,
    )
    preview_token = _load_preview(worker, tmp_path)
    invalidated: list[None] = []
    errors: list[Any] = []
    worker.preview_invalidated.connect(lambda: invalidated.append(None))
    worker.error.connect(errors.append)
    request = _scan_request(
        tmp_path,
        material=ScanMaterial.COLOR_NEGATIVE,
        frames=(RollFrameChoice(4),),
        preview_token=preview_token,
    )

    worker.scan_selected(request)
    worker.scan_selected(request)

    assert [request.mode for request in adapter.requests] == [CaptureMode.PREVIEW]
    assert len(adapter.batch_requests) == 1
    assert invalidated == [None]
    assert "output promotion failed" in errors[0].message
    assert errors[1].message == "load roll thumbnails before scanning selected frames"
