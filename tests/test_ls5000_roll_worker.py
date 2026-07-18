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
    RollWorkerFailure,
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
    ManualFrameApproval,
    ReviewedRollFingerprint,
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
    base_origin: Any


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
        manual_slots: frozenset[int] = frozenset(),
        fingerprint: ReviewedRollFingerprint | None = None,
    ) -> None:
        self.preview = object()
        self.slots = tuple(
            _Slot(
                slot_id,
                SimpleNamespace(
                    automatic=slot_id not in manual_slots,
                    manual_review=slot_id in manual_slots,
                ),
            )
            for slot_id in range(1, slot_count + 1)
        )
        self.origins = origins or {}
        self.resolve_calls: list[tuple[int, int]] = []
        self.fingerprint = fingerprint or _reviewed_fingerprint(slot_count)

    def resolve_origin(self, slot_id: int, boundary_offset_rows: int = 0) -> int:
        self.resolve_calls.append((slot_id, boundary_offset_rows))
        key = (slot_id, boundary_offset_rows)
        if key in self.origins:
            return self.origins[key]
        return (slot_id - 1) * LS5000_FRAME_PITCH_UNITS + boundary_offset_rows

    def reviewed_fingerprint(self) -> ReviewedRollFingerprint:
        return self.fingerprint

    def validate_manual_approval(
        self,
        approval: ManualFrameApproval,
        *,
        slot_id: int,
        boundary_offset_rows: int,
    ) -> bool:
        return (
            approval.reviewed_fingerprint_sha256 == self.fingerprint.binding_sha256
            and approval.slot == slot_id
            and approval.boundary_offset_rows == boundary_offset_rows
        )


def _reviewed_fingerprint(slot_count: int = 8) -> ReviewedRollFingerprint:
    return ReviewedRollFingerprint(
        source_preview_sha256="1" * 64,
        source_table_sha256="2" * 64,
        preview_shape=(6_104, 96, 3),
        frame_start_rows=tuple(100 + 143 * index for index in range(slot_count)),
        frame_native_origins=tuple(6_000 + 6_000 * index for index in range(slot_count)),
        frame_visual_hashes=tuple(f"{index:064x}" for index in range(slot_count)),
        frame_visual_log_spans=tuple(2.0 for _index in range(slot_count)),
    )


def _manual_approval(
    fingerprint: ReviewedRollFingerprint,
    *,
    slot: int,
    offset: int = 0,
) -> ManualFrameApproval:
    return ManualFrameApproval(
        reviewed_fingerprint_sha256=fingerprint.binding_sha256,
        slot=slot,
        boundary_offset_rows=offset,
        thumbnail_sha256="3" * 64,
        reviewed_lookup_row=2_400,
        reviewed_native_origin=100_000,
        review_reasons=("transport-origin-inferred",),
    )


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
    selected_slot: int | None = None,
) -> PromotedFrame:
    del selected_slot
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
    reviewed_fingerprint: ReviewedRollFingerprint | None = None,
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
        reviewed_fingerprint=(reviewed_fingerprint or _reviewed_fingerprint()),
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


def test_reload_thumbnail_failure_is_typed_and_never_terminal(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    worker, _adapter, _session, _factory_roots = _worker(tmp_path)
    _load_preview(worker, tmp_path)

    def fail_reload(_preview: Any, _slot: Any, _offset: int) -> np.ndarray:
        raise ValueError("offset is outside the decoded frame")

    monkeypatch.setattr(roll_worker_module, "reload_thumbnail", fail_reload)
    reload_errors: list[Any] = []
    terminal_errors: list[Any] = []
    worker.thumbnail_error.connect(reload_errors.append)
    worker.error.connect(terminal_errors.append)

    worker.reload_preview_thumbnail(2, -17)

    assert reload_errors == [
        roll_worker_module.RollThumbnailReloadFailure(
            slot_id=2,
            boundary_offset_rows=-17,
            message=(
                "Could not reload thumbnail: offset is outside the decoded frame"
            ),
        )
    ]
    assert terminal_errors == []


def test_inferred_transport_origin_requires_exact_operator_approval_before_batch(
    tmp_path: Path,
) -> None:
    fingerprint = _reviewed_fingerprint()
    session = _PreviewSession(
        manual_slots=frozenset({2}),
        fingerprint=fingerprint,
    )
    worker, adapter, _session, _factory_roots = _worker(tmp_path, session=session)
    preview_token = _load_preview(worker, tmp_path)
    errors: list[Any] = []
    worker.error.connect(errors.append)

    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(2),),
            preview_token=preview_token,
            reviewed_fingerprint=fingerprint,
        )
    )

    assert adapter.batch_requests == []
    assert errors[-1].message == (
        "slot 2 uses an inferred transport origin; approve its current thumbnail before scanning"
    )

    approval = _manual_approval(fingerprint, slot=2)
    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(2, manual_review_approval=approval),),
            preview_token=preview_token,
            reviewed_fingerprint=fingerprint,
        )
    )

    assert adapter.batch_requests[-1].reviewed_fingerprint == fingerprint
    assert adapter.batch_requests[-1].frames[0].manual_review_approval == approval


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
            ),
            reviewed_fingerprint=_reviewed_fingerprint(),
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
    assert write_call["slot"] == 7
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
            ),
            reviewed_fingerprint=_reviewed_fingerprint(),
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


def test_failed_receipt_promotion_cannot_hide_scanner_recovery_state(
    tmp_path: Path,
    monkeypatch: Any,
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

    worker, _adapter, _session, _factory_roots = _worker(
        tmp_path,
        result_for=result_for,
    )
    preview_token = _load_preview(worker, tmp_path)

    def fail_receipt_promotion(*_args: Any, **_kwargs: Any) -> Path:
        raise OSError("receipt filesystem is read-only")

    monkeypatch.setattr(
        worker,
        "_promote_batch_session_receipt",
        fail_receipt_promotion,
    )
    errors: list[RollWorkerFailure] = []
    worker.error.connect(errors.append)

    worker.scan_selected(
        _scan_request(
            tmp_path,
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(4),),
            preview_token=preview_token,
        )
    )

    assert len(errors) == 1
    assert errors[0].recovery_required is True
    assert "power cycle scanner before another attempt" in errors[0].message
    assert "could not preserve batch receipt" in errors[0].message
    assert "receipt filesystem is read-only" in errors[0].message


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


# --- Digital ICE roll batches -------------------------------------------------


class _IceKit:
    """Injectable fakes for the three ICE seams, with call-order recording."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.order: list[str] = []
        self.probe_backends: list[str] = []
        self.acquire_calls: list[dict[str, Any]] = []
        self.process_calls: list[dict[str, Any]] = []
        self.publish_calls: list[dict[str, Any]] = []
        self.probe_error: Exception | None = None
        self.process_error: Exception | None = None
        self.after_publish: Callable[[int], None] | None = None
        self.hybrid_config: Any = None
        self.hybrid_config_error: Exception | None = None
        self.hybrid_error: Exception | None = None
        self.hybrid_runs: list[dict[str, Any]] = []
        self.hybrid_publishes: list[dict[str, Any]] = []

    def prober(self, backend: str) -> None:
        self.order.append(f"probe:{backend}")
        self.probe_backends.append(backend)
        if self.probe_error is not None:
            raise self.probe_error

    def acquirer(self, *, device_id: str, plan: Any, bundle_root: Path, run_id: str, progress=None) -> Path:
        self.order.append(f"acquire:{plan.frame}")
        if progress is not None:
            progress(0.5)
        bundle_dir = Path(bundle_root) / run_id
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "receipt.json").write_text("{}", encoding="utf-8")
        self.acquire_calls.append(
            {"device_id": device_id, "plan": plan, "bundle_dir": bundle_dir}
        )
        return bundle_dir

    def processor(self, bundle_dir: Path, *, backend: str, progress=None) -> Any:
        self.order.append(f"process:{Path(bundle_dir).name}")
        if progress is not None:
            progress(SimpleNamespace(completed=6, total=12))
        if self.process_error is not None:
            raise self.process_error
        processed = SimpleNamespace(bundle_dir=Path(bundle_dir), backend=backend)
        self.process_calls.append({"bundle_dir": Path(bundle_dir), "backend": backend})
        return processed

    def publisher(
        self,
        processed: Any,
        *,
        service: Any,
        output_folder: str,
        filename_pattern: str,
        roll_slot: int,
        boundary_offset_rows: int,
    ) -> str:
        self.order.append(f"publish:{roll_slot}")
        self.publish_calls.append(
            {
                "processed": processed,
                "roll_slot": roll_slot,
                "boundary_offset_rows": boundary_offset_rows,
            }
        )
        path = Path(output_folder) / f"ice_{roll_slot:02d}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ice")
        if self.after_publish is not None:
            self.after_publish(roll_slot)
        return str(path)

    def hybrid_config_loader(self) -> Any:
        self.order.append("hybrid-config")
        if self.hybrid_config_error is not None:
            raise self.hybrid_config_error
        self.hybrid_config = SimpleNamespace(inpaint_device="cpu")
        return self.hybrid_config

    def hybrid_runner(self, bundle_dir: Path, out_dir: Path, *, config: Any, backend: str) -> Any:
        self.order.append(f"hybrid-run:{Path(bundle_dir).name}")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True)
        (out_dir / "hybrid-receipt.json").write_text("{}", encoding="utf-8")
        if self.hybrid_error is not None:
            raise self.hybrid_error
        run = {"bundle_dir": Path(bundle_dir), "out_dir": out_dir, "config": config, "backend": backend}
        self.hybrid_runs.append(run)
        return SimpleNamespace(out_dir=out_dir, marker="hybrid-outputs")

    def hybrid_publisher(
        self,
        outputs: Any,
        *,
        service: Any,
        output_folder: str,
        filename_pattern: str,
        roll_slot: int,
        boundary_offset_rows: int,
    ) -> str:
        self.order.append(f"hybrid-publish:{roll_slot}")
        self.hybrid_publishes.append(
            {"outputs": outputs, "roll_slot": roll_slot, "boundary_offset_rows": boundary_offset_rows}
        )
        path = Path(output_folder) / f"hybrid_{roll_slot:02d}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"hybrid")
        return str(path)


def _ice_worker(
    tmp_path: Path,
    *,
    session: _PreviewSession | None = None,
) -> tuple[LS5000RollWorker, _IceKit, _PreviewSession]:
    attempts_root = tmp_path / "attempts"
    attempts_root.mkdir(exist_ok=True)
    preview_session = session or _PreviewSession()
    kit = _IceKit(tmp_path)
    adapter = _Adapter(lambda request: _attempt_result(attempts_root, request), attempts_root)
    worker = LS5000RollWorker(
        adapter_factory=lambda _root: adapter,
        scanner_service_factory=_ScannerService,
        finalizer_factory=_Finalizer,
        preview_builder=lambda _result: preview_session,
        promoter=_default_promoter,
        bw_result_validator=lambda _result: None,
        bw_tiff_validator=lambda _path: None,
        ice_prober=kit.prober,
        ice_bundle_acquirer=kit.acquirer,
        ice_bundle_processor=kit.processor,
        ice_publisher=kit.publisher,
        hybrid_config_loader=kit.hybrid_config_loader,
        hybrid_runner=kit.hybrid_runner,
        hybrid_publisher=kit.hybrid_publisher,
    )
    return worker, kit, preview_session


def _ice_request(
    tmp_path: Path,
    *,
    preview_token: str,
    frames: tuple[RollFrameChoice, ...],
    reviewed_fingerprint: ReviewedRollFingerprint | None = None,
    ice_backend: str = "cpu-fast",
    ice_hybrid: bool = False,
) -> RollScanRequest:
    return RollScanRequest(
        device_id="coolscan3:libusb:001:002",
        adapter_frame_capacity=40,
        preview_token=preview_token,
        attempts_root=str(tmp_path / "attempts"),
        output_folder=str(tmp_path / "scans"),
        filename_pattern='roll_{{ "%03d" % seq }}',
        material=ScanMaterial.COLOR_NEGATIVE,
        frames=frames,
        reviewed_fingerprint=reviewed_fingerprint,
        ice_capture=True,
        ice_backend=ice_backend,
        ice_hybrid=ice_hybrid,
    )


def test_ice_request_requires_color_negative_and_a_processing_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="infrared-transparent color"):
        RollScanRequest(
            device_id="coolscan3:libusb:001:002",
            adapter_frame_capacity=40,
            preview_token="token",
            attempts_root=str(tmp_path / "attempts"),
            output_folder=str(tmp_path / "scans"),
            filename_pattern='roll_{{ "%03d" % seq }}',
            material=ScanMaterial.BLACK_AND_WHITE_NEGATIVE,
            frames=(RollFrameChoice(slot_id=1),),
            ice_capture=True,
        )
    with pytest.raises(ValueError, match="processing backend"):
        _ice_request(
            tmp_path,
            preview_token="token",
            frames=(RollFrameChoice(slot_id=1),),
            ice_backend="off",
        )


def test_ice_batch_probes_then_acquires_processes_and_publishes_each_slot(tmp_path: Path) -> None:
    worker, kit, session = _ice_worker(tmp_path)
    token = _load_preview(worker, tmp_path)
    finished: list[Any] = []
    frame_events: list[tuple[int, str]] = []
    worker.finished.connect(finished.append)
    worker.frame_finished.connect(lambda slot, path: frame_events.append((slot, path)))

    worker.scan_selected(
        _ice_request(
            tmp_path,
            preview_token=token,
            frames=(
                RollFrameChoice(slot_id=2),
                RollFrameChoice(slot_id=5, boundary_offset_rows=17),
            ),
            reviewed_fingerprint=session.fingerprint,
        )
    )

    assert kit.order == [
        "probe:cpu-fast",
        "acquire:2",
        "process:" + kit.acquire_calls[0]["bundle_dir"].name,
        "publish:2",
        "acquire:5",
        "process:" + kit.acquire_calls[1]["bundle_dir"].name,
        "publish:5",
    ]
    # The reviewed offset rides into the plan through the real geometry
    # translation: slot 5 offset 17 → native origin 4*pitch+17 → frame 5.
    assert [call["plan"].frame for call in kit.acquire_calls] == [2, 5]
    assert kit.acquire_calls[1]["plan"].subframe_mm is not None
    assert [call["backend"] for call in kit.process_calls] == ["cpu-fast", "cpu-fast"]
    assert [call["roll_slot"] for call in kit.publish_calls] == [2, 5]
    assert [call["boundary_offset_rows"] for call in kit.publish_calls] == [0, 17]
    # Published bundles are cleaned up.
    assert not kit.acquire_calls[0]["bundle_dir"].exists()
    assert not kit.acquire_calls[1]["bundle_dir"].exists()
    assert [slot for slot, _path in frame_events] == [2, 5]
    [completion] = finished
    assert completion.ice_cleaned is True
    assert completion.black_and_white is False
    assert completion.stopped is False
    assert len(completion.rgb_paths) == 2


def test_ice_probe_failure_stops_before_any_hardware(tmp_path: Path) -> None:
    worker, kit, session = _ice_worker(tmp_path)
    token = _load_preview(worker, tmp_path)
    errors: list[Any] = []
    worker.error.connect(errors.append)
    kit.probe_error = RuntimeError("portable Digital ICE compiled CPU backend is unavailable")

    worker.scan_selected(
        _ice_request(
            tmp_path,
            preview_token=token,
            frames=(RollFrameChoice(slot_id=1),),
            reviewed_fingerprint=session.fingerprint,
        )
    )

    assert kit.order == ["probe:cpu-fast"]
    assert kit.acquire_calls == []
    [failure] = errors
    assert "unavailable" in failure.message


def test_ice_processing_failure_preserves_the_bundle_and_partial_queue(tmp_path: Path) -> None:
    worker, kit, session = _ice_worker(tmp_path)
    token = _load_preview(worker, tmp_path)
    finished: list[Any] = []
    errors: list[Any] = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    fail_on_second = {"count": 0}

    original_processor = kit.processor

    def flaky_processor(bundle_dir: Path, *, backend: str, progress=None) -> Any:
        fail_on_second["count"] += 1
        if fail_on_second["count"] == 2:
            kit.order.append(f"process:{Path(bundle_dir).name}")
            raise RuntimeError("engine refused the frame")
        return original_processor(bundle_dir, backend=backend, progress=progress)

    worker._ice_bundle_processor = flaky_processor

    worker.scan_selected(
        _ice_request(
            tmp_path,
            preview_token=token,
            frames=(
                RollFrameChoice(slot_id=1),
                RollFrameChoice(slot_id=2),
            ),
            reviewed_fingerprint=session.fingerprint,
        )
    )

    # Frame 1 published; frame 2's verified bundle survives for reprocessing.
    assert [call["roll_slot"] for call in kit.publish_calls] == [1]
    surviving = kit.acquire_calls[1]["bundle_dir"]
    assert surviving.exists()
    [completion] = finished
    assert completion.stopped is True
    assert completion.ice_cleaned is True
    assert len(completion.rgb_paths) == 1
    [failure] = errors
    assert "slot 2" in failure.message
    assert str(surviving) in failure.message
    assert "without a rescan" in failure.message


def test_ice_stop_after_current_finishes_the_frame_then_halts(tmp_path: Path) -> None:
    worker, kit, session = _ice_worker(tmp_path)
    token = _load_preview(worker, tmp_path)
    finished: list[Any] = []
    worker.finished.connect(finished.append)
    kit.after_publish = lambda _slot: worker.request_stop()

    worker.scan_selected(
        _ice_request(
            tmp_path,
            preview_token=token,
            frames=(
                RollFrameChoice(slot_id=1),
                RollFrameChoice(slot_id=2),
            ),
            reviewed_fingerprint=session.fingerprint,
        )
    )

    # Frame 1 completed fully (including publication); frame 2 never started.
    assert [call["roll_slot"] for call in kit.publish_calls] == [1]
    assert len(kit.acquire_calls) == 1
    [completion] = finished
    assert completion.stopped is True
    assert completion.ice_cleaned is True
    assert len(completion.rgb_paths) == 1


def test_ice_batch_refuses_an_origin_that_left_its_reviewed_slot(tmp_path: Path) -> None:
    session = _PreviewSession(
        origins={(3, 0): 3 * LS5000_FRAME_PITCH_UNITS + 5},
    )
    worker, kit, session = _ice_worker(tmp_path, session=session)
    token = _load_preview(worker, tmp_path)
    errors: list[Any] = []
    worker.error.connect(errors.append)

    worker.scan_selected(
        _ice_request(
            tmp_path,
            preview_token=token,
            frames=(RollFrameChoice(slot_id=3),),
            reviewed_fingerprint=session.fingerprint,
        )
    )

    assert kit.acquire_calls == []
    [failure] = errors
    assert "not reviewed slot 3" in failure.message


def test_hybrid_request_requires_ice_capture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hybrid repair requires Digital ICE"):
        RollScanRequest(
            device_id="coolscan3:libusb:001:002",
            adapter_frame_capacity=40,
            preview_token="token",
            attempts_root=str(tmp_path / "attempts"),
            output_folder=str(tmp_path / "scans"),
            filename_pattern='roll_{{ "%03d" % seq }}',
            material=ScanMaterial.COLOR_NEGATIVE,
            frames=(RollFrameChoice(slot_id=1),),
            ice_capture=False,
            ice_hybrid=True,
        )


def test_hybrid_batch_resolves_config_before_probe_and_cleans_both_dirs(tmp_path: Path) -> None:
    worker, kit, session = _ice_worker(tmp_path)
    token = _load_preview(worker, tmp_path)
    finished: list[Any] = []
    worker.finished.connect(finished.append)

    worker.scan_selected(
        _ice_request(
            tmp_path,
            preview_token=token,
            frames=(RollFrameChoice(slot_id=4),),
            reviewed_fingerprint=session.fingerprint,
            ice_hybrid=True,
        )
    )

    bundle = kit.acquire_calls[0]["bundle_dir"]
    assert kit.order == [
        "hybrid-config",
        "probe:cpu-fast",
        "acquire:4",
        f"hybrid-run:{bundle.name}",
        "hybrid-publish:4",
    ]
    [run] = kit.hybrid_runs
    assert run["backend"] == "cpu-fast"
    assert run["config"] is kit.hybrid_config
    assert run["out_dir"] == bundle.with_name(bundle.name + "-hybrid")
    # Both the bundle and the hybrid output directory are cleaned up.
    assert not bundle.exists()
    assert not run["out_dir"].exists()
    [completion] = finished
    assert completion.ice_cleaned is True
    assert len(completion.rgb_paths) == 1
    assert completion.rgb_paths[0].endswith("hybrid_04.tif")


def test_hybrid_unconfigured_fails_before_any_probe_or_hardware(tmp_path: Path) -> None:
    worker, kit, session = _ice_worker(tmp_path)
    token = _load_preview(worker, tmp_path)
    errors: list[Any] = []
    worker.error.connect(errors.append)
    kit.hybrid_config_error = RuntimeError(
        "hybrid repair is not configured; launch through the NegPy ICE launcher script"
    )

    worker.scan_selected(
        _ice_request(
            tmp_path,
            preview_token=token,
            frames=(RollFrameChoice(slot_id=1),),
            reviewed_fingerprint=session.fingerprint,
            ice_hybrid=True,
        )
    )

    assert kit.order == ["hybrid-config"]
    assert kit.acquire_calls == []
    [failure] = errors
    assert "not configured" in failure.message


def test_hybrid_failure_preserves_bundle_and_partial_output_dir(tmp_path: Path) -> None:
    worker, kit, session = _ice_worker(tmp_path)
    token = _load_preview(worker, tmp_path)
    errors: list[Any] = []
    worker.error.connect(errors.append)
    kit.hybrid_error = RuntimeError("inpainter refused the frame")

    worker.scan_selected(
        _ice_request(
            tmp_path,
            preview_token=token,
            frames=(RollFrameChoice(slot_id=2),),
            reviewed_fingerprint=session.fingerprint,
            ice_hybrid=True,
        )
    )

    bundle = kit.acquire_calls[0]["bundle_dir"]
    out_dir = bundle.with_name(bundle.name + "-hybrid")
    assert bundle.exists()
    assert out_dir.exists()
    [failure] = errors
    assert str(bundle) in failure.message
    assert str(out_dir) in failure.message
    assert "without a rescan" in failure.message
