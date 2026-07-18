"""Background orchestration for Nikon-style LS-5000 roll scanning.

The hardware capture remains process-isolated.  This Qt worker owns only the
application workflow around those attempts: build the thumbnail index, apply
Film Spacing Offset choices, finalize color RGBI captures, or route conventional
silver B&W through the proven SANE RGB4x path with infrared disabled.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date as dt_date
from enum import StrEnum
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.infrastructure.scanners.ls5000_single_pass.capture_process import (
    BatchAckAction,
    CaptureAttemptResult,
    CaptureBatchProcessError,
    CaptureBatchRequest,
    CaptureBatchResult,
    CaptureMode,
    CaptureOutcome,
    CaptureProcessAdapter,
    CaptureRequest,
    CaptureStopped,
    ManualFrameApproval,
    ReviewedRollFingerprint,
)
from negpy.infrastructure.scanners.dice_dual_source_runner import DiceDualSourcePlan
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.portable_digital_ice_adapter import (
    PortableDigitalIceBackend,
    probe_backend,
)
from negpy.services.scanning.ls5000_ice import (
    acquire_ice_bundle,
    process_ice_bundle,
    publish_ice_frame,
)
from negpy.services.scanning.ls5000_roll_outputs import (
    PromotedFrame,
    promote_single_pass_frame,
)
from negpy.services.scanning.ls5000_roll_session import (
    RollPreviewSession,
    build_roll_preview_session,
    reload_thumbnail,
)
from negpy.services.scanning.ls5000_sane_rgb import (
    ice_geometry_for_origin,
    orient_sane_rgb_for_storage,
    sane_rgb_geometry_for_origin,
    validate_sane_rgb_result,
    validate_sane_rgb_tiff,
)
from negpy.services.scanning.ls5000_single_pass_workflow import (
    LS5000SinglePassWorkflow,
    SinglePassAttempt,
    SinglePassSession,
)
from negpy.services.scanning.roll_preview_controls import (
    ScanMaterial,
    validate_boundary_offset,
)
from negpy.services.scanning.service import ScannerService
from negpy.services.scanning.templating import require_sequence_varying_scan_filename


SA30_ADAPTER_FRAME_CAPACITY = 40


@dataclass(frozen=True)
class RollPreviewRequest:
    attempts_root: str
    device_id: str
    adapter_frame_capacity: int

    def __post_init__(self) -> None:
        if not str(self.attempts_root).strip():
            raise ValueError("attempts_root is required")
        if not str(self.device_id).strip():
            raise ValueError("device_id is required")
        if (
            type(self.adapter_frame_capacity) is not int
            or self.adapter_frame_capacity != SA30_ADAPTER_FRAME_CAPACITY
        ):
            raise ValueError(
                "whole-roll preview requires an SA-30-compatible 40-frame adapter"
            )


@dataclass(frozen=True)
class RollFrameChoice:
    slot_id: int
    boundary_offset_rows: int = 0
    manual_review_approval: ManualFrameApproval | None = None

    def __post_init__(self) -> None:
        if type(self.slot_id) is not int or not 1 <= self.slot_id <= 40:
            raise ValueError("slot_id must be an integer in 1..40")
        validate_boundary_offset(self.slot_id, self.boundary_offset_rows)
        approval = self.manual_review_approval
        if approval is not None:
            if not isinstance(approval, ManualFrameApproval):
                raise TypeError("manual_review_approval has the wrong type")
            if (
                approval.slot != self.slot_id
                or approval.boundary_offset_rows != self.boundary_offset_rows
            ):
                raise ValueError(
                    "manual_review_approval does not match the selected slot and offset"
                )


@dataclass(frozen=True)
class RollScanRequest:
    device_id: str
    adapter_frame_capacity: int
    preview_token: str
    attempts_root: str
    output_folder: str
    filename_pattern: str
    material: ScanMaterial
    frames: tuple[RollFrameChoice, ...]
    reviewed_fingerprint: ReviewedRollFingerprint | None = None
    #: Capture selected frames as dual-RGBI pairs for portable Digital ICE
    #: instead of the RGB4x+IR packed path.  Color negative only: the engine's
    #: repair method needs infrared-transparent dyes.
    ice_capture: bool = False
    #: Explicit engine backend for ICE processing; never "off" here — an ICE
    #: batch that will not process has no reason to scan.
    ice_backend: str = PortableDigitalIceBackend.CPU_FAST.value

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id is required")
        if self.ice_capture:
            if self.material is not ScanMaterial.COLOR_NEGATIVE:
                raise ValueError(
                    "Digital ICE capture requires infrared-transparent color "
                    "negative film"
                )
            if PortableDigitalIceBackend(self.ice_backend) is PortableDigitalIceBackend.OFF:
                raise ValueError("a Digital ICE batch needs a processing backend")
        if (
            type(self.adapter_frame_capacity) is not int
            or self.adapter_frame_capacity != SA30_ADAPTER_FRAME_CAPACITY
        ):
            raise ValueError(
                "whole-roll scanning requires an SA-30-compatible 40-frame adapter"
            )
        if not self.preview_token.strip():
            raise ValueError("preview_token is required")
        if not self.attempts_root.strip() or not self.output_folder.strip():
            raise ValueError("attempts_root and output_folder are required")
        if not self.filename_pattern.strip():
            raise ValueError("filename_pattern is required")
        require_sequence_varying_scan_filename(
            self.filename_pattern,
            dt_date.today().strftime("%Y%m%d"),
        )
        if not isinstance(self.material, ScanMaterial):
            raise TypeError("material must be a ScanMaterial")
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("at least one frame must be selected")
        if any(not isinstance(frame, RollFrameChoice) for frame in frames):
            raise TypeError("frames must contain RollFrameChoice values")
        if self.reviewed_fingerprint is not None and not isinstance(
            self.reviewed_fingerprint,
            ReviewedRollFingerprint,
        ):
            raise TypeError("reviewed_fingerprint has the wrong type")
        slot_ids = [frame.slot_id for frame in frames]
        if slot_ids != sorted(set(slot_ids)):
            raise ValueError("selected roll slots must be unique and ordered")
        object.__setattr__(self, "frames", frames)


class RollOperation(StrEnum):
    """Stable operation identity for UI status and completion handling."""

    PREVIEW = "preview"
    FULL_SCAN = "full-scan"


@dataclass(frozen=True)
class RollProgress:
    completed: int
    total: int
    message: str
    operation: RollOperation = RollOperation.FULL_SCAN
    slot_id: int | None = None
    percent: int | None = None


@dataclass(frozen=True)
class RollScanCompletion:
    rgb_paths: tuple[str, ...]
    black_and_white: bool
    stopped: bool
    operation: RollOperation = RollOperation.FULL_SCAN
    #: True when the paths are ICE-cleaned masters: dust is already repaired
    #: and there is no IR sidecar, so import must not re-arm the file-based
    #: IR heal on them.
    ice_cleaned: bool = False


@dataclass(frozen=True)
class RollWorkerFailure:
    message: str
    recovery_required: bool


@dataclass(frozen=True)
class RollThumbnailReloadFailure:
    """Nonterminal failure for one identified offline thumbnail adjustment."""

    slot_id: int
    boundary_offset_rows: int
    message: str


AdapterFactory = Callable[[Path], CaptureProcessAdapter]
ScannerServiceFactory = Callable[[], ScannerService]
FinalizerFactory = Callable[[], LS5000SinglePassWorkflow]
ResultValidator = Callable[..., object]
TiffValidator = Callable[[str | Path], object]


class LS5000RollWorker(QObject):
    """Serialize roll operations on the existing scanner worker thread."""

    preview_ready = pyqtSignal(str, object)  # stable token, RollPreviewSession
    preview_invalidated = pyqtSignal()
    thumbnail_ready = pyqtSignal(int, int, object)  # slot ID, offset, uint16 HxWx3
    thumbnail_error = pyqtSignal(object)  # RollThumbnailReloadFailure
    progress = pyqtSignal(object)  # RollProgress
    frame_finished = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # RollScanCompletion
    error = pyqtSignal(object)  # RollWorkerFailure

    def __init__(
        self,
        *,
        adapter_factory: AdapterFactory = CaptureProcessAdapter.packaged,
        scanner_service_factory: ScannerServiceFactory = ScannerService,
        finalizer_factory: FinalizerFactory = LS5000SinglePassWorkflow,
        preview_builder: Callable[..., RollPreviewSession] = build_roll_preview_session,
        promoter: Callable[..., PromotedFrame] = promote_single_pass_frame,
        bw_result_validator: ResultValidator = validate_sane_rgb_result,
        bw_tiff_validator: TiffValidator = validate_sane_rgb_tiff,
        ice_prober: Callable[[str], None] = probe_backend,
        ice_bundle_acquirer: Callable[..., Path] = acquire_ice_bundle,
        ice_bundle_processor: Callable[..., object] = process_ice_bundle,
        ice_publisher: Callable[..., str] = publish_ice_frame,
    ) -> None:
        super().__init__()
        self._adapter_factory = adapter_factory
        self._scanner_service_factory = scanner_service_factory
        self._finalizer_factory = finalizer_factory
        self._preview_builder = preview_builder
        self._promoter = promoter
        self._ice_prober = ice_prober
        self._ice_bundle_acquirer = ice_bundle_acquirer
        self._ice_bundle_processor = ice_bundle_processor
        self._ice_publisher = ice_publisher
        self._bw_result_validator = bw_result_validator
        self._bw_tiff_validator = bw_tiff_validator
        self._adapter: CaptureProcessAdapter | None = None
        self._adapter_root: Path | None = None
        self._preview_session: RollPreviewSession | None = None
        self._preview_token: str | None = None
        self._preview_device_id: str | None = None
        self._preview_adapter_capacity: int | None = None
        self._recovery_required = False
        self._stop_after_current = threading.Event()

    def _discard_preview(
        self,
        *,
        recovery_required: bool = False,
        notify: bool = True,
    ) -> None:
        """Remove every coordinate capable of launching a full scan.

        Recovery is intentionally latched.  Clearing thumbnails because the
        user changed devices must not also clear a hardware-recovery state;
        only a newly completed whole-roll preview proves the scanner is ready
        again.
        """

        self._preview_session = None
        self._preview_token = None
        self._preview_device_id = None
        self._preview_adapter_capacity = None
        self._recovery_required = self._recovery_required or recovery_required
        if notify:
            self.preview_invalidated.emit()

    @pyqtSlot()
    def invalidate_preview(self) -> None:
        """Invalidate coordinates after device changes or an explicit refresh."""

        self._discard_preview()

    def _adapter_for(self, root: Path) -> CaptureProcessAdapter:
        root = root.expanduser().resolve()
        if self._adapter is None or self._adapter_root != root:
            self._adapter = self._adapter_factory(root)
            self._adapter_root = root
        return self._adapter

    @staticmethod
    def _failure_for_result(result: CaptureAttemptResult, operation: str) -> RollWorkerFailure:
        journal_error = result.journal_error
        worker_error = None if result.journal is None else result.journal.get("error")
        detail = journal_error or worker_error or result.stderr.strip() or f"worker exit {result.returncode}"
        return RollWorkerFailure(
            message=f"{operation} failed: {detail}",
            recovery_required=result.outcome is CaptureOutcome.RECOVERY_REQUIRED,
        )

    @staticmethod
    def _failure_for_batch_result(
        result: CaptureBatchResult,
        operation: str,
    ) -> RollWorkerFailure:
        worker_error = result.session_journal.get("error")
        detail = worker_error or result.stderr.strip() or f"worker exit {result.returncode}"
        return RollWorkerFailure(
            message=f"{operation} failed: {detail}",
            recovery_required=result.outcome is CaptureOutcome.RECOVERY_REQUIRED,
        )

    @pyqtSlot(object)
    def load_preview(self, request: RollPreviewRequest) -> None:
        """Capture and decode the complete low-resolution roll index."""

        if not isinstance(request, RollPreviewRequest):
            self.error.emit(RollWorkerFailure("invalid roll preview request", False))
            return
        self._stop_after_current.clear()
        # A failed refresh must never leave the prior insertion's coordinates
        # available for a later full scan.
        self._discard_preview()
        try:
            adapter = self._adapter_for(Path(request.attempts_root))
            adapter.clear_stop()
            self.progress.emit(
                RollProgress(
                    0,
                    1,
                    "Making previews · Reading the whole roll",
                    operation=RollOperation.PREVIEW,
                )
            )
            result = adapter.run_attempt(CaptureRequest(mode=CaptureMode.PREVIEW))
            if result.outcome is not CaptureOutcome.COMPLETE:
                failure = self._failure_for_result(result, "Roll preview")
                self._recovery_required = (
                    self._recovery_required or failure.recovery_required
                )
                self.error.emit(failure)
                return
            session = self._preview_builder(result)
            token = uuid.uuid4().hex
            self._preview_session = session
            self._preview_token = token
            self._preview_device_id = request.device_id
            self._preview_adapter_capacity = request.adapter_frame_capacity
            self._recovery_required = False
            self.progress.emit(
                RollProgress(
                    1,
                    1,
                    f"Preview complete · {len(session.slots)} scanner slots loaded",
                    operation=RollOperation.PREVIEW,
                    percent=100,
                )
            )
            self.preview_ready.emit(token, session)
        except CaptureStopped:
            self.finished.emit(
                RollScanCompletion(
                    (),
                    False,
                    True,
                    operation=RollOperation.PREVIEW,
                )
            )
        except Exception as error:
            self.error.emit(RollWorkerFailure(f"Roll preview failed: {error}", False))

    @pyqtSlot(int, int)
    def reload_preview_thumbnail(self, slot_id: int, boundary_offset_rows: int) -> None:
        """Re-crop one thumbnail from the persisted preview without hardware I/O."""

        try:
            session = self._preview_session
            if session is None:
                raise RuntimeError("load roll thumbnails before adjusting a boundary")
            slot = session.slots[slot_id - 1]
            if slot.slot_id != slot_id:
                raise ValueError(f"unknown roll slot {slot_id}")
            thumbnail = reload_thumbnail(
                session.preview,
                slot,
                boundary_offset_rows,
            )
            # Resolve the exact native record now as well.  A visually valid
            # crop must not be accepted if the transport table cannot express
            # the same correction for the full scan.
            session.resolve_origin(slot_id, boundary_offset_rows)
            self.thumbnail_ready.emit(slot_id, boundary_offset_rows, thumbnail)
        except Exception as error:
            self.thumbnail_error.emit(
                RollThumbnailReloadFailure(
                    slot_id=slot_id,
                    boundary_offset_rows=boundary_offset_rows,
                    message=f"Could not reload thumbnail: {error}",
                )
            )

    @pyqtSlot(object)
    def scan_selected(self, request: RollScanRequest) -> None:
        """Finish every selected frame, observing stop only between frames."""

        if not isinstance(request, RollScanRequest):
            self.error.emit(RollWorkerFailure("invalid roll scan request", False))
            return
        if self._recovery_required:
            self.error.emit(
                RollWorkerFailure(
                    "scanner recovery is required; power-cycle it, then load a new whole-roll preview",
                    True,
                )
            )
            return
        session = self._preview_session
        if session is None:
            self.error.emit(RollWorkerFailure("load roll thumbnails before scanning selected frames", False))
            return
        if (
            request.preview_token != self._preview_token
            or request.device_id != self._preview_device_id
            or request.adapter_frame_capacity != self._preview_adapter_capacity
        ):
            self._discard_preview()
            self.error.emit(
                RollWorkerFailure(
                    "roll thumbnails do not match this scanner; load a new whole-roll preview",
                    False,
                )
            )
            return

        reviewed_fingerprint = request.reviewed_fingerprint
        current_fingerprint = session.reviewed_fingerprint()
        if reviewed_fingerprint is None:
            self.error.emit(
                RollWorkerFailure(
                    "reviewed roll fingerprint is missing; load and review the whole roll again",
                    False,
                )
            )
            return
        if reviewed_fingerprint != current_fingerprint:
            self._discard_preview()
            self.error.emit(
                RollWorkerFailure(
                    "reviewed roll fingerprint does not match the loaded thumbnails; load the whole roll again",
                    False,
                )
            )
            return
        for frame in request.frames:
            slot = session.slots[frame.slot_id - 1]
            origin = slot.base_origin
            requires_approval = bool(
                not origin.automatic or origin.manual_review
            )
            approval = frame.manual_review_approval
            if requires_approval:
                if approval is None:
                    self.error.emit(
                        RollWorkerFailure(
                            f"slot {frame.slot_id} uses an inferred transport origin; "
                            "approve its current thumbnail before scanning",
                            False,
                        )
                    )
                    return
                if not session.validate_manual_approval(
                    approval,
                    slot_id=frame.slot_id,
                    boundary_offset_rows=frame.boundary_offset_rows,
                ):
                    self.error.emit(
                        RollWorkerFailure(
                            f"slot {frame.slot_id} manual approval is stale or belongs "
                            "to another thumbnail; review it again",
                            False,
                        )
                    )
                    return
            if not requires_approval and approval is not None:
                self.error.emit(
                    RollWorkerFailure(
                        f"slot {frame.slot_id} no longer requires its manual approval; "
                        "review the current thumbnails again",
                        False,
                    )
                )
                return

        self._stop_after_current.clear()
        attempts_root = Path(request.attempts_root).expanduser().resolve()
        output_folder = Path(request.output_folder).expanduser().resolve()
        adapter = self._adapter_for(attempts_root)
        adapter.clear_stop()
        total = len(request.frames)
        paths: list[str] = []
        try:
            if request.ice_capture:
                stopped = self._scan_ice_batch(
                    request,
                    session=session,
                    attempts_root=attempts_root,
                    paths=paths,
                )
            elif request.material is ScanMaterial.COLOR_NEGATIVE:
                stopped = self._scan_color_batch(
                    adapter,
                    attempts_root=attempts_root,
                    output_folder=output_folder,
                    filename_pattern=request.filename_pattern,
                    frames=request.frames,
                    reviewed_fingerprint=reviewed_fingerprint,
                    paths=paths,
                )
            else:
                for index, frame in enumerate(request.frames):
                    if self._stop_after_current.is_set():
                        break
                    self.progress.emit(
                        RollProgress(
                            index,
                            total,
                            f"Scanning frame {index + 1} of {total} · "
                            f"Slot {frame.slot_id:02d}",
                            slot_id=frame.slot_id,
                        )
                    )
                    path = self._scan_bw_frame(
                        request,
                        session=session,
                        frame=frame,
                        progress_index=index,
                        progress_total=total,
                    )
                    paths.append(path)
                    self.frame_finished.emit(frame.slot_id, path)
                    self.progress.emit(
                        RollProgress(
                            index + 1,
                            total,
                            f"Finished frame {index + 1} of {total} · "
                            f"Slot {frame.slot_id:02d}",
                            slot_id=frame.slot_id,
                            percent=100,
                        )
                    )
                stopped = self._stop_after_current.is_set()
            self.finished.emit(
                RollScanCompletion(
                    rgb_paths=tuple(paths),
                    black_and_white=request.material is ScanMaterial.BLACK_AND_WHITE_NEGATIVE,
                    stopped=stopped,
                    ice_cleaned=request.ice_capture,
                )
            )
        except CaptureStopped:
            self._discard_preview()
            self.finished.emit(
                RollScanCompletion(
                    tuple(paths),
                    request.material is ScanMaterial.BLACK_AND_WHITE_NEGATIVE,
                    True,
                    ice_cleaned=request.ice_capture,
                )
            )
        except CaptureBatchProcessError as error:
            receipt_error: Exception | None = None
            if error.session_journal is not None:
                try:
                    self._promote_batch_session_receipt(
                        error.paths.session_journal,
                        error.session_journal,
                        output_folder,
                    )
                except Exception as promotion_error:
                    receipt_error = promotion_error
            self._emit_partial_completion(paths, request.material, ice_cleaned=request.ice_capture)
            self._discard_preview(
                recovery_required=error.recovery_required,
            )
            self.error.emit(
                RollWorkerFailure(
                    "Full-quality roll scan failed: "
                    f"{error}"
                    + (
                        f"; could not preserve batch receipt: {receipt_error}"
                        if receipt_error is not None
                        else ""
                    ),
                    error.recovery_required,
                )
            )
        except _CaptureRefusal as refusal:
            self._emit_partial_completion(paths, request.material, ice_cleaned=request.ice_capture)
            self._discard_preview(
                recovery_required=refusal.failure.recovery_required,
            )
            self.error.emit(refusal.failure)
        except Exception as error:
            self._emit_partial_completion(paths, request.material, ice_cleaned=request.ice_capture)
            self._discard_preview()
            self.error.emit(RollWorkerFailure(f"Full-quality roll scan failed: {error}", False))

    def _scan_color_batch(
        self,
        adapter: CaptureProcessAdapter,
        *,
        attempts_root: Path,
        output_folder: Path,
        filename_pattern: str,
        frames: tuple[RollFrameChoice, ...],
        reviewed_fingerprint: ReviewedRollFingerprint,
        paths: list[str],
    ) -> bool:
        batch_request = CaptureBatchRequest(
            frames=tuple(
                CaptureRequest(
                    mode=CaptureMode.FULL,
                    selected_slot=frame.slot_id,
                    boundary_offset_rows=frame.boundary_offset_rows,
                    manual_review_approval=frame.manual_review_approval,
                )
                for frame in frames
            ),
            reviewed_fingerprint=reviewed_fingerprint,
        )
        finalizer = self._finalizer_factory()
        capture_session: SinglePassSession | None = None
        next_sequence = 1
        total = len(frames)

        def emit_scanning(index: int) -> None:
            frame = frames[index]
            self.progress.emit(
                RollProgress(
                    index,
                    total,
                    f"Scanning frame {index + 1} of {total} · "
                    f"Slot {frame.slot_id:02d}",
                    slot_id=frame.slot_id,
                )
            )

        def finalize_frame(result: CaptureAttemptResult) -> BatchAckAction:
            nonlocal capture_session, next_sequence
            index = len(paths)
            if index >= total:
                raise RuntimeError("batch worker produced an unexpected extra frame")
            frame = frames[index]
            if (
                result.request.selected_slot != frame.slot_id
                or result.request.boundary_offset_rows != frame.boundary_offset_rows
            ):
                raise RuntimeError("batch worker frame order changed before finalization")
            if result.batch_session_id is None:
                raise RuntimeError("batch worker result has no session identity")
            if capture_session is None:
                capture_session = SinglePassSession(
                    root=attempts_root,
                    session_id=result.batch_session_id,
                )
            elif capture_session.session_id != result.batch_session_id:
                raise RuntimeError("batch worker session identity changed between frames")
            attempt = SinglePassAttempt.from_capture_result(
                session=capture_session,
                result=result,
            )
            finalization = finalizer.finalize_attempt(
                attempt,
                delete_scratch=True,
            )
            promoted = self._promoter(
                finalization,
                output_folder=output_folder,
                filename_pattern=filename_pattern,
                minimum_sequence=next_sequence,
                selected_slot=frame.slot_id,
            )
            path = str(promoted.rgb_path)
            next_sequence = promoted.sequence + 1
            paths.append(path)
            self.frame_finished.emit(frame.slot_id, path)
            self.progress.emit(
                RollProgress(
                    index + 1,
                    total,
                    f"Finished frame {index + 1} of {total} · "
                    f"Slot {frame.slot_id:02d}",
                    slot_id=frame.slot_id,
                    percent=100,
                )
            )
            if self._stop_after_current.is_set():
                return BatchAckAction.STOP
            if index + 1 < total:
                emit_scanning(index + 1)
            return BatchAckAction.CONTINUE

        emit_scanning(0)
        result = adapter.run_batch_session(
            batch_request,
            frame_handler=finalize_frame,
        )
        if result.outcome is not CaptureOutcome.COMPLETE:
            failed_index = len(result.frames)
            operation = "Full-quality roll scan"
            if failed_index < total:
                operation = f"Slot {frames[failed_index].slot_id:02d}"
            failure = self._failure_for_batch_result(result, operation)
            try:
                self._promote_batch_session_receipt(
                    result.paths.session_journal,
                    result.session_journal,
                    output_folder,
                )
            except Exception as receipt_error:
                failure = RollWorkerFailure(
                    f"{failure.message}; could not preserve batch receipt: {receipt_error}",
                    failure.recovery_required,
                )
            raise _CaptureRefusal(failure)
        self._promote_batch_session_receipt(
            result.paths.session_journal,
            result.session_journal,
            output_folder,
        )
        return result.stopped or self._stop_after_current.is_set()

    @staticmethod
    def _promote_batch_session_receipt(
        source: Path,
        expected: dict[str, Any],
        output_folder: Path,
    ) -> Path:
        """Copy the immutable final release receipt beside promoted frames."""

        payload = source.read_bytes()
        if json.loads(payload) != expected:
            raise RuntimeError("batch session receipt changed after validation")
        session_id = expected.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("batch session receipt has no session identity")
        output_folder.mkdir(parents=True, exist_ok=True)
        destination = output_folder / f"negpy-ls5000-batch-{session_id}.json"
        with destination.open("xb") as stream:
            if stream.write(payload) != len(payload):
                raise OSError("short batch session receipt write")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(output_folder, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination

    def _emit_partial_completion(
        self,
        paths: list[str],
        material: ScanMaterial,
        *,
        ice_cleaned: bool = False,
    ) -> None:
        """Publish frames finished before a later frame failed."""

        if paths:
            self.finished.emit(
                RollScanCompletion(
                    rgb_paths=tuple(paths),
                    black_and_white=material is ScanMaterial.BLACK_AND_WHITE_NEGATIVE,
                    stopped=True,
                    ice_cleaned=ice_cleaned,
                )
            )

    def _scan_bw_frame(
        self,
        request: RollScanRequest,
        *,
        session: RollPreviewSession,
        frame: RollFrameChoice,
        progress_index: int,
        progress_total: int,
    ) -> str:
        if "coolscan3:" not in request.device_id.lower():
            raise RuntimeError("B&W roll scanning requires the patched local coolscan3 LS-5000 backend")
        origin = session.resolve_origin(frame.slot_id, frame.boundary_offset_rows)
        binding = sane_rgb_geometry_for_origin(origin)
        params = ScanParams(
            dpi=4_000,
            depth=16,
            capture_ir=False,
            autofocus=True,
            samples_per_scan=4,
            registered_geometry=binding.geometry,
            auto_exposure=True,
        )
        service = self._scanner_service_factory()
        active_cancel = threading.Event()

        def on_progress(value: float) -> None:
            percent = max(0, min(100, int(float(value) * 100)))
            self.progress.emit(
                RollProgress(
                    progress_index,
                    progress_total,
                    f"Scanning frame {progress_index + 1} of {progress_total} · "
                    f"Slot {frame.slot_id:02d} · {percent}%",
                    slot_id=frame.slot_id,
                    percent=percent,
                )
            )

        result = service.run_scan(
            device_id=request.device_id,
            params=params,
            progress=on_progress,
            # Deliberately never set during the active frame.  The user's Stop
            # action is honored between frames so it cannot strand the feeder
            # halfway through a hardware transaction.
            cancel=active_cancel,
        )
        self._bw_result_validator(result)
        storage_result = orient_sane_rgb_for_storage(result)
        output = service.write_result(
            result=storage_result,
            output_folder=request.output_folder,
            filename_pattern=request.filename_pattern,
            output_format="TIFF",
            slot=frame.slot_id,
        )
        try:
            self._bw_tiff_validator(output)
        except BaseException:
            Path(output).unlink(missing_ok=True)
            raise
        return output

    def _scan_ice_batch(
        self,
        request: RollScanRequest,
        *,
        session: RollPreviewSession,
        attempts_root: Path,
        paths: list[str],
    ) -> bool:
        """Capture, process, and publish each selected frame for Digital ICE.

        Each frame is one dedicated SANE session (the proven per-frame model
        of the B&W path), one on-disk bundle, one engine run after the handle
        is closed, and one published master.  Processing never overlaps a held
        scanner reservation, and a processing failure keeps the bundle on disk
        as the recovery artifact so the frame does not need a rescan.
        """

        if "coolscan3:" not in request.device_id.lower():
            raise RuntimeError(
                "Digital ICE roll capture requires the patched local coolscan3 "
                "LS-5000 backend"
            )
        # Prove the backend can run before any hardware is touched; the
        # compiled backends' byte-parity self-test doubles as their warmup.
        self._ice_prober(request.ice_backend)
        service = self._scanner_service_factory()
        bundle_root = attempts_root / "ice-bundles"
        total = len(request.frames)
        for index, frame in enumerate(request.frames):
            if self._stop_after_current.is_set():
                return True
            origin = session.resolve_origin(frame.slot_id, frame.boundary_offset_rows)
            binding = ice_geometry_for_origin(origin, slot_id=frame.slot_id)
            plan = DiceDualSourcePlan.for_transport(
                "roll",
                frame=binding.geometry.frame,
                subframe_mm=binding.geometry.subframe_mm,
            )

            def emit_acquiring(value: float, *, index: int = index, slot_id: int = frame.slot_id) -> None:
                percent = max(0, min(100, int(float(value) * 100)))
                self.progress.emit(
                    RollProgress(
                        index,
                        total,
                        f"Scanning frame {index + 1} of {total} · "
                        f"Slot {slot_id:02d} · dual RGBI · {percent}%",
                        slot_id=slot_id,
                        percent=percent,
                    )
                )

            bundle_dir = self._ice_bundle_acquirer(
                device_id=request.device_id,
                plan=plan,
                bundle_root=bundle_root,
                run_id=f"slot{frame.slot_id:02d}-{uuid.uuid4().hex[:8]}",
                progress=emit_acquiring,
            )
            # The SANE handle is closed; the scanner owes nothing to the rest
            # of this iteration.
            try:
                def emit_processing(event: object, *, index: int = index, slot_id: int = frame.slot_id) -> None:
                    completed = getattr(event, "completed", None)
                    event_total = getattr(event, "total", None)
                    percent: int | None = None
                    if isinstance(completed, int) and isinstance(event_total, int) and event_total > 0:
                        percent = max(0, min(100, int(completed * 100 / event_total)))
                    self.progress.emit(
                        RollProgress(
                            index,
                            total,
                            f"Digital ICE frame {index + 1} of {total} · "
                            f"Slot {slot_id:02d} · repairing",
                            slot_id=slot_id,
                            percent=percent,
                        )
                    )

                processed = self._ice_bundle_processor(
                    bundle_dir,
                    backend=request.ice_backend,
                    progress=emit_processing,
                )
                path = self._ice_publisher(
                    processed,
                    service=service,
                    output_folder=str(request.output_folder),
                    filename_pattern=request.filename_pattern,
                    roll_slot=frame.slot_id,
                    boundary_offset_rows=frame.boundary_offset_rows,
                )
            except Exception as error:
                raise RuntimeError(
                    f"Digital ICE processing failed for slot {frame.slot_id}; "
                    f"the verified capture bundle is preserved at {bundle_dir} "
                    f"and can be processed without a rescan: {error}"
                ) from error
            shutil.rmtree(bundle_dir, ignore_errors=True)
            paths.append(path)
            self.frame_finished.emit(frame.slot_id, path)
            self.progress.emit(
                RollProgress(
                    index + 1,
                    total,
                    f"Finished frame {index + 1} of {total} · "
                    f"Slot {frame.slot_id:02d} · ICE cleaned",
                    slot_id=frame.slot_id,
                    percent=100,
                )
            )
        return self._stop_after_current.is_set()

    def request_stop(self) -> None:
        """Finish the active frame, then prevent the next attempt."""

        self._stop_after_current.set()
        if self._adapter is not None:
            self._adapter.request_stop()


class _CaptureRefusal(RuntimeError):
    def __init__(self, failure: RollWorkerFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


def frame_choices(
    slot_ids: Iterable[int],
    offsets: dict[int, int],
    approvals: Mapping[int, ManualFrameApproval] | None = None,
) -> tuple[RollFrameChoice, ...]:
    """Build the ordered immutable request payload used by the sidebar."""

    approval_by_slot = {} if approvals is None else approvals
    return tuple(
        RollFrameChoice(
            slot_id=slot_id,
            boundary_offset_rows=offsets.get(slot_id, 0),
            manual_review_approval=approval_by_slot.get(slot_id),
        )
        for slot_id in sorted(set(slot_ids))
    )


__all__ = [
    "LS5000RollWorker",
    "RollFrameChoice",
    "RollOperation",
    "RollPreviewRequest",
    "RollProgress",
    "RollScanCompletion",
    "RollScanRequest",
    "RollThumbnailReloadFailure",
    "RollWorkerFailure",
    "SA30_ADAPTER_FRAME_CAPACITY",
    "frame_choices",
]
