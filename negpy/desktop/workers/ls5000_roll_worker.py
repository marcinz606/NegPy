"""Background orchestration for Nikon-style LS-5000 roll scanning.

The hardware capture remains process-isolated.  This Qt worker owns only the
application workflow around those attempts: build the thumbnail index, apply
Boundary Offset choices, finalize color RGBI captures, or route conventional
silver B&W through the proven SANE RGB4x path with infrared disabled.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date as dt_date
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.infrastructure.scanners.ls5000_single_pass.capture_process import (
    CaptureAttemptResult,
    CaptureMode,
    CaptureOutcome,
    CaptureProcessAdapter,
    CaptureRequest,
    CaptureStopped,
)
from negpy.infrastructure.scanners.params import ScanParams
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

    def __post_init__(self) -> None:
        if type(self.slot_id) is not int or not 1 <= self.slot_id <= 40:
            raise ValueError("slot_id must be an integer in 1..40")
        validate_boundary_offset(self.slot_id, self.boundary_offset_rows)


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

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id is required")
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
        slot_ids = [frame.slot_id for frame in frames]
        if slot_ids != sorted(set(slot_ids)):
            raise ValueError("selected roll slots must be unique and ordered")
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True)
class RollProgress:
    completed: int
    total: int
    message: str


@dataclass(frozen=True)
class RollScanCompletion:
    rgb_paths: tuple[str, ...]
    black_and_white: bool
    stopped: bool


@dataclass(frozen=True)
class RollWorkerFailure:
    message: str
    recovery_required: bool


AdapterFactory = Callable[[Path], CaptureProcessAdapter]
ScannerServiceFactory = Callable[[], ScannerService]
FinalizerFactory = Callable[[], LS5000SinglePassWorkflow]
ResultValidator = Callable[[object], object]
TiffValidator = Callable[[str | Path], object]


class LS5000RollWorker(QObject):
    """Serialize roll operations on the existing scanner worker thread."""

    preview_ready = pyqtSignal(str, object)  # stable token, RollPreviewSession
    preview_invalidated = pyqtSignal()
    thumbnail_ready = pyqtSignal(int, int, object)  # slot ID, offset, uint16 HxWx3
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
    ) -> None:
        super().__init__()
        self._adapter_factory = adapter_factory
        self._scanner_service_factory = scanner_service_factory
        self._finalizer_factory = finalizer_factory
        self._preview_builder = preview_builder
        self._promoter = promoter
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
            self.progress.emit(RollProgress(0, 1, "Reading the whole-roll thumbnail index"))
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
            self.progress.emit(RollProgress(1, 1, f"Loaded {len(session.slots)} scanner slots"))
            self.preview_ready.emit(token, session)
        except CaptureStopped:
            self.finished.emit(RollScanCompletion((), False, True))
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
            self.error.emit(RollWorkerFailure(f"Could not reload thumbnail: {error}", False))

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

        self._stop_after_current.clear()
        attempts_root = Path(request.attempts_root).expanduser().resolve()
        output_folder = Path(request.output_folder).expanduser().resolve()
        adapter = self._adapter_for(attempts_root)
        adapter.clear_stop()
        total = len(request.frames)
        paths: list[str] = []
        next_sequence = 1
        try:
            for index, frame in enumerate(request.frames):
                if self._stop_after_current.is_set():
                    break
                self.progress.emit(
                    RollProgress(
                        index,
                        total,
                        f"Scanning slot {frame.slot_id:02d} at full quality",
                    )
                )
                if request.material is ScanMaterial.COLOR_NEGATIVE:
                    path, next_sequence = self._scan_color_frame(
                        adapter,
                        attempts_root=attempts_root,
                        output_folder=output_folder,
                        filename_pattern=request.filename_pattern,
                        frame=frame,
                        minimum_sequence=next_sequence,
                    )
                else:
                    path = self._scan_bw_frame(
                        request,
                        session=session,
                        frame=frame,
                        progress_index=index,
                        progress_total=total,
                    )
                paths.append(path)
                self.frame_finished.emit(frame.slot_id, path)
                self.progress.emit(RollProgress(index + 1, total, f"Finished slot {frame.slot_id:02d}"))
            self.finished.emit(
                RollScanCompletion(
                    rgb_paths=tuple(paths),
                    black_and_white=request.material is ScanMaterial.BLACK_AND_WHITE_NEGATIVE,
                    stopped=self._stop_after_current.is_set(),
                )
            )
        except CaptureStopped:
            self._discard_preview()
            self.finished.emit(
                RollScanCompletion(
                    tuple(paths),
                    request.material is ScanMaterial.BLACK_AND_WHITE_NEGATIVE,
                    True,
                )
            )
        except _CaptureRefusal as refusal:
            self._emit_partial_completion(paths, request.material)
            self._discard_preview(
                recovery_required=refusal.failure.recovery_required,
            )
            self.error.emit(refusal.failure)
        except Exception as error:
            self._emit_partial_completion(paths, request.material)
            self._discard_preview()
            self.error.emit(RollWorkerFailure(f"Full-quality roll scan failed: {error}", False))

    def _emit_partial_completion(
        self,
        paths: list[str],
        material: ScanMaterial,
    ) -> None:
        """Publish frames finished before a later frame failed."""

        if paths:
            self.finished.emit(
                RollScanCompletion(
                    rgb_paths=tuple(paths),
                    black_and_white=material is ScanMaterial.BLACK_AND_WHITE_NEGATIVE,
                    stopped=True,
                )
            )

    def _scan_color_frame(
        self,
        adapter: CaptureProcessAdapter,
        *,
        attempts_root: Path,
        output_folder: Path,
        filename_pattern: str,
        frame: RollFrameChoice,
        minimum_sequence: int,
    ) -> tuple[str, int]:
        result = adapter.run_attempt(
            CaptureRequest(
                mode=CaptureMode.FULL,
                selected_slot=frame.slot_id,
                boundary_offset_rows=frame.boundary_offset_rows,
            )
        )
        if result.outcome is not CaptureOutcome.COMPLETE:
            raise _CaptureRefusal(self._failure_for_result(result, f"Slot {frame.slot_id:02d}"))
        capture_session = SinglePassSession(
            root=attempts_root,
            session_id=f"roll-{uuid.uuid4().hex}",
        )
        attempt = SinglePassAttempt.from_capture_result(
            session=capture_session,
            result=result,
        )
        finalization = self._finalizer_factory().finalize_attempt(
            attempt,
            delete_scratch=True,
        )
        promoted = self._promoter(
            finalization,
            output_folder=output_folder,
            filename_pattern=filename_pattern,
            minimum_sequence=minimum_sequence,
        )
        return str(promoted.rgb_path), promoted.sequence + 1

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
                    f"Scanning B&W slot {frame.slot_id:02d}: {percent}%",
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
        )
        try:
            self._bw_tiff_validator(output)
        except BaseException:
            Path(output).unlink(missing_ok=True)
            raise
        return output

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
) -> tuple[RollFrameChoice, ...]:
    """Build the ordered immutable request payload used by the sidebar."""

    return tuple(RollFrameChoice(slot_id=slot_id, boundary_offset_rows=offsets.get(slot_id, 0)) for slot_id in sorted(set(slot_ids)))


__all__ = [
    "LS5000RollWorker",
    "RollFrameChoice",
    "RollPreviewRequest",
    "RollProgress",
    "RollScanCompletion",
    "RollScanRequest",
    "RollWorkerFailure",
    "SA30_ADAPTER_FRAME_CAPACITY",
    "frame_choices",
]
