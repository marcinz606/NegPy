import dataclasses
import threading
from dataclasses import dataclass, field

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.desktop.power_assertion import acquire_unattended_power_assertion
from negpy.infrastructure.scanners.base import ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams
from negpy.services.scanning.service import ScannerService
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScanRequest:
    device_id: str
    params: ScanParams
    output_folder: str
    filename_pattern: str
    output_format: str  # "TIFF" or "DNG"


@dataclass(frozen=True)
class RollPreviewRequest:
    """Preview a set of strip slots. Offsets are fractions of one frame pitch, raw —
    the session clamps them and reports back what it reached."""

    device: ScannerDevice
    slots: tuple[int, ...]
    dpi: int
    offsets: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchRequest:
    """Scan an explicit set of frames, one SANE session each, frame-numbered output."""

    device_id: str
    params: ScanParams  # base; frame + window + offset overridden per iteration
    output_folder: str
    filename_pattern: str
    output_format: str
    frames: tuple[int, ...]
    frame_windows: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    # Feed-axis drift (mm/frame): frame N scans at
    # frame_offset_mm + (N-1) * modifier, floored at 0.
    frame_offset_modifier_mm: float = 0.0


class ScanWorker(QObject):
    """Background worker for scanner operations. Mirrors RenderWorker pattern."""

    devices_ready = pyqtSignal(list)  # list[ScannerDevice]
    progress = pyqtSignal(float)  # 0.0..1.0
    finished = pyqtSignal(str)  # output rgb file path
    frame_done = pyqtSignal(int, str)  # batch: frame number, rgb file path
    batch_finished = pyqtSignal(list)  # batch: all written rgb paths (also on stop/error)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    ejected = pyqtSignal(bool)
    eject_error = pyqtSignal(str)
    roll_preview_ready = pyqtSignal(object)  # roll preview: one RollPreview per slot
    roll_preview_finished = pyqtSignal()  # the whole strip is done (also after a failed slot)

    def __init__(self) -> None:
        super().__init__()
        self._service: ScannerService | None = None
        self._cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._request_prepared = False
        self._scanning = False

    def _ensure_service(self) -> ScannerService:
        if self._service is None:
            self._service = ScannerService()
        return self._service

    @pyqtSlot()
    def list_devices(self) -> None:
        """Fetch devices on a background thread. Emit devices_ready on finish."""
        try:
            service = self._ensure_service()
            devices = service.refresh_devices()
            self.devices_ready.emit(devices)
        except Exception as e:
            logger.exception("Device listing failed")
            # Empty list first: the sidebar's devices_ready handler overwrites the
            # status label, so emitting it last would clobber the failure message
            # (which carries the backend's install hint).
            self.devices_ready.emit([])
            self.error.emit(str(e))

    @pyqtSlot(ScanRequest)
    def run_scan(self, req: ScanRequest) -> None:
        """Execute a scan and emit exactly one terminal outcome."""

        # AppController prepares a queued request synchronously on the GUI
        # thread.  Do not clear its Event here: Stop may have arrived after the
        # request was queued but before this slot began running.  Direct legacy
        # callers that skip prepare_scan() still receive a fresh Event.
        with self._state_lock:
            if not self._request_prepared:
                self._cancel_event.clear()
            self._request_prepared = False
            self._scanning = True

        outcome: tuple[str, str | None] | None = None
        try:
            if self._cancel_event.is_set():
                outcome = ("cancelled", None)
            else:
                service = self._ensure_service()
                try:
                    result = service.run_scan(
                        device_id=req.device_id,
                        params=req.params,
                        progress=self.progress.emit,
                        cancel=self._cancel_event,
                    )
                except Exception as error:
                    if self._cancel_event.is_set():
                        outcome = ("cancelled", None)
                    else:
                        logger.exception("Scan failed")
                        outcome = ("error", str(error))
                else:
                    if self._cancel_event.is_set():
                        outcome = ("cancelled", None)
                    else:
                        # Acquisition is complete now.  Cancellation cannot abort
                        # an in-progress file write, and it must never disguise a
                        # disk or encoder failure as a cleanly stopped scan.
                        try:
                            path = service.write_result(
                                result=result,
                                output_folder=req.output_folder,
                                filename_pattern=req.filename_pattern,
                                output_format=req.output_format,
                            )
                        except Exception as error:
                            logger.exception("Could not write scan result")
                            outcome = ("error", str(error))
                        else:
                            outcome = ("finished", path)
        except Exception as error:
            logger.exception("Could not initialize scanner service")
            outcome = ("error", str(error))
        finally:
            with self._state_lock:
                self._scanning = False

        if outcome is None:
            return
        kind, payload = outcome
        if kind == "finished":
            self.finished.emit(payload or "")
        elif kind == "cancelled":
            self.cancelled.emit()
        else:
            self.error.emit(payload or "Unknown scan error")

    @pyqtSlot(BatchRequest)
    def run_batch(self, req: BatchRequest) -> None:
        """Scan a frame range, one SANE session per frame, frame-numbered output.

        Holds an idle-sleep assertion for the whole run — a 40-frame SA-30 batch
        at 4000 dpi is a long unattended operation. `batch_finished` always fires
        with the frames that completed, so a stop or error still imports them.
        """

        with self._state_lock:
            if not self._request_prepared:
                self._cancel_event.clear()
            self._request_prepared = False
            self._scanning = True

        assertion = acquire_unattended_power_assertion("NegPy film scan batch")
        frames = list(req.frames)
        total = max(1, len(frames))
        paths: list[str] = []
        outcome: tuple[str, str | None] = ("finished", None)
        try:
            service = self._ensure_service()
            for index, frame in enumerate(frames):
                if self._cancel_event.is_set():
                    outcome = ("cancelled", None)
                    break
                window = req.frame_windows.get(frame, req.params.window)
                offset = max(0.0, req.params.frame_offset_mm + (frame - 1) * req.frame_offset_modifier_mm)
                frame_params = dataclasses.replace(req.params, frame=frame, window=window, frame_offset_mm=offset)
                base = index / total

                def _progress(fraction: float, _base: float = base) -> None:
                    self.progress.emit(_base + min(1.0, max(0.0, fraction)) / total)

                try:
                    result = service.run_scan(req.device_id, frame_params, _progress, self._cancel_event)
                except Exception as error:
                    if self._cancel_event.is_set():
                        outcome = ("cancelled", None)
                    else:
                        logger.exception("Batch frame %s scan failed", frame)
                        outcome = ("error", str(error))
                    break
                if self._cancel_event.is_set():
                    outcome = ("cancelled", None)
                    break
                try:
                    path = service.write_result(
                        result=result,
                        output_folder=req.output_folder,
                        filename_pattern=req.filename_pattern,
                        output_format=req.output_format,
                        seq=frame,
                    )
                except Exception as error:
                    logger.exception("Could not write batch frame %s", frame)
                    outcome = ("error", str(error))
                    break
                paths.append(path)
                self.frame_done.emit(frame, path)
        except Exception as error:
            logger.exception("Could not run scan batch")
            outcome = ("error", str(error))
        finally:
            assertion.release()
            with self._state_lock:
                self._scanning = False

        self.batch_finished.emit(paths)
        kind, payload = outcome
        if kind == "cancelled":
            self.cancelled.emit()
        elif kind == "error":
            self.error.emit(payload or "Unknown scan error")
        elif kind == "finished":
            # Return the strip so it need not wait for the feeder's auto-park.
            # Capability-gated no-op on devices without an eject option.
            self.eject(req.device_id)

    @pyqtSlot(RollPreviewRequest)
    def run_roll_preview(self, req: RollPreviewRequest) -> None:
        """Preview strip slots, emitting one RollPreview per slot as it lands.

        A slot that fails arrives as a RollPreview carrying `error` and the strip
        continues; only a failure to open the strip at all is a terminal `error`.
        """

        with self._state_lock:
            if not self._request_prepared:
                self._cancel_event.clear()
            self._request_prepared = False
            self._scanning = True

        outcome: tuple[str, str | None] = ("finished", None)
        try:
            if self._cancel_event.is_set():
                outcome = ("cancelled", None)
            else:
                service = self._ensure_service()
                session = service.open_roll(req.device, dpi=req.dpi)
                try:
                    for slot, offset in req.offsets.items():
                        session.set_offset(slot, offset)
                    for preview in session.preview(req.slots, cancel=self._cancel_event):
                        self.roll_preview_ready.emit(preview)
                finally:
                    session.close()
                if self._cancel_event.is_set():
                    outcome = ("cancelled", None)
        except Exception as error:
            logger.exception("Could not preview the strip")
            outcome = ("cancelled", None) if self._cancel_event.is_set() else ("error", str(error))
        finally:
            with self._state_lock:
                self._scanning = False

        kind, payload = outcome
        if kind == "cancelled":
            self.cancelled.emit()
        elif kind == "error":
            self.error.emit(payload or "Unknown scan error")
        else:
            self.roll_preview_finished.emit()

    def prepare_scan(self) -> None:
        """Arm one queued scan without losing a Stop pressed before it starts."""

        with self._state_lock:
            if self._scanning or self._request_prepared:
                raise RuntimeError("A scanner request is already active")
            self._cancel_event.clear()
            self._request_prepared = True

    def cancel(self) -> None:
        """Signal the scan to stop."""
        self._cancel_event.set()

    @pyqtSlot(str)
    def eject(self, device_id: str) -> None:
        """Eject the loaded medium without blocking the UI thread."""

        if self._scanning:
            self.eject_error.emit("Cannot eject while a scan is active")
            return
        try:
            self.ejected.emit(self._ensure_service().eject(device_id))
        except Exception as error:
            logger.exception("Film eject failed")
            self.eject_error.emit(str(error))
