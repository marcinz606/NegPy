import dataclasses
import threading
from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.desktop.power_assertion import acquire_unattended_power_assertion
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
class BatchRequest:
    """A range scan over a roll/strip feeder: frames from..to, one per session."""

    device_id: str
    params: ScanParams  # base; frame is overridden per iteration
    output_folder: str
    filename_pattern: str
    output_format: str
    frame_from: int
    frame_to: int


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
    preview_ready = pyqtSignal(object)  # preview: raw rgb ndarray

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
            self.error.emit(str(e))
            self.devices_ready.emit([])

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
        frames = list(range(req.frame_from, req.frame_to + 1))
        total = max(1, len(frames))
        paths: list[str] = []
        outcome: tuple[str, str | None] = ("finished", None)
        try:
            service = self._ensure_service()
            for index, frame in enumerate(frames):
                if self._cancel_event.is_set():
                    outcome = ("cancelled", None)
                    break
                frame_params = dataclasses.replace(req.params, frame=frame)
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

    @pyqtSlot(ScanRequest)
    def run_preview(self, req: ScanRequest) -> None:
        """Preview scan: acquire one frame, emit rgb via preview_ready, write no file."""

        with self._state_lock:
            if not self._request_prepared:
                self._cancel_event.clear()
            self._request_prepared = False
            self._scanning = True

        outcome: tuple[str, str | None] | None = None
        rgb = None
        try:
            if self._cancel_event.is_set():
                outcome = ("cancelled", None)
            else:
                service = self._ensure_service()
                try:
                    # No-op progress: preview must not drive the sidebar's shared progress bar.
                    result = service.run_scan(
                        device_id=req.device_id,
                        params=req.params,
                        progress=lambda _fraction: None,
                        cancel=self._cancel_event,
                    )
                except Exception as error:
                    if self._cancel_event.is_set():
                        outcome = ("cancelled", None)
                    else:
                        logger.exception("Preview scan failed")
                        outcome = ("error", str(error))
                else:
                    if self._cancel_event.is_set():
                        outcome = ("cancelled", None)
                    else:
                        rgb = result.rgb
                        outcome = ("preview", None)
        except Exception as error:
            logger.exception("Could not initialize scanner service")
            outcome = ("error", str(error))
        finally:
            with self._state_lock:
                self._scanning = False

        if outcome is None:
            return
        kind, payload = outcome
        if kind == "preview":
            self.preview_ready.emit(rgb)
        elif kind == "cancelled":
            self.cancelled.emit()
        elif kind == "error":
            self.error.emit(payload or "Unknown scan error")

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
