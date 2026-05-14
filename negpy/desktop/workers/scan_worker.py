import threading
from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

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


class ScanWorker(QObject):
    """Background worker for scanner operations. Mirrors RenderWorker pattern."""

    devices_ready = pyqtSignal(list)  # list[ScannerDevice]
    progress = pyqtSignal(float)  # 0.0..1.0
    finished = pyqtSignal(str)  # output rgb file path
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._service: ScannerService | None = None
        self._cancel_event = threading.Event()
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
        """Execute a scan and write output. Emits finished(path) or error(msg)."""
        self._cancel_event.clear()
        self._scanning = True
        try:
            service = self._ensure_service()

            result = service.run_scan(
                device_id=req.device_id,
                params=req.params,
                progress=self.progress.emit,
                cancel=self._cancel_event,
            )

            if self._cancel_event.is_set():
                return  # silently stop, no error

            path = service.write_result(
                result=result,
                output_folder=req.output_folder,
                filename_pattern=req.filename_pattern,
                output_format=req.output_format,
            )

            self.finished.emit(path)
        except Exception as e:
            if self._cancel_event.is_set():
                return
            logger.exception("Scan failed")
            self.error.emit(str(e))
        finally:
            self._scanning = False

    def cancel(self) -> None:
        """Signal the scan to stop."""
        self._cancel_event.set()
