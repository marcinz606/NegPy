"""Background worker for coolscanpy roll scanning. Mirrors CaptureWorker.

Owns one open RollScanningService reservation (a coolscanpy `Device` plus
its `Roll` extension), held across preview/approve/scan calls and opened
lazily for whichever device the sidebar asks for -- mirrors how
CaptureWorker's `_acquire_camera()` opens and holds one libgphoto2 session
rather than reconnecting for every call.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.infrastructure.roll import coolscanpy_roll
from negpy.infrastructure.roll.repair import RepairCancelled
from negpy.infrastructure.roll.settings import RollScanSettings
from negpy.kernel.system.logging import get_logger
from negpy.services.repair.fauxice_hybrid_runner import HybridRuntimeConfig
from negpy.services.roll.service import RollFrameOutput, RollScanningService

logger = get_logger(__name__)


class RollShutdownBlocked(RuntimeError):
    """The scanner worker has not reached a state that is safe to close."""


@dataclass(frozen=True)
class RollPreviewRequest:
    device_id: str
    slots: tuple[int, ...] = ()  # empty = the whole roll


@dataclass(frozen=True)
class RollBatchScanRequest:
    device_id: str
    slots: tuple[int, ...]
    output_folder: str
    filename_pattern: str
    # Which of the three output tiers to write -- see
    # `RollScanningService.write_frame`. Defaults mirror `RollScanSettings`:
    # first-run requests retain Tier 1 and produce Hybrid repair plus the
    # Nikon-exact positive while their frame-bound evidence is available.
    write_unrepaired: bool = RollScanSettings.defaults().write_unrepaired
    write_repaired: bool = RollScanSettings.defaults().write_repaired
    write_positive: bool = RollScanSettings.defaults().write_positive
    repair_mode: str = RollScanSettings.defaults().repair_mode
    positive_mode: str = RollScanSettings.defaults().positive_mode


class RollWorker(QObject):
    """Drives one coolscanpy Device + Roll reservation off the UI thread."""

    devices_ready = pyqtSignal(list)  # list[coolscanpy.DeviceInfo]
    opened = pyqtSignal(str)  # device_id now open
    closed = pyqtSignal()
    preview_ready = pyqtSignal(list)  # list[coolscanpy.Thumbnail]
    spacing_offset_set = pyqtSignal(int, int)  # slot, offset_rows actually applied
    approved = pyqtSignal(int)  # slot
    progress = pyqtSignal(float, str)  # 0.0..1.0, message
    frame_written = pyqtSignal(object)  # RollFrameOutput
    finished = pyqtSignal(list)  # list[RollFrameOutput] written this batch
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(
        self,
        *,
        hybrid_runtime: HybridRuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self._service = RollScanningService(hybrid_runtime=hybrid_runtime)
        self._open_device_id: str | None = None
        self._operation_lock = threading.RLock()
        self._shutdown_requested = threading.Event()

    def _reject_if_shutting_down(self) -> None:
        if self._shutdown_requested.is_set():
            raise RollShutdownBlocked(
                "scanner teardown has started; no new operation may begin"
            )

    # ----- device / roll lifecycle -----

    @pyqtSlot()
    def list_devices(self) -> None:
        try:
            with self._operation_lock:
                self._reject_if_shutting_down()
                devices = self._service.list_devices()
            self.devices_ready.emit(devices)
        except Exception as e:
            logger.exception("roll device listing failed")
            self.error.emit(str(e))
            self.devices_ready.emit([])

    def _ensure_open(self, device_id: str) -> None:
        """Open `device_id`'s roll extension if it isn't already the one held.

        Switching to a different device closes the previous reservation
        first -- coolscanpy allows only one `Roll` open per device, and a
        stale reservation on the old device would otherwise sit open with
        nothing left able to use it.
        """
        self._reject_if_shutting_down()
        if self._open_device_id == device_id:
            return
        if self._open_device_id is not None:
            self._service.close()
            self._open_device_id = None
        self.status.emit("Opening roll…")
        self._service.open_roll(device_id)
        self._open_device_id = device_id
        self.opened.emit(device_id)

    @pyqtSlot()
    def close_roll(self) -> bool:
        try:
            with self._operation_lock:
                self._service.close()
        except Exception as error:
            logger.exception("error closing roll")
            self.status.emit(
                "Scanner close is unresolved; the reservation remains open."
            )
            self.error.emit(f"Close roll: {error}")
            return False
        else:
            self._open_device_id = None
            self.closed.emit()
            return True

    # ----- preview / approval -----

    @pyqtSlot(RollPreviewRequest)
    def run_preview(self, req: RollPreviewRequest) -> None:
        try:
            with self._operation_lock:
                self._reject_if_shutting_down()
                self._ensure_open(req.device_id)
                self.status.emit("Reading roll transport…")
                thumbnails = self._service.preview(
                    req.slots or None,
                    on_progress=lambda p: self.progress.emit(
                        p.fraction,
                        p.message,
                    ),
                )
            self.preview_ready.emit(thumbnails)
            self.status.emit(f"Previewed {len(thumbnails)} slot(s).")
        except Exception as e:
            logger.exception("roll preview failed")
            self.error.emit(f"Preview: {e}")

    @pyqtSlot(int, int)
    def set_spacing_offset(self, slot: int, offset_rows: int) -> None:
        try:
            with self._operation_lock:
                self._reject_if_shutting_down()
                self._service.set_spacing_offset(slot, offset_rows)
            self.spacing_offset_set.emit(slot, offset_rows)
        except Exception as e:
            logger.exception("set_spacing_offset failed")
            self.error.emit(f"Spacing offset: {e}")

    @pyqtSlot(int)
    def approve(self, slot: int) -> None:
        try:
            with self._operation_lock:
                self._reject_if_shutting_down()
                self._service.approve(slot)
            self.approved.emit(slot)
        except Exception as e:
            logger.exception("approve failed")
            self.error.emit(f"Approve: {e}")

    # ----- scanning -----

    def prepare_batch(self) -> None:
        """Reset cancellation before the controller queues this batch."""

        self._reject_if_shutting_down()
        self._service.prepare_batch()

    @pyqtSlot(RollBatchScanRequest)
    def run_batch_scan(self, req: RollBatchScanRequest) -> None:
        written: list[RollFrameOutput] = []
        try:
            with self._operation_lock:
                self._reject_if_shutting_down()
                self._ensure_open(req.device_id)
                self.status.emit("Scanning…")
                for frame in self._service.scan_many(
                    req.slots,
                    on_progress=lambda p: self.progress.emit(
                        p.fraction,
                        p.message,
                    ),
                ):
                    output = self._service.write_frame(
                        frame,
                        req.output_folder,
                        req.filename_pattern,
                        write_unrepaired=req.write_unrepaired,
                        write_repaired=req.write_repaired,
                        write_positive=req.write_positive,
                        repair_mode=req.repair_mode,
                        positive_mode=req.positive_mode,
                        on_repair_progress=lambda fraction: self.progress.emit(
                            fraction,
                            "Applying Digital ICE repair…",
                        ),
                    )
                    written.append(output)
                    self.frame_written.emit(output)
            self.finished.emit(written)
        except Exception as e:
            if coolscanpy_roll.is_safe_stop(e) or isinstance(
                e,
                RepairCancelled,
            ):
                # safe_stop() was requested deliberately (see `safe_stop` below): the frame
                # already in flight always finishes -- and is already written, already on
                # disk, already in `written` -- so this is a clean stop, not a failure.
                self.cancelled.emit()
                return
            logger.exception("roll batch scan failed")
            self.error.emit(str(e))

    def safe_stop(self) -> None:
        """Request a graceful stop of an in-progress batch scan.

        Not a `@pyqtSlot`, called directly cross-thread like `CaptureWorker
        .cancel()`: `RollScanningService.safe_stop()` only sets a
        `threading.Event`, which is thread-safe on its own and needs no Qt
        queueing. Unlike `CaptureWorker`, there is no separate local cancel
        flag here -- gphoto2 has no stop primitive of its own, so
        `CaptureWorker` keeps its own `threading.Event` and checks it
        between channels; coolscanpy already exposes `Roll.safe_stop()` for
        exactly this, so adding a second flag here would only duplicate it.
        Either way the frame already in flight finishes; only the next one
        is refused.
        """
        self._service.safe_stop()

    def shutdown(self, *, timeout_seconds: float = 5.0) -> None:
        """Stop any scan and release the roll, or block application exit.

        The operation lock prevents teardown from racing the scanner-owning
        worker path.  A timed-out or failed close leaves the service and
        device identity intact so the user can retry instead of abandoning
        an ownership-uncertain helper/reservation.
        """

        self._shutdown_requested.set()
        self.safe_stop()
        acquired = self._operation_lock.acquire(timeout=timeout_seconds)
        if not acquired:
            message = (
                "the current scanner operation is still stopping; "
                "application exit is blocked"
            )
            self.status.emit(message)
            raise RollShutdownBlocked(message)
        try:
            self._service.close()
        except Exception as error:
            logger.exception("error closing roll on shutdown")
            self.status.emit(
                "Scanner close is unresolved; application exit is blocked."
            )
            self.error.emit(f"Shutdown: {error}")
            raise
        else:
            was_open = self._open_device_id is not None
            self._open_device_id = None
            if was_open:
                self.closed.emit()
        finally:
            self._operation_lock.release()
