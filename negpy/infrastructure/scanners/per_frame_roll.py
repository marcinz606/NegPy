"""RollSession for transports that scan one frame at a time."""

import threading
from collections.abc import Iterable, Iterator

from negpy.infrastructure.scanners.base import ScannerBackend, ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams, clamp_frame_offset_mm
from negpy.infrastructure.scanners.roll import RollPreview, effective_pitch_mm
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_PREVIEW_DEPTH = 8  # previews are for looking at, not for developing


class PerFrameRollSession:
    """Whole-strip preview built from single-frame scans.

    Works over any ScannerBackend — every SANE film scanner reaches a strip this
    way. A transport that reads the whole strip in one traversal implements
    RollSession directly instead of going through here.

    Holds no device open between slots: each scan opens and closes, which is what
    keeps ScannerBackend's re-enumeration recovery reachable mid-strip.
    """

    def __init__(self, backend: ScannerBackend, device: ScannerDevice, *, dpi: int) -> None:
        self._backend = backend
        self._device = device
        self._dpi = dpi
        caps = device.capabilities
        self.slot_count = max(1, caps.adapter_frame_capacity or 1)
        self.offset_range = (0.0, 1.0)
        self.supports_single_slot_preview = True
        self._pitch_mm = effective_pitch_mm(caps)
        self._offsets: dict[int, float] = {}

    def set_offset(self, slot: int, offset: float) -> None:
        self._offsets[slot] = offset

    def approve(self, slot: int) -> None:
        """No-op: a per-frame transport addresses frames directly, never infers a boundary."""

    def preview(self, slots: Iterable[int], *, cancel: threading.Event) -> Iterator[RollPreview]:
        for slot in slots:
            if cancel.is_set():
                return
            offset_mm = self._offset_mm(slot)
            params = ScanParams(
                dpi=self._dpi,
                depth=_PREVIEW_DEPTH,
                capture_ir=False,
                window=None,
                frame_offset_mm=offset_mm,
                autofocus=False,
                auto_exposure=False,
                frame=slot,
            )
            try:
                result = self._backend.scan(self._device.id, params, lambda _fraction: None, cancel)
            except Exception as error:
                if cancel.is_set():
                    return
                logger.warning("Preview of slot %s failed: %s", slot, error)
                yield RollPreview(slot=slot, error=str(error), offset=self._as_fraction(offset_mm))
                continue
            yield RollPreview(slot=slot, rgb=result.rgb, offset=self._as_fraction(offset_mm))

    def close(self) -> None:
        """Nothing is held open between slots."""

    def _offset_mm(self, slot: int) -> float:
        if not self._pitch_mm:
            return 0.0
        return clamp_frame_offset_mm(self._offsets.get(slot, 0.0) * self._pitch_mm, self._pitch_mm)

    def _as_fraction(self, offset_mm: float) -> float:
        return offset_mm / self._pitch_mm if self._pitch_mm else 0.0
