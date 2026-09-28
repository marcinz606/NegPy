"""A film scanner backend with no device behind it: one device per shape of Scan panel.

Feeder (frames by index, IR, eject), prescan (full-window preview, then crop) and roll
(frames found on the strip, boundaries confirmed by the user).
"""

import threading
from collections.abc import Iterable, Iterator
from typing import Callable

import numpy as np

from negpy.infrastructure.scanners.base import ScanMode, ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.nkscan_backend import FILM_FORMATS
from negpy.infrastructure.scanners.params import FILM_TYPES, ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.infrastructure.scanners.roll import RollPreview
from negpy.infrastructure.simulated.images import negative

_AREA_MM = (36.0, 24.0)
_MAX_EDGE_PX = 2000  # a 4000 dpi frame at full size makes every simulated scan slow
_PROGRESS_STEPS = 10
_STEP_S = 0.05
_ROLL_FRAMES = 6

DEVICES = (
    ScannerDevice(
        id="sim:feeder",
        vendor="Simulated",
        model="Coolscan (feeder)",
        capabilities=ScannerCapabilities(
            ir_channel=True,
            supported_dpi=(1000, 2000, 4000),
            supported_depths=(8, 16),
            sources=(ScanMode.NEGATIVE, ScanMode.POSITIVE),
            max_area_mm=_AREA_MM,
            auto_exposure=True,
            autofocus=True,
            multi_exposure=True,
            max_n_passes=4,
            adapter_frame_capacity=6,
            adapter_frame_control=True,
            can_eject=True,
            frame_pitch_mm=38.0,
            exposure_time_us=(100, 100_000),
            max_samples=16,
        ),
    ),
    ScannerDevice(
        id="sim:prescan",
        vendor="Simulated",
        model="OpticFilm (prescan)",
        capabilities=ScannerCapabilities(
            ir_channel=True,
            supported_dpi=(900, 1800, 3600, 7200),
            supported_depths=(8, 16),
            sources=(ScanMode.TRANSPARENCY,),
            max_area_mm=_AREA_MM,
            prescan=True,
            prescan_dpi=300,
        ),
    ),
    ScannerDevice(
        id="sim:roll",
        vendor="Simulated",
        model="LS-50 (roll)",
        capabilities=ScannerCapabilities(
            ir_channel=True,
            supported_dpi=(1000, 2000, 4000),
            supported_depths=(16,),
            sources=(ScanMode.NEGATIVE, ScanMode.POSITIVE),
            max_area_mm=_AREA_MM,
            can_eject=True,
            hw_clean=True,
            roll_discovery=True,
            film_formats=FILM_FORMATS,
            film_types=tuple(FILM_TYPES),
            max_samples=16,
            superfine=True,
            exposure_lock=True,
        ),
    ),
)
_BY_ID = {d.id: d for d in DEVICES}


def _frame(params: ScanParams, dpi: int) -> tuple[np.ndarray, np.ndarray]:
    w_mm, h_mm = _AREA_MM
    if params.window is not None:
        x1, y1, x2, y2 = params.window
        w_mm, h_mm = w_mm * (x2 - x1), h_mm * (y2 - y1)
    w, h = w_mm / 25.4 * dpi, h_mm / 25.4 * dpi
    scale = min(1.0, _MAX_EDGE_PX / max(w, h, 1.0))
    return negative(max(1, int(h * scale)), max(1, int(w * scale)), seed=params.frame or 0)


class SimulatedBackend:
    def list_devices(self) -> list[ScannerDevice]:
        return list(DEVICES)

    def refresh_devices(self) -> list[ScannerDevice]:
        return self.list_devices()

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[..., None],
        cancel: threading.Event,
    ) -> ScanResult:
        for step in range(_PROGRESS_STEPS):
            if cancel.wait(_STEP_S):
                raise RuntimeError("Scan cancelled")
            progress(step / _PROGRESS_STEPS)
        progress(1.0)
        rgb, ir = _frame(params, params.dpi)
        if params.depth == 8:
            rgb = (rgb >> 8).astype(np.uint8)
        return ScanResult(
            rgb=rgb,
            ir=ir if params.capture_ir else None,
            dpi=params.dpi,
            device_model=_BY_ID[device_id].model,
        )

    def open_session(self, device_id: str) -> "_Session":
        return _Session(self, device_id)

    def eject(self, device_id: str) -> bool:
        return _BY_ID[device_id].capabilities.can_eject

    def detect_frames(self, device_id: str, *, film_format: str | None = None) -> int:
        return _ROLL_FRAMES

    def meter(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[..., None],
        cancel: threading.Event,
    ) -> dict[str, int]:
        return {"red": 1200, "green": 1800, "blue": 2600}

    def open_roll(
        self,
        device: ScannerDevice,
        *,
        dpi: int,
        film_format: str | None = None,
        film_type: str = "negative",
    ) -> "_Roll":
        return _Roll()


class _Session:
    def __init__(self, backend: SimulatedBackend, device_id: str) -> None:
        self._backend = backend
        self.device_id = device_id

    def scan(self, params: ScanParams, progress: Callable[..., None], cancel: threading.Event) -> ScanResult:
        return self._backend.scan(self.device_id, params, progress, cancel)

    def eject(self) -> bool:
        return self._backend.eject(self.device_id)

    def close(self) -> None:
        pass

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _Roll:
    """The last frame's boundary is inferred, so the strip dialog asks to confirm it."""

    slot_count = _ROLL_FRAMES
    offset_range = (-0.5, 0.5)
    supports_single_slot_preview = False

    def __init__(self) -> None:
        self._offsets: dict[int, float] = {}
        self._approved: set[int] = set()

    def preview(self, slots: Iterable[int], *, cancel: threading.Event) -> Iterator[RollPreview]:
        for slot in slots:
            if cancel.wait(_STEP_S):
                return
            rgb, _ir = negative(160, 240, seed=slot)
            yield RollPreview(
                slot=slot,
                rgb=rgb,
                offset=self._offsets.get(slot, 0.0),
                needs_approval=slot == _ROLL_FRAMES and slot not in self._approved,
            )

    def set_offset(self, slot: int, offset: float) -> None:
        self._offsets[slot] = max(self.offset_range[0], min(offset, self.offset_range[1]))

    def approve(self, slot: int) -> None:
        self._approved.add(slot)

    def close(self) -> None:
        pass
