"""Scanner orchestration for lossless adjacent-window full-negative capture."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np

from negpy.infrastructure.scanners.params import RegisteredScanGeometry, ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.full_negative import (
    FullNegativeGeometry,
    FullNegativeResult,
    LS5000_FULL_WINDOW_ROWS,
    full_negative_geometry,
    reconstruct_full_negative,
)
from negpy.services.scanning.quality import (
    assess_stopped_transport_smear,
    measure_scan_clipping,
    split_alignment_metrics_confident,
)
from negpy.services.scanning.roll_service import FrameRegistration


class FullNegativeScanner(Protocol):
    """The ScannerService surface needed by the two-window workflow."""

    def run_scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult: ...


@dataclass(frozen=True)
class FullNegativeCaptureResult:
    """The two scanner sources and their reconstructed full negative."""

    geometry: FullNegativeGeometry
    first_source: ScanResult
    second_source: ScanResult
    result: FullNegativeResult


def _progress_segment(
    callback: Callable[[float], None] | None,
    start: float,
    end: float,
) -> Callable[[float], None]:
    if callback is None:
        return lambda _value: None
    return lambda value: callback(start + (end - start) * value)


def _validate_recipe(recipe: ScanParams) -> None:
    if recipe.dpi != 4000:
        raise ValueError("full-negative capture requires a 4000 dpi recipe")
    if recipe.depth != 16 or not recipe.capture_ir or recipe.samples_per_scan != 4:
        raise ValueError("full-negative capture requires 16-bit RGB4x+IR")
    if recipe.area is not None:
        raise ValueError("full-negative capture requires area=None")


def _validate_rgb_quality(rgb: np.ndarray, *, dpi: int, label: str) -> None:
    # Clipping is scene- and rebate-dependent telemetry, not a hard capture
    # failure.  A cut first exposure on the reference roll legitimately has
    # about two percent clear/white area.  The roll acceptance report records
    # the fractions; transport smear remains the fail-closed image gate here.
    measure_scan_clipping(rgb)
    smear = assess_stopped_transport_smear(rgb, dpi=dpi)
    if smear.verdict == "smear":
        raise RuntimeError(f"{label} failed stopped-transport smear QC: start_row={smear.start_row}, suffix_rows={smear.suffix_rows}")


def _validate_source(
    source: ScanResult,
    *,
    expected_rows: int,
    label: str,
) -> None:
    if source.dpi != 4000:
        raise RuntimeError(f"{label} source returned {source.dpi} dpi instead of 4000")
    if source.rgb.ndim != 3 or source.rgb.shape[0] != expected_rows or source.rgb.shape[2] != 3:
        raise RuntimeError(f"{label} source did not return the expected {expected_rows}-row RGB canvas")
    if source.rgb.dtype != np.uint16:
        raise RuntimeError(f"{label} source RGB must be uint16")
    if source.capture_state is None:
        raise RuntimeError(f"{label} source is missing the actual focus/exposure state")
    if source.capture_state.focus_position <= 0:
        raise RuntimeError(f"{label} source reported uncalibrated focus position {source.capture_state.focus_position}")
    if not split_alignment_metrics_confident(
        source.ir_alignment,
        image_width=int(source.rgb.shape[1]),
    ):
        raise RuntimeError(f"{label} source split RGB/IR alignment failed QC")
    if source.ir is None or source.ir_valid_mask is None:
        raise RuntimeError(f"{label} source is missing IR or its validity mask")
    if source.ir.shape != source.rgb.shape[:2] or source.ir_valid_mask.shape != source.ir.shape:
        raise RuntimeError(f"{label} source RGB, IR, and validity mask do not share one canvas")
    if source.ir.dtype != np.uint16 or source.ir_valid_mask.dtype != np.bool_:
        raise RuntimeError(f"{label} source IR must be uint16 with a boolean validity mask")
    if not np.any(source.ir_valid_mask):
        raise RuntimeError(f"{label} source IR validity mask contains no scanner samples")
    _validate_rgb_quality(source.rgb, dpi=source.dpi, label=f"{label} source")


class FullNegativeCapturer:
    """Capture one registered negative from a full window and bounded tail."""

    def __init__(
        self,
        *,
        scanner: FullNegativeScanner,
        pitch_rows: int = LS5000_FULL_WINDOW_ROWS,
    ) -> None:
        if type(pitch_rows) is not int or pitch_rows < 1:
            raise ValueError("pitch_rows must be a positive integer")
        self._scanner = scanner
        self._pitch_rows = pitch_rows

    def capture(
        self,
        *,
        device_id: str,
        registration: FrameRegistration,
        recipe: ScanParams,
        stop: threading.Event,
        progress: Callable[[float], None] | None = None,
        on_source: Callable[[str, int, ScanResult], None] | None = None,
    ) -> FullNegativeCaptureResult:
        """Run two uninterrupted reservations, honoring stop between them."""

        if type(device_id) is not str or not device_id:
            raise ValueError("device_id must be a nonempty string")
        if registration.frame < 1:
            raise ValueError("registration frame must be positive")
        _validate_recipe(recipe)
        geometry = full_negative_geometry(
            logical_frame=registration.frame,
            registration=registration.geometry,
            dpi=recipe.dpi,
            pitch_rows=self._pitch_rows,
        )
        if geometry.phase_row == 0:
            raise ValueError("full-negative capture requires a nonzero successor-tail phase")
        if stop.is_set():
            raise RuntimeError("full-negative capture cancelled before the first transfer")

        transfer_cancel = threading.Event()
        first_params = replace(
            recipe,
            frame=geometry.first_window_frame,
            registered_geometry=RegisteredScanGeometry(
                frame=geometry.first_window_frame,
                subframe_mm=0.0,
                br_y_device_px=geometry.pitch_rows - 1,
            ),
            autofocus=True,
            auto_exposure=True,
            capture_state=None,
            preserve_full_canvas=True,
        )
        first = self._scanner.run_scan(
            device_id,
            first_params,
            _progress_segment(progress, 0.0, 0.5),
            transfer_cancel,
        )
        _validate_source(first, expected_rows=geometry.pitch_rows, label="first")
        if on_source is not None:
            on_source("first", geometry.first_window_frame, first)
            first = replace(first, split_source=None)

        if stop.is_set():
            raise RuntimeError("full-negative capture cancelled between transfers")

        second_params = replace(
            recipe,
            frame=geometry.second_window_frame,
            registered_geometry=RegisteredScanGeometry(
                frame=geometry.second_window_frame,
                subframe_mm=0.0,
                br_y_device_px=geometry.phase_row - 1,
            ),
            autofocus=False,
            auto_exposure=False,
            capture_state=first.capture_state,
            preserve_full_canvas=True,
        )
        second = self._scanner.run_scan(
            device_id,
            second_params,
            _progress_segment(progress, 0.5, 1.0),
            transfer_cancel,
        )
        _validate_source(second, expected_rows=geometry.phase_row, label="second")
        if on_source is not None:
            on_source("second", geometry.second_window_frame, second)
            second = replace(second, split_source=None)
        result = reconstruct_full_negative(first, second, geometry=geometry)
        _validate_rgb_quality(result.rgb, dpi=recipe.dpi, label="reconstructed frame")
        return FullNegativeCaptureResult(
            geometry=geometry,
            first_source=first,
            second_source=second,
            result=result,
        )


__all__ = [
    "FullNegativeCaptureResult",
    "FullNegativeCapturer",
    "FullNegativeScanner",
]
