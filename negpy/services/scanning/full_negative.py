"""Lossless reconstruction of a film frame spanning adjacent feeder windows."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isclose
from typing import cast

import numpy as np

from negpy.infrastructure.scanners.params import RegisteredScanGeometry
from negpy.infrastructure.scanners.result import ScanResult


LS5000_FULL_WINDOW_ROWS = 5959


@dataclass(frozen=True)
class FullNegativeGeometry:
    """Two consecutive feeder windows and the target's phase within them."""

    logical_frame: int
    first_window_frame: int
    second_window_frame: int
    phase_row: int
    pitch_rows: int = LS5000_FULL_WINDOW_ROWS

    def __post_init__(self) -> None:
        for field in ("logical_frame", "first_window_frame", "second_window_frame", "pitch_rows"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.second_window_frame != self.first_window_frame + 1:
            raise ValueError("full-negative source windows must be consecutive")
        if type(self.phase_row) is not int or not 0 <= self.phase_row < self.pitch_rows:
            raise ValueError("phase_row must fall within the feeder pitch")


@dataclass(frozen=True)
class FullNegativeResult:
    """A full RGB frame plus same-grid IR and its validity mask."""

    rgb: np.ndarray
    ir: np.ndarray
    ir_valid_mask: np.ndarray
    geometry: FullNegativeGeometry


def full_negative_geometry(
    *,
    logical_frame: int,
    registration: RegisteredScanGeometry,
    dpi: int,
    pitch_rows: int = LS5000_FULL_WINDOW_ROWS,
) -> FullNegativeGeometry:
    """Translate approved registration into adjacent full-window geometry."""

    if type(logical_frame) is not int or logical_frame < 1:
        raise ValueError("logical_frame must be a positive integer")
    if registration.frame is None:
        raise ValueError("full-negative reconstruction requires a physical feeder frame")
    if type(dpi) is not int or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    if pitch_rows == LS5000_FULL_WINDOW_ROWS and dpi != 4000:
        raise ValueError("the 5,959-row LS-5000 pitch requires 4000 dpi")
    phase_row = floor(registration.subframe_mm * dpi / 25.4)
    if not 0 <= phase_row < pitch_rows:
        raise ValueError("registered subframe falls outside the full-window pitch")
    return FullNegativeGeometry(
        logical_frame=logical_frame,
        first_window_frame=registration.frame,
        second_window_frame=registration.frame + 1,
        phase_row=phase_row,
        pitch_rows=pitch_rows,
    )


def reconstruct_adjacent_windows(
    first: np.ndarray,
    second: np.ndarray,
    *,
    phase_row: int,
    pitch_rows: int = LS5000_FULL_WINDOW_ROWS,
) -> np.ndarray:
    """Return one full window starting ``phase_row`` rows into ``first``.

    The two inputs are consecutive, unshifted scanner windows. Their transport
    coordinates tile at ``pitch_rows``, so slicing and concatenation preserves
    every source sample without interpolation or seam blending.
    """

    first_array = np.asarray(first)
    second_array = np.asarray(second)
    if type(phase_row) is not int or not 0 <= phase_row < pitch_rows:
        raise ValueError(f"phase_row must be an integer in [0, {pitch_rows})")
    if first_array.shape[:1] != (pitch_rows,):
        raise ValueError(f"the first source must be a full {pitch_rows}-row window")
    if first_array.shape[1:] != second_array.shape[1:]:
        raise ValueError("adjacent sources must have the same non-transport shape")
    if second_array.shape[0] < phase_row:
        raise ValueError("the successor source must contain at least phase_row rows")
    if first_array.dtype != second_array.dtype:
        raise ValueError("adjacent sources must have the same dtype")

    return np.concatenate((first_array[phase_row:], second_array[:phase_row]), axis=0)


def _capture_states_match(first: ScanResult, second: ScanResult) -> bool:
    a = first.capture_state
    b = second.capture_state
    if a is None or b is None or a.focus_position != b.focus_position:
        return False
    return all(
        isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
        for left, right in (
            (a.exposure_multiplier, b.exposure_multiplier),
            (a.red_exposure_us, b.red_exposure_us),
            (a.green_exposure_us, b.green_exposure_us),
            (a.blue_exposure_us, b.blue_exposure_us),
        )
    )


def reconstruct_full_negative(
    first: ScanResult,
    second: ScanResult,
    *,
    geometry: FullNegativeGeometry,
) -> FullNegativeResult:
    """Reconstruct one archival frame from two complete adjacent sources.

    RGB is only sliced and concatenated. IR has already been registered within
    each source reservation; its mask is spliced on the same coordinates and
    invalid pixels are set to the brightest uint16 value so a dust detector
    looking for dark IR defects cannot consume a synthetic border.
    """

    pitch = geometry.pitch_rows
    expected_rows = {
        "first": {pitch},
        "second": {geometry.phase_row, pitch},
    }
    for label, source in (("first", first), ("second", second)):
        if source.dpi != 4000:
            raise ValueError(f"the {label} full-negative source must be 4000 dpi")
        if source.rgb.ndim != 3 or source.rgb.shape[2] != 3 or source.rgb.shape[0] not in expected_rows[label]:
            if label == "first":
                raise ValueError(f"the first source must be a complete {pitch}-row RGB window")
            raise ValueError(f"the second source must be an exact {geometry.phase_row}-row tail or complete {pitch}-row RGB window")
        if source.rgb.dtype != np.uint16:
            raise ValueError(f"the {label} source RGB must be uint16")
        if source.ir is None or source.ir_valid_mask is None:
            raise ValueError(f"the {label} source must include IR and its validity mask")
        if source.ir.shape != source.rgb.shape[:2] or source.ir_valid_mask.shape != source.ir.shape:
            raise ValueError(f"the {label} source RGB, IR, and validity mask must share one canvas")
        if source.ir.dtype != np.uint16 or source.ir_valid_mask.dtype != np.bool_:
            raise ValueError(f"the {label} source IR must be uint16 with a boolean validity mask")
        if source.ir_alignment is None:
            raise ValueError(f"the {label} source is missing split-capture alignment evidence")
        if source.capture_state is not None and source.capture_state.focus_position <= 0:
            raise ValueError(f"the {label} source has uncalibrated focus position {source.capture_state.focus_position}")
    if first.rgb.shape[1:] != second.rgb.shape[1:]:
        raise ValueError("adjacent full-negative sources must have equal non-transport dimensions")
    if not _capture_states_match(first, second):
        raise ValueError("adjacent full-negative sources do not share one focus/exposure state")

    rgb = reconstruct_adjacent_windows(
        first.rgb,
        second.rgb,
        phase_row=geometry.phase_row,
        pitch_rows=pitch,
    )
    first_ir = cast(np.ndarray, first.ir)
    second_ir = cast(np.ndarray, second.ir)
    first_valid = cast(np.ndarray, first.ir_valid_mask)
    second_valid = cast(np.ndarray, second.ir_valid_mask)
    ir = reconstruct_adjacent_windows(
        first_ir,
        second_ir,
        phase_row=geometry.phase_row,
        pitch_rows=pitch,
    )
    valid = reconstruct_adjacent_windows(
        first_valid,
        second_valid,
        phase_row=geometry.phase_row,
        pitch_rows=pitch,
    )
    ir = ir.copy()
    ir[~valid] = np.iinfo(ir.dtype).max
    return FullNegativeResult(
        rgb=rgb,
        ir=ir,
        ir_valid_mask=valid,
        geometry=geometry,
    )


__all__ = [
    "FullNegativeGeometry",
    "FullNegativeResult",
    "LS5000_FULL_WINDOW_ROWS",
    "full_negative_geometry",
    "reconstruct_adjacent_windows",
    "reconstruct_full_negative",
]
