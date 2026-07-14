"""Translate an exact roll-index origin into patched coolscan3 geometry.

The package-owned RGBI worker can write an arbitrary 4000 dpi Y origin
directly.  Conventional silver B&W must instead use the established SANE RGB
path with IR disabled.  Patched coolscan3 represents that same coordinate as a
coarse 5,959-unit frame index plus a non-negative ``subframe`` distance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from negpy.infrastructure.scanners.params import RegisteredScanGeometry
from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.quality import (
    StoppedTransportSmearAssessment,
    TiffPayloadInspection,
    assess_stopped_transport_smear,
    inspect_tiff_payload,
)


LS5000_NATIVE_DPI = 4_000
LS5000_FRAME_PITCH_UNITS = 5_959
LS5000_FULL_WINDOW_ROWS = 5_959
LS5000_FINE_WIDTH = 3_946
LS5000_MAX_DRIVER_FRAME = 40
_MM_PER_NATIVE_UNIT = 25.4 / LS5000_NATIVE_DPI


class NativeOriginLike(Protocol):
    native_origin: int


class SaneRgbGeometryError(ValueError):
    """The exact preview origin cannot be represented by patched coolscan3."""


class SaneRgbQualityError(RuntimeError):
    """A B&W archival result did not satisfy the RGB-only scan contract."""


@dataclass(frozen=True)
class SaneRgbBinding:
    """Auditable decomposition of one exact 0x8e transport origin."""

    geometry: RegisteredScanGeometry
    native_origin: int
    frame_base_native_origin: int
    subframe_native_units: int
    sane_fixed_word: int
    effective_subframe_native_units: int


def sane_rgb_geometry_for_origin(origin: NativeOriginLike | int) -> SaneRgbBinding:
    """Decompose a native origin into coolscan3 frame + subframe coordinates.

    ``subframe_mm`` is placed one quarter of a native unit inside the target
    unit.  SANE stores fixed-point millimetres and coolscan3 truncates back to
    device units; using the exact mathematical boundary would round just below
    the intended integer for roughly half of possible positions.
    """

    native_origin = origin if type(origin) is int else getattr(origin, "native_origin", None)
    if type(native_origin) is not int or native_origin < 0:
        raise SaneRgbGeometryError("native origin must be a non-negative integer")

    frame_index, subframe_units = divmod(native_origin, LS5000_FRAME_PITCH_UNITS)
    driver_frame = frame_index + 1
    if driver_frame > LS5000_MAX_DRIVER_FRAME:
        raise SaneRgbGeometryError(
            f"native origin {native_origin} requires driver frame {driver_frame}, outside 1..{LS5000_MAX_DRIVER_FRAME}"
        )

    # The option's inclusive maximum is exactly 5,958 device units, but its
    # 16.16 mm encoding lands microscopically below that boundary and the C
    # backend truncates it to 5,957.  Refuse the one unrepresentable remainder
    # rather than silently shift a requested full-negative scan by one pixel.
    if subframe_units == LS5000_FRAME_PITCH_UNITS - 1:
        raise SaneRgbGeometryError("native origin lands on coolscan3's unrepresentable maximum subframe unit")

    subframe_mm = (subframe_units + 0.25) * _MM_PER_NATIVE_UNIT
    sane_fixed_word = int(subframe_mm * 65_536)
    encoded_subframe_mm = sane_fixed_word / 65_536
    effective_subframe_units = int(encoded_subframe_mm / _MM_PER_NATIVE_UNIT)
    if effective_subframe_units != subframe_units:
        raise SaneRgbGeometryError("SANE fixed-point encoding cannot preserve the requested subframe unit")
    geometry = RegisteredScanGeometry(
        frame=driver_frame,
        subframe_mm=subframe_mm,
        br_y_device_px=LS5000_FULL_WINDOW_ROWS - 1,
    )
    return SaneRgbBinding(
        geometry=geometry,
        native_origin=native_origin,
        frame_base_native_origin=frame_index * LS5000_FRAME_PITCH_UNITS,
        subframe_native_units=subframe_units,
        sane_fixed_word=sane_fixed_word,
        effective_subframe_native_units=effective_subframe_units,
    )


def validate_sane_rgb_result(
    result: ScanResult,
) -> StoppedTransportSmearAssessment:
    """Fail closed unless a capture is a clean, full-size RGB-only master."""

    if not isinstance(result, ScanResult):
        raise SaneRgbQualityError("B&W capture did not return a ScanResult")
    if result.dpi != LS5000_NATIVE_DPI:
        raise SaneRgbQualityError(f"B&W capture returned {result.dpi!r} dpi; expected {LS5000_NATIVE_DPI}")
    if (
        not isinstance(result.rgb, np.ndarray)
        or result.rgb.dtype != np.uint16
        or result.rgb.shape != (LS5000_FULL_WINDOW_ROWS, LS5000_FINE_WIDTH, 3)
    ):
        shape = getattr(result.rgb, "shape", None)
        dtype = getattr(result.rgb, "dtype", None)
        raise SaneRgbQualityError(f"B&W capture must be a 5959x3946 uint16 RGB raster; got shape={shape}, dtype={dtype}")
    if result.ir is not None or result.ir_alignment is not None or result.ir_valid_mask is not None or result.split_source is not None:
        raise SaneRgbQualityError("B&W capture unexpectedly contains infrared or split-capture data")

    smear = assess_stopped_transport_smear(result.rgb, dpi=result.dpi)
    if smear.verdict != "clean":
        raise SaneRgbQualityError(f"B&W capture did not pass stopped-transport smear QC: {smear.verdict} ({smear.reason})")
    return smear


def orient_sane_rgb_for_storage(result: ScanResult) -> ScanResult:
    """Rotate a validated scanner-native B&W raster like the color path."""

    upright = np.ascontiguousarray(np.rot90(result.rgb, k=1, axes=(0, 1)))
    return replace(result, rgb=upright)


def validate_sane_rgb_tiff(path: str | Path) -> TiffPayloadInspection:
    """Verify the committed TIFF metadata without decoding it a second time."""

    inspection = inspect_tiff_payload(path)
    expected_resolution = (LS5000_NATIVE_DPI, 1)
    if (
        inspection.shape != (LS5000_FINE_WIDTH, LS5000_FULL_WINDOW_ROWS, 3)
        or inspection.dtype != "uint16"
        or inspection.page_count != 1
        or not inspection.payload_within_file
        or inspection.x_resolution != expected_resolution
        or inspection.y_resolution != expected_resolution
        or inspection.resolution_unit != "INCH"
    ):
        raise SaneRgbQualityError("committed B&W TIFF does not preserve the full 4000 dpi uint16 RGB master")
    return inspection


__all__ = [
    "LS5000_FRAME_PITCH_UNITS",
    "LS5000_FINE_WIDTH",
    "LS5000_FULL_WINDOW_ROWS",
    "LS5000_MAX_DRIVER_FRAME",
    "LS5000_NATIVE_DPI",
    "SaneRgbBinding",
    "SaneRgbGeometryError",
    "SaneRgbQualityError",
    "orient_sane_rgb_for_storage",
    "sane_rgb_geometry_for_origin",
    "validate_sane_rgb_result",
    "validate_sane_rgb_tiff",
]
