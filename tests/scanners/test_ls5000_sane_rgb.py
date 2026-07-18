from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.ls5000_sane_rgb import (
    LS5000_FINE_WIDTH,
    LS5000_FRAME_PITCH_UNITS,
    LS5000_FULL_WINDOW_ROWS,
    SaneRgbGeometryError,
    SaneRgbQualityError,
    ice_geometry_for_origin,
    orient_sane_rgb_for_storage,
    sane_rgb_geometry_for_origin,
    validate_sane_rgb_result,
    validate_sane_rgb_tiff,
)
from negpy.services.scanning.quality import TiffPayloadInspection


def _coolscan3_fixed_point_subframe_units(value_mm: float) -> int:
    """Model SANE_FIX/SANE_UNFIX followed by coolscan3's unsigned cast."""

    # python-sane 2.9.2's `_sane.c` uses the SANE_FIX macro's truncating
    # conversion here; it does not add a half-step before the integer cast.
    sane_word = int(value_mm * 65_536)
    decoded_mm = sane_word / 65_536
    return int(decoded_mm / (25.4 / 4_000))


@pytest.mark.parametrize("native_origin", [0, 6_258, 12_278, 109_060, 236_054])
def test_exact_origin_survives_sane_fixed_point_and_driver_decomposition(
    native_origin: int,
) -> None:
    binding = sane_rgb_geometry_for_origin(native_origin)

    assert binding.geometry.br_y_device_px == LS5000_FULL_WINDOW_ROWS - 1
    assert binding.frame_base_native_origin + binding.subframe_native_units == native_origin
    assert binding.effective_subframe_native_units == binding.subframe_native_units
    assert binding.sane_fixed_word == int(binding.geometry.subframe_mm * 65_536)
    assert _coolscan3_fixed_point_subframe_units(binding.geometry.subframe_mm) == binding.subframe_native_units
    reconstructed = (binding.geometry.frame - 1) * LS5000_FRAME_PITCH_UNITS + _coolscan3_fixed_point_subframe_units(
        binding.geometry.subframe_mm
    )
    assert reconstructed == native_origin


def test_coordinate_decomposition_can_cross_a_nominal_slot_boundary() -> None:
    native_origin = 2 * LS5000_FRAME_PITCH_UNITS + 17
    binding = sane_rgb_geometry_for_origin(native_origin)

    assert binding.geometry.frame == 3
    assert binding.subframe_native_units == 17


def test_tail_origin_outside_driver_capacity_refuses() -> None:
    with pytest.raises(SaneRgbGeometryError, match="outside 1..40"):
        sane_rgb_geometry_for_origin(40 * LS5000_FRAME_PITCH_UNITS)


def test_unrepresentable_maximum_subframe_refuses_instead_of_shifting() -> None:
    with pytest.raises(SaneRgbGeometryError, match="unrepresentable maximum"):
        sane_rgb_geometry_for_origin(LS5000_FRAME_PITCH_UNITS - 1)


@pytest.mark.parametrize("value", [-1, True, 1.5, object()])
def test_invalid_native_origins_refuse(value: object) -> None:
    with pytest.raises(SaneRgbGeometryError, match="non-negative integer"):
        sane_rgb_geometry_for_origin(value)  # type: ignore[arg-type]


def _full_rgb() -> np.ndarray:
    rows = np.random.default_rng(20260714).integers(
        1_000,
        64_000,
        size=(LS5000_FULL_WINDOW_ROWS, 1, 3),
        dtype=np.uint16,
    )
    return np.broadcast_to(
        rows,
        (LS5000_FULL_WINDOW_ROWS, LS5000_FINE_WIDTH, 3),
    )


def _full_result() -> ScanResult:
    return ScanResult(
        rgb=_full_rgb(),
        ir=None,
        dpi=4_000,
        device_model="Nikon LS-5000 ED",
    )


def test_bw_result_contract_accepts_clean_full_size_rgb_only() -> None:
    assessment = validate_sane_rgb_result(_full_result())

    assert assessment.verdict == "clean"


def test_bw_result_contract_refuses_wrong_shape_or_any_ir() -> None:
    result = _full_result()

    with pytest.raises(SaneRgbQualityError, match="5959x3946"):
        validate_sane_rgb_result(replace(result, rgb=np.zeros((10, 12, 3), dtype=np.uint16)))
    with pytest.raises(SaneRgbQualityError, match="infrared"):
        validate_sane_rgb_result(
            replace(
                result,
                ir=np.zeros(
                    (LS5000_FULL_WINDOW_ROWS, LS5000_FINE_WIDTH),
                    dtype=np.uint16,
                ),
            )
        )


def test_bw_result_contract_refuses_indeterminate_uniform_colored_tail() -> None:
    rows = np.array(_full_rgb()[:, :1, :], copy=True)
    rows[-100:] = np.array([[[2_000, 30_000, 60_000]]], dtype=np.uint16)
    smeared = np.broadcast_to(
        rows,
        (LS5000_FULL_WINDOW_ROWS, LS5000_FINE_WIDTH, 3),
    )

    with pytest.raises(SaneRgbQualityError, match="smear QC: indeterminate"):
        validate_sane_rgb_result(replace(_full_result(), rgb=smeared))


def test_bw_storage_orientation_matches_single_pass_color_output() -> None:
    native = np.arange(4 * 3 * 3, dtype=np.uint16).reshape(4, 3, 3)
    result = ScanResult(
        rgb=native,
        ir=None,
        dpi=4_000,
        device_model="Nikon LS-5000 ED",
    )

    stored = orient_sane_rgb_for_storage(result)

    np.testing.assert_array_equal(
        stored.rgb,
        np.rot90(native, k=1, axes=(0, 1)),
    )
    assert stored.rgb.flags.c_contiguous
    assert stored.ir is None
    assert stored.dpi == result.dpi
    assert stored.device_model == result.device_model


def test_bw_tiff_contract_requires_full_archival_metadata(monkeypatch) -> None:
    valid = TiffPayloadInspection(
        sha256="a" * 64,
        byte_length=123,
        shape=(LS5000_FINE_WIDTH, LS5000_FULL_WINDOW_ROWS, 3),
        dtype="uint16",
        page_count=1,
        x_resolution=(4_000, 1),
        y_resolution=(4_000, 1),
        resolution_unit="INCH",
        payload_within_file=True,
    )
    monkeypatch.setattr(
        "negpy.services.scanning.ls5000_sane_rgb.inspect_tiff_payload",
        lambda _path: valid,
    )
    assert validate_sane_rgb_tiff("frame.tif") is valid

    monkeypatch.setattr(
        "negpy.services.scanning.ls5000_sane_rgb.inspect_tiff_payload",
        lambda _path: replace(valid, x_resolution=(1, 1)),
    )
    with pytest.raises(SaneRgbQualityError, match="committed B&W TIFF"):
        validate_sane_rgb_tiff("frame.tif")


def test_ice_geometry_binds_reviewed_slot_to_derived_driver_frame() -> None:
    native_origin = 2 * LS5000_FRAME_PITCH_UNITS + 17

    binding = ice_geometry_for_origin(native_origin, slot_id=3)

    assert binding.geometry.frame == 3
    assert binding.native_origin == native_origin


def test_ice_geometry_refuses_an_origin_that_crossed_a_slot_boundary() -> None:
    # A large film-spacing offset can move a reviewed slot's origin past one
    # full frame pitch; the derived driver frame then addresses a different
    # physical slot and the capture must refuse rather than shift.
    crossed_origin = 3 * LS5000_FRAME_PITCH_UNITS + 5

    with pytest.raises(SaneRgbGeometryError, match="not reviewed slot 3"):
        ice_geometry_for_origin(crossed_origin, slot_id=3)


@pytest.mark.parametrize("slot_id", [0, -1, 1.5, "3"])
def test_ice_geometry_refuses_invalid_slot_ids(slot_id: object) -> None:
    with pytest.raises(SaneRgbGeometryError, match="slot_id"):
        ice_geometry_for_origin(17, slot_id=slot_id)  # type: ignore[arg-type]
