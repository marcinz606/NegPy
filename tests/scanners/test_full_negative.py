from __future__ import annotations

import numpy as np
import pytest

from negpy.infrastructure.scanners.params import RegisteredScanGeometry, ScannerCaptureState
from negpy.infrastructure.scanners.result import ScanResult, SplitIrAlignment
from negpy.services.scanning.full_negative import (
    FullNegativeGeometry,
    full_negative_geometry,
    reconstruct_adjacent_windows,
    reconstruct_full_negative,
)


PITCH_ROWS = 5959


@pytest.mark.parametrize("phase_row", [0, 1, 1000, 5955, 5958])
def test_reconstruct_adjacent_windows_preserves_every_transport_coordinate(phase_row: int) -> None:
    first_rows = np.arange(PITCH_ROWS, dtype=np.uint16)
    second_rows = np.arange(PITCH_ROWS, 2 * PITCH_ROWS, dtype=np.uint16)
    first = np.broadcast_to(first_rows[:, None, None], (PITCH_ROWS, 3, 4)).copy()
    second = np.broadcast_to(second_rows[:, None, None], (PITCH_ROWS, 3, 4)).copy()

    reconstructed = reconstruct_adjacent_windows(first, second, phase_row=phase_row)

    assert reconstructed.shape == first.shape
    assert reconstructed.dtype == np.uint16
    np.testing.assert_array_equal(
        reconstructed[:, 0, 0],
        np.arange(phase_row, phase_row + PITCH_ROWS, dtype=np.uint16),
    )


def test_reconstruct_adjacent_windows_rejects_nonconsecutive_source_geometry() -> None:
    first = np.zeros((PITCH_ROWS, 3, 4), dtype=np.uint16)
    second = np.zeros((999, 3, 4), dtype=np.uint16)

    with pytest.raises(ValueError, match="at least phase_row rows"):
        reconstruct_adjacent_windows(first, second, phase_row=1000)


def test_reconstruct_adjacent_windows_accepts_a_bounded_successor_tail() -> None:
    phase_row = 1000
    first_rows = np.arange(PITCH_ROWS, dtype=np.uint16)
    second_rows = np.arange(PITCH_ROWS, PITCH_ROWS + phase_row, dtype=np.uint16)
    first = np.broadcast_to(first_rows[:, None], (PITCH_ROWS, 3)).copy()
    second = np.broadcast_to(second_rows[:, None], (phase_row, 3)).copy()

    reconstructed = reconstruct_adjacent_windows(first, second, phase_row=phase_row)

    np.testing.assert_array_equal(
        reconstructed[:, 0],
        np.arange(phase_row, phase_row + PITCH_ROWS, dtype=np.uint16),
    )


def test_full_negative_geometry_uses_the_registered_physical_window_pair() -> None:
    registration = RegisteredScanGeometry(frame=3, subframe_mm=6.6, br_y_device_px=5000)

    geometry = full_negative_geometry(
        logical_frame=3,
        registration=registration,
        dpi=4000,
    )

    assert geometry == FullNegativeGeometry(
        logical_frame=3,
        first_window_frame=3,
        second_window_frame=4,
        phase_row=1039,
        pitch_rows=5959,
    )


def test_full_negative_geometry_rejects_a_dpi_that_disagrees_with_native_pitch() -> None:
    registration = RegisteredScanGeometry(frame=3, subframe_mm=6.6, br_y_device_px=5000)

    with pytest.raises(ValueError, match="4000 dpi"):
        full_negative_geometry(logical_frame=3, registration=registration, dpi=1000)


def _capture_state(*, focus: int = 216) -> ScannerCaptureState:
    return ScannerCaptureState(
        focus_position=focus,
        exposure_multiplier=1.0,
        red_exposure_us=1370.0,
        green_exposure_us=1290.0,
        blue_exposure_us=1120.0,
    )


def _source(rows: np.ndarray, *, valid: np.ndarray | None = None, state: ScannerCaptureState | None = None) -> ScanResult:
    height = len(rows)
    rgb = np.broadcast_to(rows[:, None, None], (height, 3, 3)).astype(np.uint16, copy=True)
    ir = np.broadcast_to((rows + 100)[:, None], (height, 3)).astype(np.uint16, copy=True)
    return ScanResult(
        rgb=rgb,
        ir=ir,
        dpi=4000,
        device_model="Nikon LS-5000",
        ir_alignment=SplitIrAlignment(mode="identity", dx_px=0.0, dy_px=0.0),
        ir_valid_mask=np.ones((height, 3), dtype=np.bool_) if valid is None else valid,
        capture_state=_capture_state() if state is None else state,
    )


def test_reconstruct_full_negative_splices_rgb_ir_and_validity_without_resampling() -> None:
    pitch = 8
    phase = 3
    first = _source(np.arange(pitch, dtype=np.uint16))
    second_valid = np.ones((pitch, 3), dtype=np.bool_)
    second_valid[1, 1] = False
    second = _source(np.arange(pitch, 2 * pitch, dtype=np.uint16), valid=second_valid)
    geometry = FullNegativeGeometry(
        logical_frame=3,
        first_window_frame=3,
        second_window_frame=4,
        phase_row=phase,
        pitch_rows=pitch,
    )

    result = reconstruct_full_negative(first, second, geometry=geometry)

    np.testing.assert_array_equal(result.rgb[:, 0, 0], np.arange(phase, phase + pitch))
    assert result.ir_valid_mask[6, 1] is np.False_
    assert result.ir[6, 1] == np.iinfo(np.uint16).max
    assert result.rgb.shape == (pitch, 3, 3)


def test_reconstruct_full_negative_rejects_state_drift_between_windows() -> None:
    pitch = 8
    geometry = FullNegativeGeometry(3, 3, 4, 3, pitch)

    with pytest.raises(ValueError, match="focus/exposure state"):
        reconstruct_full_negative(
            _source(np.arange(pitch, dtype=np.uint16)),
            _source(np.arange(pitch, 2 * pitch, dtype=np.uint16), state=_capture_state(focus=217)),
            geometry=geometry,
        )


def test_reconstruct_full_negative_rejects_matching_uncalibrated_focus_state() -> None:
    pitch = 8
    geometry = FullNegativeGeometry(3, 3, 4, 3, pitch)
    uncalibrated = _capture_state(focus=0)

    with pytest.raises(ValueError, match="uncalibrated focus position"):
        reconstruct_full_negative(
            _source(np.arange(pitch, dtype=np.uint16), state=uncalibrated),
            _source(np.arange(pitch, 2 * pitch, dtype=np.uint16), state=uncalibrated),
            geometry=geometry,
        )


def test_reconstruct_full_negative_accepts_an_exact_bounded_successor_tail() -> None:
    pitch = 8
    geometry = FullNegativeGeometry(3, 3, 4, 3, pitch)

    result = reconstruct_full_negative(
        _source(np.arange(pitch, dtype=np.uint16)),
        _source(np.arange(pitch, pitch + 3, dtype=np.uint16)),
        geometry=geometry,
    )

    np.testing.assert_array_equal(result.rgb[:, 0, 0], np.arange(3, 11))


def test_reconstruct_full_negative_rejects_an_incomplete_successor_tail() -> None:
    pitch = 8
    geometry = FullNegativeGeometry(3, 3, 4, 3, pitch)

    with pytest.raises(ValueError, match="exact 3-row tail or complete 8-row"):
        reconstruct_full_negative(
            _source(np.arange(pitch, dtype=np.uint16)),
            _source(np.arange(2, dtype=np.uint16)),
            geometry=geometry,
        )
