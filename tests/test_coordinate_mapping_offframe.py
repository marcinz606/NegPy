"""Viewport <-> raw mapping for points off the frame.

A dodge/burn handle can sit outside the picture. The uv grid has no sample there, so
both directions continue past the boundary at the grid rate.
"""

import numpy as np
import pytest

from negpy.services.view.coordinate_mapping import CoordinateMapping

CASES = [
    {"rotation": 0, "fine_rot": 0.0, "flip_h": False},
    {"rotation": 1, "fine_rot": 0.0, "flip_h": False},
    {"rotation": 0, "fine_rot": 0.0, "flip_h": True},
    {"rotation": 2, "fine_rot": 6.0, "flip_h": False},
]

OFF_FRAME = [(1.4, 0.5), (-0.3, 0.25), (0.5, -0.2), (1.2, 1.3)]


def _grid(rotation: int = 0, fine_rot: float = 0.0, flip_h: bool = False) -> np.ndarray:
    return CoordinateMapping.create_uv_grid(240, 320, rotation, fine_rot, flip_h=flip_h)


def test_an_inside_point_is_unchanged() -> None:
    grid = _grid()
    assert CoordinateMapping.map_click_to_raw(0.25, 0.75, grid) == pytest.approx((0.25, 0.75), abs=0.01)


def test_an_outside_point_keeps_its_distance() -> None:
    grid = _grid()
    assert CoordinateMapping.map_click_to_raw(1.5, 0.5, grid) == pytest.approx((1.5, 0.5), abs=0.01)
    assert CoordinateMapping.map_click_to_raw(-0.4, 0.5, grid) == pytest.approx((-0.4, 0.5), abs=0.01)


def test_the_boundary_has_no_step() -> None:
    """The affine model agrees with the grid, so the two paths meet at the edge."""
    grid = _grid(rotation=1)
    inside = np.array(CoordinateMapping.map_click_to_raw(0.999, 0.4, grid))
    outside = np.array(CoordinateMapping.map_click_to_raw(1.001, 0.4, grid))
    assert float(np.abs(outside - inside).max()) < 0.01


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("point", OFF_FRAME)
def test_an_off_frame_point_round_trips(case, point) -> None:
    grid = _grid(**case)
    raw = CoordinateMapping.map_click_to_raw(*point, grid)
    assert CoordinateMapping.map_raw_to_viewport(*raw, grid) == pytest.approx(point, abs=0.02)


@pytest.mark.parametrize("case", CASES)
def test_an_inside_point_still_round_trips(case) -> None:
    grid = _grid(**case)
    raw = CoordinateMapping.map_click_to_raw(0.4, 0.6, grid)
    assert CoordinateMapping.map_raw_to_viewport(*raw, grid) == pytest.approx((0.4, 0.6), abs=0.02)


def test_a_click_in_the_fine_rotation_wedge_maps_off_frame_not_to_the_corner() -> None:
    grid = _grid(fine_rot=6.0)
    rx, ry = CoordinateMapping.map_click_to_raw(0.005, 0.005, grid)
    assert (rx, ry) != pytest.approx((0.0, 0.0), abs=1e-3)
    assert rx < 0.05 or ry < 0.0 or rx < 0.0


def test_the_rotated_grid_matches_a_warp_inside_the_frame() -> None:
    import cv2

    flat = _grid()
    m = cv2.getRotationMatrix2D((320 / 2.0, 240 / 2.0), 6.0, 1.0)
    warped = cv2.warpAffine(flat, m, (320, 240), flags=cv2.INTER_LINEAR)
    solved = _grid(fine_rot=6.0)
    # warpAffine interpolates in 1/32 px steps; the solved grid is exact.
    np.testing.assert_allclose(solved[60:180, 80:240], warped[60:180, 80:240], atol=2e-4)


@pytest.mark.parametrize("case", CASES)
def test_zoomed_in_subpixel_click_has_zero_top_left_drift(case) -> None:
    """Clicking off-center inside any preview pixel round-trips to sub-thousandth-pixel accuracy
    instead of flooring to the top-left grid cell."""
    grid = CoordinateMapping.create_uv_grid(120, 160, **case)
    for nx, ny in [(0.8045, 0.7085), (0.1234, 0.8765), (0.5031, 0.5042), (0.991, 0.989)]:
        rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, grid)
        bx, by = CoordinateMapping.map_raw_to_viewport(rx, ry, grid)
        assert (bx, by) == pytest.approx((nx, ny), abs=1e-4)


def test_pixel_center_click_hits_the_exact_displayed_pixel() -> None:
    """On a tiny WxH frame, clicking anywhere inside displayed pixel (col, row) maps to raw pixel (col, row)."""
    from negpy.features.retouch.logic import exclusion_cover

    h, w = 8, 10
    grid = CoordinateMapping.create_uv_grid(h, w, 0, 0.0)
    for row in (0, 3, 7):
        for col in (0, 5, 9):
            # Click at 25%, 50%, and 80% across displayed pixel (col, row).
            for fx, fy in ((0.25, 0.25), (0.5, 0.5), (0.8, 0.8)):
                nx, ny = (col + fx) / w, (row + fy) / h
                rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, grid)
                cover = exclusion_cover([([[rx, ry]], 1.0)], (h, w))
                ys, xs = np.nonzero(cover)
                assert int(np.round(xs.mean())) == col
                assert int(np.round(ys.mean())) == row
