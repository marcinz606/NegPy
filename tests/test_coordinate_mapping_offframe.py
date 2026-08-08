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
