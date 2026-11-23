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


@pytest.mark.parametrize("point", [(1.02, 0.5), (-0.02, 0.3), (0.4, 1.03), (0.6, -0.015)])
def test_cropped_roi_near_edge_off_frame_point_round_trips(point) -> None:
    """Points just outside a tight crop project off-frame instead of clamping onto [0, 1]."""
    grid = CoordinateMapping.create_uv_grid(
        1000,
        1500,
        0,
        2.0,
        autocrop=True,
        autocrop_params={"roi": (400, 560, 600, 840)},
    )
    rx, ry = CoordinateMapping.map_click_to_raw(*point, grid)
    assert CoordinateMapping.map_raw_to_viewport(rx, ry, grid) == pytest.approx(point, abs=1e-3)


@pytest.mark.parametrize("point", [(1.003, 0.5), (-0.003, 0.5), (0.5, 1.003), (0.5, -0.003)])
def test_near_boundary_off_frame_point_does_not_clamp_to_edge(point) -> None:
    grid = _grid()
    rx, ry = CoordinateMapping.map_click_to_raw(*point, grid)
    bx, by = CoordinateMapping.map_raw_to_viewport(rx, ry, grid)
    assert (bx, by) == pytest.approx(point, abs=1e-3)


def test_panoramic_distorted_grid_round_trips_between_rows() -> None:
    """A 5:1 stitched strip with radial distortion keeps in-frame mid-row points on the Newton path."""
    grid = CoordinateMapping.create_uv_grid(100, 500, 0, 0.0, distortion_k1=-0.08)
    for nx in (0.15, 0.5, 0.88):
        ny = 51.0 / 100.0  # gy = 50.5, halfway between rows 50 and 51
        rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, grid)
        bx, by = CoordinateMapping.map_raw_to_viewport(rx, ry, grid)
        assert (bx, by) == pytest.approx((nx, ny), abs=1e-4)


def test_bottom_right_retouch_tools_align_with_displayed_pixels() -> None:
    """paint_to_score, apply_clone_strokes, and trace_scratch align at (w - 1, h - 1) without 1-px drift."""
    from negpy.features.retouch.clone import apply_clone_strokes
    from negpy.features.retouch.logic import lines_to_score, paint_to_score, trace_scratch

    h, w = 64, 80
    grid = CoordinateMapping.create_uv_grid(h, w, 0, 0.0)

    # 1. paint_to_score at bottom-right pixel (w - 1, h - 1).
    nx_br, ny_br = (w - 0.5) / w, (h - 0.5) / h
    rx_br, ry_br = CoordinateMapping.map_click_to_raw(nx_br, ny_br, grid)
    plane = np.full((h, w, 3), 0.25, dtype=np.float32)
    score, _ = paint_to_score(plane, [([[rx_br, ry_br]], 1.0)], [])
    assert score is not None
    assert np.unravel_index(int(np.argmin(score)), score.shape) == (h - 1, w - 1)
    assert float(score[h - 1, w - 1]) < 0.2

    # 2. apply_clone_strokes copying a bright pixel from (2, 3) to (w - 2, h - 2).
    nx_src, ny_src = 2.5 / w, 3.5 / h
    nx_dst, ny_dst = (w - 1.5) / w, (h - 1.5) / h
    rx_src, ry_src = CoordinateMapping.map_click_to_raw(nx_src, ny_src, grid)
    rx_dst, ry_dst = CoordinateMapping.map_click_to_raw(nx_dst, ny_dst, grid)
    clone_img = np.full((h, w, 3), 0.1, dtype=np.float32)
    clone_img[3, 2] = 0.9
    cloned = apply_clone_strokes(
        clone_img,
        [([[rx_dst, ry_dst]], 20.0, rx_src - rx_dst, ry_src - ry_dst, 1.0, 0.0, False)],
    )
    assert float(cloned[h - 2, w - 2, 0]) > 0.8

    # 3. trace_scratch & lines_to_score near the bottom edge (row = h - 6).
    scratch_img = np.full((h, w, 3), 0.25, dtype=np.float32)
    target_row = h - 6
    scratch_img[target_row, :] = 0.85
    nx_sc, ny_sc = 0.5, (target_row + 0.5) / h
    rx_sc, ry_sc = CoordinateMapping.map_click_to_raw(nx_sc, ny_sc, grid)
    traced = trace_scratch(scratch_img, rx_sc, ry_sc, threshold=0.2)
    assert traced is not None
    tx0, ty0, tx1, ty1, _ = traced
    _, vy0 = CoordinateMapping.map_raw_to_viewport(tx0, ty0, grid)
    _, vy1 = CoordinateMapping.map_raw_to_viewport(tx1, ty1, grid)
    assert (vy0 + vy1) * 0.5 == pytest.approx((target_row + 0.5) / h, abs=1e-4)
    assert vy0 == pytest.approx((target_row + 0.5) / h, abs=5e-4)
    assert vy1 == pytest.approx((target_row + 0.5) / h, abs=5e-4)
    line_score = lines_to_score(scratch_img, [traced], threshold=0.2)
    assert line_score is not None
    assert float(line_score[target_row, w // 2]) < 0.1


def test_degenerate_one_row_or_col_grid_handles_off_frame_points() -> None:
    for h_uv, w_uv in ((1, 16), (16, 1), (1, 1)):
        grid = np.zeros((h_uv, w_uv, 2), dtype=np.float32)
        grid[..., 0] = 0.25
        grid[..., 1] = 0.75
        assert CoordinateMapping.map_click_to_raw(1.4, -0.2, grid) == pytest.approx((0.25, 0.75))
        assert CoordinateMapping.map_raw_to_viewport(1.4, -0.2, grid) == (0.5, 0.5)
