import numpy as np
import pytest

from negpy.features.exposure.analysis import (
    STRIP_DENSITIES,
    STRIP_GRADES,
    strip_cell_at,
    strip_cells,
    strip_mosaic,
    strip_nearest_cell,
    strip_patch_rect,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig


def _tiles(h: int = 40, w: int = 60) -> list[np.ndarray]:
    """One flat tile per cell, each a distinct value = its index."""
    return [np.full((h, w, 3), float(i), dtype=np.float32) for i in range(len(strip_cells()))]


def test_ladder_covers_every_cell_row_major():
    cells = strip_cells()
    assert len(cells) == len(STRIP_GRADES) * len(STRIP_DENSITIES)
    assert cells[0] == (0, 0, STRIP_DENSITIES[0], STRIP_GRADES[0])
    assert cells[-1] == (len(STRIP_GRADES) - 1, len(STRIP_DENSITIES) - 1, STRIP_DENSITIES[-1], STRIP_GRADES[-1])


def test_grade_ladder_survives_the_legacy_post_init_ladder():
    # ExposureConfig.__post_init__ rewrites any grade <= 5 as 150 - 20*g (old 0-5
    # paper grades), so a ladder value at or below 5 would silently become something else.
    for grade in STRIP_GRADES:
        assert grade > 5.0
        assert ExposureConfig(grade=grade).grade == grade


def test_every_patch_comes_from_its_own_tile():
    tiles = _tiles()
    mosaic = strip_mosaic(tiles)
    h, w = mosaic.shape[:2]
    for i, (row, col, _, _) in enumerate(strip_cells()):
        x0, y0, x1, y1 = strip_patch_rect(h, w, row, col)
        assert np.all(mosaic[y0:y1, x0:x1] == float(i))


def test_patches_tile_exactly_with_no_gap_or_overlap():
    h, w = 41, 63  # deliberately not divisible by the grid
    covered = np.zeros((h, w), dtype=int)
    for row, col, _, _ in strip_cells():
        x0, y0, x1, y1 = strip_patch_rect(h, w, row, col)
        covered[y0:y1, x0:x1] += 1
    assert np.all(covered == 1)


def test_mosaic_rejects_a_short_or_mismatched_tile_set():
    with pytest.raises(ValueError):
        strip_mosaic(_tiles()[:-1])
    tiles = _tiles()
    tiles[5] = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        strip_mosaic(tiles)


def test_click_maps_to_the_patch_under_it():
    rows, cols = len(STRIP_GRADES), len(STRIP_DENSITIES)
    assert strip_cell_at(0.0, 0.0) == (0, 0)
    assert strip_cell_at(0.99, 0.99) == (rows - 1, cols - 1)
    # Dead centre of the patch at (1, 2).
    assert strip_cell_at(2.5 / cols, 1.5 / rows) == (1, 2)


def test_click_on_the_far_edge_stays_in_the_grid():
    assert strip_cell_at(1.0, 1.0) == (len(STRIP_GRADES) - 1, len(STRIP_DENSITIES) - 1)
    assert strip_cell_at(1.4, -0.2) == (0, len(STRIP_DENSITIES) - 1)


def test_current_settings_highlight_the_nearest_patch():
    assert strip_nearest_cell(STRIP_DENSITIES[2], STRIP_GRADES[1]) == (1, 2)
    # The default density is a rung of its own; the default grade sits between two and the
    # nearest wins. Either way the highlight lands where the sliders sit.
    assert ExposureConfig().density in STRIP_DENSITIES
    assert strip_nearest_cell(1.0, 115.0) == (2, STRIP_DENSITIES.index(1.0))
    assert strip_nearest_cell(-5.0, 999.0) == (len(STRIP_GRADES) - 1, 0)


def test_both_ladders_straddle_the_defaults_in_even_steps():
    defaults = ExposureConfig()
    for ladder, current in ((STRIP_DENSITIES, defaults.density), (STRIP_GRADES, defaults.grade)):
        assert len(ladder) == 6
        steps = {round(b - a, 6) for a, b in zip(ladder, ladder[1:])}
        assert len(steps) == 1 and steps.pop() > 0  # evenly spaced, ascending
        assert ladder[0] < current < ladder[-1]


def test_ladders_stay_inside_the_ranges_their_controls_accept():
    """Outside these, values clamp — two rungs would render identically and the strip
    would show duplicate patches instead of a ladder."""
    assert STRIP_DENSITIES[0] >= 0.0 and STRIP_DENSITIES[-1] <= 2.0  # Print Density slider
    assert STRIP_GRADES[0] >= float(EXPOSURE_CONSTANTS["iso_r_min"])
    assert STRIP_GRADES[-1] <= float(EXPOSURE_CONSTANTS["iso_r_max"])
