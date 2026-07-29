"""Colour ring-around ladder: ±5cc on the magenta and yellow axes around the filtration in
force, so the centre patch is the print being judged.

Relative rather than absolute, which makes two properties load-bearing: the centre must be
*exactly* the input, and at a slider rail the ring legitimately prints duplicate patches.
"""

import pytest

from negpy.features.exposure.analysis import (
    RING_CC_PER_UNIT,
    RING_CC_STEP,
    RING_GRID,
    ring_cc_offset,
    ring_cells,
    ring_overrides,
    strip_cell_at,
)
from negpy.features.exposure.models import ExposureConfig


def _cell(cells, row, col):
    return cells[row * RING_GRID[1] + col]


def test_the_centre_patch_is_exactly_the_filtration_passed_in():
    """Not merely close: the centre is the print being judged, so any rounding there would
    make the ring compare against something the user isn't looking at."""
    cells = ring_cells(0.2, -0.1)
    assert _cell(cells, 1, 1) == (1, 1, 0.2, -0.1)


def test_the_grid_is_nine_cells_row_major():
    cells = ring_cells(0.0, 0.0)
    assert len(cells) == RING_GRID[0] * RING_GRID[1] == 9
    assert [(r, c) for r, c, _, _ in cells] == [(r, c) for r in range(3) for c in range(3)]


def test_rows_step_magenta_and_columns_step_yellow():
    cells = ring_cells(0.0, 0.0)
    # Within a row only yellow varies...
    for row in range(3):
        magentas = {_cell(cells, row, c)[2] for c in range(3)}
        assert len(magentas) == 1
        yellows = [_cell(cells, row, c)[3] for c in range(3)]
        assert yellows == [-RING_CC_STEP, 0.0, RING_CC_STEP]
    # ...and within a column only magenta.
    for col in range(3):
        yellows = {_cell(cells, r, col)[3] for r in range(3)}
        assert len(yellows) == 1
        magentas = [_cell(cells, r, col)[2] for r in range(3)]
        assert magentas == [-RING_CC_STEP, 0.0, RING_CC_STEP]


def test_the_step_is_the_classic_five_cc():
    # filtration_offsets documents 1.0 slider = 20cc, so this pins the step to a real
    # darkroom increment rather than an arbitrary slider fraction.
    assert RING_CC_STEP * RING_CC_PER_UNIT == 5.0
    assert [ring_cc_offset(i) for i in range(3)] == [-5.0, 0.0, 5.0]


def test_the_centre_stays_unclamped_across_the_usable_range():
    for value in (-0.75, -0.4, 0.0, 0.4, 0.75):
        cells = ring_cells(value, value)
        assert _cell(cells, 1, 1) == (1, 1, value, value)
        # Every rung is a distinct printable value away from the rails.
        assert len({m for _, _, m, _ in cells}) == 3
        assert len({y for _, _, _, y in cells}) == 3


def test_at_a_rail_the_ring_prints_duplicate_patches():
    """The head has run out of travel. That is real behaviour, not a bug — the strip forbids
    duplicates, the ring documents when they legitimately happen."""
    cells = ring_cells(1.0, 0.0)
    assert _cell(cells, 1, 1)[2] == 1.0  # centre still exact
    # The plus row clamps onto the centre row's magenta, so those patches coincide.
    assert _cell(cells, 2, 0)[2] == _cell(cells, 1, 0)[2] == 1.0
    assert _cell(cells, 0, 0)[2] == pytest.approx(1.0 - RING_CC_STEP)


def test_overrides_touch_only_the_two_colour_head_fields():
    """A replace() built from these must not be able to disturb density, grade or cyan."""
    overrides = ring_overrides(0.1, -0.2)
    assert len(overrides) == 9
    for override in overrides:
        assert set(override) == {"wb_magenta", "wb_yellow"}
    # And the values really do reach an ExposureConfig unchanged.
    cfg = ExposureConfig(**overrides[0])
    assert cfg.wb_cyan == 0.0 and cfg.wb_magenta == overrides[0]["wb_magenta"]


def test_ring_clicks_map_across_the_three_by_three_grid():
    assert strip_cell_at(0.0, 0.0, RING_GRID) == (0, 0)
    assert strip_cell_at(0.5, 0.5, RING_GRID) == (1, 1)
    assert strip_cell_at(0.99, 0.99, RING_GRID) == (2, 2)
    assert strip_cell_at(1.4, -0.2, RING_GRID) == (0, 2)  # clamped, not wrapped
