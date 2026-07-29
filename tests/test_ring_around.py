"""Colour ring-around ladder: absolute filtration, 2cc steps out to ±4cc either way of
neutral on the magenta and yellow axes.

Absolute like the strip's ladders, which is what makes the mosaic invariant to the filtration
in force: the memo can pin M/Y, so picking a patch doesn't force a reprint.
"""

import pytest

from negpy.features.exposure.analysis import (
    RING_CC_PER_UNIT,
    RING_CC_STEP,
    RING_GRID,
    ring_cc,
    ring_cells,
    ring_nearest_cell,
    ring_overrides,
    ring_rungs,
    strip_cell_at,
)
from negpy.features.exposure.models import ExposureConfig

_ROWS, _COLS = RING_GRID
_MID = _ROWS // 2


def _cell(cells, row, col):
    return cells[row * _COLS + col]


def test_the_rungs_are_absolute_and_centred_on_neutral():
    """Absolute is what makes the mosaic invariant to the filtration in force, and so
    cacheable across a pick."""
    rungs = ring_rungs()
    assert len(rungs) == _ROWS
    assert rungs[_MID] == 0.0
    assert rungs == pytest.approx([-2 * RING_CC_STEP, -RING_CC_STEP, 0.0, RING_CC_STEP, 2 * RING_CC_STEP])
    # Symmetric, so the ring never favours one direction.
    assert rungs[0] == pytest.approx(-rungs[-1])


def test_the_grid_is_five_by_five_row_major():
    cells = ring_cells()
    assert RING_GRID == (5, 5)
    assert len(cells) == 25
    assert [(r, c) for r, c, _, _ in cells] == [(r, c) for r in range(_ROWS) for c in range(_COLS)]
    assert _cell(cells, _MID, _MID)[2:] == (0.0, 0.0)  # centre prints neutral


def test_rows_step_magenta_and_columns_step_yellow():
    cells = ring_cells()
    rungs = list(ring_rungs())
    # Within a row only yellow varies...
    for row in range(_ROWS):
        assert len({_cell(cells, row, c)[2] for c in range(_COLS)}) == 1
        assert [_cell(cells, row, c)[3] for c in range(_COLS)] == pytest.approx(rungs)
    # ...and within a column only magenta.
    for col in range(_COLS):
        assert len({_cell(cells, r, col)[3] for r in range(_ROWS)}) == 1
        assert [_cell(cells, r, col)[2] for r in range(_ROWS)] == pytest.approx(rungs)


def test_the_rungs_run_two_cc_at_a_time_out_to_four():
    # filtration_offsets documents 1.0 slider = 20cc, which sets the cc scale.
    assert RING_CC_STEP * RING_CC_PER_UNIT == 2.0
    assert [ring_cc(i) for i in range(_COLS)] == [-4.0, -2.0, 0.0, 2.0, 4.0]
    assert [f"{ring_cc(i):+g}" for i in range(_COLS)] == ["-4", "-2", "+0", "+2", "+4"]


def test_every_rung_is_inside_the_sliders_travel():
    """Outside [-1, 1] the value clamps and two rungs would print the same patch."""
    assert all(-1.0 <= m <= 1.0 for m in ring_rungs())


def test_the_accent_follows_the_filtration_in_force():
    assert ring_nearest_cell(0.0, 0.0) == (_MID, _MID)
    assert ring_nearest_cell(RING_CC_STEP, -RING_CC_STEP) == (_MID + 1, _MID - 1)
    # Beyond the ladder's reach the nearest end wins rather than wrapping.
    assert ring_nearest_cell(0.9, -0.9) == (_ROWS - 1, 0)


def test_overrides_touch_only_the_two_colour_head_fields():
    """A replace() built from these must not be able to disturb density, grade or cyan."""
    overrides = ring_overrides()
    assert len(overrides) == _ROWS * _COLS
    for override in overrides:
        assert set(override) == {"wb_magenta", "wb_yellow"}
    # And the values really do reach an ExposureConfig unchanged.
    cfg = ExposureConfig(**overrides[0])
    assert cfg.wb_cyan == 0.0 and cfg.wb_magenta == overrides[0]["wb_magenta"]


def test_ring_clicks_map_across_the_whole_grid():
    assert strip_cell_at(0.0, 0.0, RING_GRID) == (0, 0)
    assert strip_cell_at(0.5, 0.5, RING_GRID) == (_MID, _MID)
    assert strip_cell_at(0.99, 0.99, RING_GRID) == (_ROWS - 1, _COLS - 1)
    assert strip_cell_at(1.4, -0.2, RING_GRID) == (0, _COLS - 1)  # clamped, not wrapped
