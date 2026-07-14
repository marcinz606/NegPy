from __future__ import annotations

import numpy as np
import pytest

from negpy.desktop.view.widgets.roll_thumbnail_renderer import (
    render_roll_thumbnail_rgb8,
)


def test_roll_thumbnail_renderer_rotates_and_preserves_midtones() -> None:
    ramp = np.linspace(0, 65_535, 101, dtype=np.uint16)
    raw = np.repeat(ramp[:, None, None], 7, axis=1)
    raw = np.repeat(raw, 3, axis=2)

    rendered = render_roll_thumbnail_rgb8(raw)

    assert rendered.shape == (7, 101, 3)
    assert rendered.dtype == np.uint8
    # Linear inversion keeps the middle of the density range near middle
    # gray. The former extra 1/2.2 curve put it near 186 and washed out this
    # exact diagnostic ramp.
    assert 120 <= int(rendered[3, 50, 0]) <= 135
    assert rendered.flags.c_contiguous


@pytest.mark.parametrize(
    "bad",
    (
        np.zeros((4, 5), dtype=np.uint16),
        np.zeros((4, 5, 4), dtype=np.uint16),
        np.zeros((4, 5, 3), dtype=np.uint8),
    ),
)
def test_roll_thumbnail_renderer_rejects_invalid_scanner_payloads(
    bad: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="roll thumbnail"):
        render_roll_thumbnail_rgb8(bad)
