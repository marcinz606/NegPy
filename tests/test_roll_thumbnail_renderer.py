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


def test_roll_thumbnail_renderer_meters_the_image_interior_not_bright_rebate() -> None:
    interior_ramp = np.linspace(20_000, 40_000, 80, dtype=np.uint16)
    raw = np.full((100, 100, 3), 65_535, dtype=np.uint16)
    raw[10:90, 10:90, :] = interior_ramp[:, None, None]

    rendered = render_roll_thumbnail_rgb8(raw)

    # The bright outer 10% models clear film rebate. It stays visible, but it
    # must not set the thumbnail's exposure. The middle of the photographed
    # area should remain near middle gray instead of being washed out.
    assert 120 <= int(rendered[50, 50, 0]) <= 135
    assert int(rendered[0, 0, 0]) == 0


def test_roll_thumbnail_renderer_accepts_a_configurable_meter_inset() -> None:
    interior_ramp = np.linspace(20_000, 40_000, 80, dtype=np.uint16)
    raw = np.full((100, 100, 3), 65_535, dtype=np.uint16)
    raw[10:90, 10:90, :] = interior_ramp[:, None, None]

    whole_negative_meter = render_roll_thumbnail_rgb8(
        raw,
        meter_inset_percent=0,
    )
    center_meter = render_roll_thumbnail_rgb8(
        raw,
        meter_inset_percent=10,
    )

    assert int(whole_negative_meter[50, 50, 0]) > int(center_meter[50, 50, 0]) + 40


@pytest.mark.parametrize(
    "bad_inset",
    (-1, 31, True, 10.5, "10", None),
)
def test_roll_thumbnail_renderer_rejects_invalid_meter_insets(
    bad_inset: object,
) -> None:
    raw = np.zeros((8, 10, 3), dtype=np.uint16)

    with pytest.raises(ValueError, match="meter inset"):
        render_roll_thumbnail_rgb8(
            raw,
            meter_inset_percent=bad_inset,  # type: ignore[arg-type]
        )


def test_roll_thumbnail_renderer_accepts_the_30_percent_upper_bound() -> None:
    raw = np.arange(20 * 20 * 3, dtype=np.uint16).reshape(20, 20, 3)

    rendered = render_roll_thumbnail_rgb8(raw, meter_inset_percent=30)

    assert rendered.shape == (20, 20, 3)


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
