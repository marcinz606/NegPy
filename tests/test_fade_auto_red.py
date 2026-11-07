"""Restore Range: Red Survival from a faded slide's density span."""

from dataclasses import replace

import numpy as np
import pytest

from negpy.features.process.fade_auto_red import RANGE_PCT, range_points


def test_the_points_are_the_mean_density_quantiles_light_end_first():
    rng = np.random.default_rng(0)
    grid = np.repeat(rng.uniform(0.2, 1.4, (64, 64, 1)), 3, axis=-1)
    points = range_points(grid)
    assert len(points) == len(RANGE_PCT) and points == tuple(sorted(points))
    assert points[0] == pytest.approx(0.2, abs=0.02) and points[-1] == pytest.approx(1.4, abs=0.02)
    assert range_points(np.full((16, 16, 3), 0.5)) is None


def _scaled(points: tuple[float, ...], red: float, pivot: float = 0.0):
    """The points a frame read at *red* shows at another red: density scales by red / new red about *pivot*."""
    return lambda new: tuple(pivot + (p - pivot) * red / new for p in points)


def test_auto_red_scales_the_span_onto_the_adaptive_span_and_holds_its_bounds() -> None:
    from negpy.features.process.fade_auto_red import AUTO_RED_MIN, AUTO_RED_SPAN, auto_red

    def points(span: float, light: float = 0.3) -> tuple[float, ...]:
        return tuple(np.linspace(light, light + span, len(RANGE_PCT)))

    changes, reason = auto_red(_scaled(points(1.2), 0.9), 0.9)
    assert reason == "" and changes["fade_ratio_r"] == pytest.approx(0.9 * 1.2 / AUTO_RED_SPAN, abs=0.006)
    assert auto_red(_scaled(points(0.3), 1.0), 1.0)[0]["fade_ratio_r"] == AUTO_RED_MIN
    assert auto_red(_scaled(points(2.0), 0.6), 0.6)[0]["fade_ratio_r"] == 0.75
    assert auto_red(_scaled(points(4.0), 1.0), 1.0)[0]["fade_ratio_r"] == 1.0
    assert auto_red(_scaled(points(0.01), 1.0), 1.0)[0] is None
    assert auto_red(lambda _red: None, 1.0)[0] is None


def test_auto_red_shifts_density_by_what_the_median_gains() -> None:
    """A span of 1.2 D sets red 0.75; the shift takes back what the median gains, on top of the shift the
    frame already had."""
    from negpy.features.process.fade_auto_red import auto_red

    points = tuple(np.linspace(1.0, 2.2, len(RANGE_PCT)))
    gain = 1.6 / 0.75 - 1.6
    assert auto_red(_scaled(points, 1.0), 1.0) == ({"fade_ratio_r": 0.75, "fade_density_shift": round(-gain, 4)}, "")
    assert auto_red(_scaled(points, 1.0), 1.0, shift=0.1)[0]["fade_density_shift"] == pytest.approx(0.1 - gain, abs=1e-4)


def test_auto_red_keeps_the_light_end_off_white() -> None:
    """The shift stops where the 0.5 % point reaches AUTO_RED_LIGHT, or where it was when it was lighter."""
    from negpy.features.process.fade_auto_red import AUTO_RED_LIGHT, auto_red

    points = tuple(np.linspace(0.3, 1.5, len(RANGE_PCT)))
    assert auto_red(_scaled(points, 1.0), 1.0)[0]["fade_density_shift"] == pytest.approx(AUTO_RED_LIGHT - 0.3 / 0.75, abs=1e-4)
    light = tuple(np.linspace(0.1, 1.3, len(RANGE_PCT)))
    assert auto_red(_scaled(light, 1.0), 1.0)[0]["fade_density_shift"] == pytest.approx(0.1 - 0.1 / 0.75, abs=1e-4)
    centred = tuple(np.linspace(-0.6, 0.6, len(RANGE_PCT)))
    assert auto_red(_scaled(centred, 1.0), 1.0)[0]["fade_density_shift"] == pytest.approx(0.6 / 0.75 - 0.6, abs=1e-4)


def test_the_density_shift_adds_density_to_every_channel_after_the_fade() -> None:
    from negpy.features.process.fade import fade_density
    from negpy.features.process.models import ProcessConfig, ProcessMode

    base = replace(ProcessConfig(), process_mode=ProcessMode.E6, fade_ratio_r=0.5)
    plain = fade_density(base, ProcessMode.E6).matrix
    shifted = fade_density(replace(base, fade_density_shift=0.3), ProcessMode.E6).matrix
    assert np.allclose(shifted[:, :3], plain[:, :3]) and np.allclose(shifted[:, 3], -0.3)
    assert fade_density(replace(base, process_mode=ProcessMode.C41, fade_density_shift=0.3), ProcessMode.C41).matrix is None
