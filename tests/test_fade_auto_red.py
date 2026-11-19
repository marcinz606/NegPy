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
    expected = {"fade_ratio_r": 0.75, "fade_density_shift": round(-gain, 4), "fade_red_trim": 1.0}
    assert auto_red(_scaled(points, 1.0), 1.0) == (expected, "")
    # A NumPy scalar in the config makes the card's edited count a NumPy bool, which Qt refuses.
    assert all(type(v) is float for v in auto_red(_scaled(points, 1.0), 1.0)[0].values())
    assert auto_red(_scaled(points, 1.0), 1.0, shift=0.1)[0]["fade_density_shift"] == pytest.approx(0.1 - gain, abs=1e-4)


def test_a_hand_red_is_a_trim_on_the_fit_that_each_fit_applies_again() -> None:
    from negpy.features.process.fade_auto_red import auto_red

    points = tuple(np.linspace(1.0, 2.2, len(RANGE_PCT)))
    hand, _ = auto_red(_scaled(points, 1.0), 1.0, target=0.6)
    assert hand["fade_ratio_r"] == 0.6 and hand["fade_red_trim"] == pytest.approx(0.8, abs=1e-4)
    assert hand["fade_density_shift"] == pytest.approx(-(1.6 / 0.6 - 1.6), abs=1e-4)
    again, _ = auto_red(_scaled(points, 1.0), 1.0, trim=hand["fade_red_trim"])
    assert again["fade_ratio_r"] == pytest.approx(0.6, abs=1e-3) and again["fade_red_trim"] == hand["fade_red_trim"]


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


def test_a_hand_red_below_the_fit_s_floor_and_a_trim_above_one_hold_to_the_bounds() -> None:
    from negpy.features.process.fade_auto_red import AUTO_RED_MIN, auto_red

    points = tuple(np.linspace(1.0, 1.3, len(RANGE_PCT)))
    hand, _ = auto_red(_scaled(points, 1.0), 1.0, target=0.3)
    assert hand["fade_ratio_r"] == 0.3 and hand["fade_red_trim"] == pytest.approx(0.3 / AUTO_RED_MIN, abs=1e-4)
    wide = tuple(np.linspace(0.3, 2.3, len(RANGE_PCT)))
    high, _ = auto_red(_scaled(wide, 0.9), 0.9, trim=1.5)
    assert high["fade_ratio_r"] == 1.0 and high["fade_red_trim"] == 1.5


def test_the_white_is_read_once_per_grid() -> None:
    from unittest.mock import patch

    from negpy.features.process import fade
    from negpy.features.process.fade_auto_red import grid_white

    a, b = np.full((8, 8, 3), -0.5, np.float32), np.full((8, 8, 3), -0.4, np.float32)
    with patch.object(fade, "measure_white", wraps=fade.measure_white) as measured:
        grid_white(a)
        grid_white(a)
        grid_white(b)
    assert measured.call_count == 2
