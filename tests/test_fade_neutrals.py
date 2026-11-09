"""Fade Estimate's gray readers: the gray finder, the tone bands, and the Gray Mix slider between them."""

import numpy as np
import pytest

from fade_crossover_scene import estimate_fade, CROSSOVER, build_scene, estimate_grid, scan_image
from negpy.features.process.fade import (
    FadeEstimate,
    blend_gray_readings,
    estimate_fade_with,
    gray_readings,
    gray_refs,
    tone_band_refs,
    tone_bands_only,
    neutral_axis_from_grid,
    toward_no_change,
)
from negpy.features.process.models import ProcessConfig


@pytest.fixture(scope="module")
def grid():
    return estimate_grid(scan_image(CROSSOVER, build_scene(192, 144)))


def test_mix_ends_are_the_two_readers_exactly(grid):
    assert gray_refs(grid, 0.0) == neutral_axis_from_grid(grid)
    assert gray_refs(grid, 1.0) == tone_band_refs(grid)


def test_mix_moves_each_ref_in_log(grid):
    ours, theirs = neutral_axis_from_grid(grid)[0], tone_band_refs(grid)[0]
    half, reason = gray_refs(grid, 0.5)
    assert reason == ""
    for k in (0, 1):
        assert np.allclose(half[k], 0.5 * (np.asarray(ours[k]) + np.asarray(theirs[k])))


def test_mix_moves_the_estimate_between_the_two(grid):
    ends = [estimate_fade(grid, mix=m) for m in (0.0, 0.5, 1.0)]
    lo, hi = sorted((ends[0].blue, ends[2].blue))
    assert ends[0].blue != ends[2].blue and lo < ends[1].blue < hi


def test_the_mix_defaults_to_negpy_s_reader():
    assert ProcessConfig().fade_neutrals_mix == 0.0


@pytest.mark.parametrize("mix", [0.0, 0.3, 1.0])
def test_blending_both_readings_is_gray_refs(grid, mix):
    assert blend_gray_readings(gray_readings(grid), mix) == gray_refs(grid, mix)


def test_a_live_estimate_keeps_the_survivals_and_leaves_out_the_crossover(grid):
    reader = lambda g: gray_refs(g, 0.5)  # noqa: E731
    full, live = estimate_fade_with(grid, reader), estimate_fade_with(grid, reader, crossover=False)
    assert (live.red, live.green, live.blue) == (full.red, full.green, full.blue)
    assert live.crossover is None


def test_a_blend_uses_the_second_readers_grays_where_the_first_finds_none() -> None:
    refs = ((0.5, 0.6, 0.7), (1.0, 1.1, 1.2))
    readings = ((None, "none"), (refs, ""))
    assert tone_bands_only(readings)
    for mix in (0.0, 0.5, 1.0):
        assert blend_gray_readings(readings, mix) == (refs, "")
    assert blend_gray_readings(((refs, ""), (None, "nc")), 0.5) == (refs, "")
    assert not tone_bands_only(((refs, ""), (refs, "")))
    assert blend_gray_readings(((None, "none"), (None, "nc")), 0.5) == (None, "none")


def test_toward_no_change_runs_from_no_survival_change_to_the_estimate() -> None:
    est = FadeEstimate(0.5, 1.2, 2.0, "", crossover=(0.1,))
    assert toward_no_change(est, 1.0) == est
    zero = toward_no_change(est, 0.0)
    assert (zero.red, zero.green, zero.blue, zero.crossover) == (1.0, 1.0, 1.0, None)
    half = toward_no_change(est, 0.5)
    assert half.blue == pytest.approx(2.0**0.5) and half.red == pytest.approx(1.0 / 2.0**0.5)
    none = FadeEstimate(None, 1.0, 1.0, "No grays.")
    assert toward_no_change(none, 0.5) == none


def test_pick_neutrals_takes_the_candidate_with_the_least_neutral_spread():
    from fade_crossover_scene import CROSSOVER, build_scene, estimate_grid, scan_image

    from negpy.features.process import fade

    grid = estimate_grid(scan_image(CROSSOVER, build_scene(96, 72)))
    readings = fade.gray_readings(grid)
    each = [fade.estimate_fade_with(grid, lambda _, m=m: fade.blend_gray_readings(readings, m)) for m in fade.NEUTRALS_CANDIDATES]
    best = min(range(len(each)), key=lambda i: fade.neutral_spread(each[i].crossover))
    mix, est = fade.pick_neutrals(grid, readings)
    assert (mix, est) == (fade.NEUTRALS_CANDIDATES[best], each[best])


def test_pick_neutrals_without_grays_picks_nothing():
    from negpy.features.process import fade

    assert fade.pick_neutrals(np.zeros((8, 8, 3), np.float32), ((None, "No grays."), (None, "No grays."))) is None
