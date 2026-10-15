"""Same-pixel color floor refs: the color bounds pass must read the dense
(highlight) endpoint from one shared, chroma-gated pixel set instead of
independent per-channel percentiles — colored highlight content (sunset sky,
red car) is scene content, not film cast. The thin end stays percentile-based:
nothing on real film is thinner than base, so per-channel ceils are physically
anchored. When the dense end holds no trustworthy neutrals, the pass falls back
to the percentile method bit-exact.

Images stay under the analysis grid so the block-median prefilter is a no-op and
in-test replications of the percentile pass are exact.
"""

import numpy as np
import pytest

from negpy.features.exposure.normalization import (
    _sample_log_bounds,
    analyze_log_exposure_bounds_from_log,
)

_H, _W = 400, 300
_BASE = (-0.10, -0.22, -0.32)
_GAMMA = 0.7


def _ramp() -> np.ndarray:
    E = np.linspace(0.0, 1.0, _H, dtype=np.float32)
    log = np.empty((_H, _W, 3), np.float32)
    for ch in range(3):
        log[:, :, ch] = (_BASE[ch] - _GAMMA * E)[:, None]
    return log


def _percentile_recombined(img_log: np.ndarray, color_clip: float = 1.0):
    """The recombination on independent per-channel percentiles at both ends."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    floors, ceils = _sample_log_bounds(img_log, 0.0, float(EXPOSURE_CONSTANTS["base_luma_clip"]))
    c_floors, c_ceils = _sample_log_bounds(img_log, color_clip, 0.0)
    spans = [c_floors[ch] - c_ceils[ch] for ch in range(3)]
    total = sum(spans)
    reach_f = sum(floors[ch] - c_floors[ch] for ch in range(3)) / total
    reach_c = sum(ceils[ch] - c_ceils[ch] for ch in range(3)) / total
    shift = (sum(c_ceils) - 3.0 * sorted(c_ceils)[1]) / total
    return (
        [c_floors[ch] + (reach_f + shift) * spans[ch] for ch in range(3)],
        [c_ceils[ch] + (reach_c + shift) * spans[ch] for ch in range(3)],
    )


def _dev(vals) -> np.ndarray:
    v = np.asarray(vals, dtype=np.float64)
    return v - v.mean()


def test_colored_dense_content_no_longer_reads_as_cast():
    # A red content block denser in R than any neutral highlight owns red's
    # dense percentile; the luma-band same-pixel refs must ignore it.
    log = _ramp()
    block = slice(0, int(0.08 * _W))
    log[:, block, 0] = _BASE[0] - 0.95
    log[:, block, 1] = _BASE[1] - 0.20
    log[:, block, 2] = _BASE[2] - 0.20

    bounds = analyze_log_exposure_bounds_from_log(log, color_clip=1.0)
    truth = _dev(_BASE)  # equal gammas: floor deviations = base (mask) deviations
    new_err = np.abs(_dev(bounds.floors) - truth).max()
    assert new_err < 0.02, f"same-pixel floors polluted (err={new_err:.3f})"

    pct_floors, _ = _percentile_recombined(log)
    pct_err = np.abs(_dev(pct_floors) - truth).max()
    assert pct_err > 0.10, "fixture must actually break the percentile pass"


def test_neutral_frame_matches_percentile_pass():
    # With neutral extremes both estimators must agree (no invented tint).
    log = _ramp()
    bounds = analyze_log_exposure_bounds_from_log(log, color_clip=1.0)
    pct_floors, _ = _percentile_recombined(log)
    np.testing.assert_allclose(_dev(bounds.floors), _dev(pct_floors), atol=5e-3)


def test_no_neutral_dense_end_falls_back_bit_exact():
    # Dense end made ONLY of strongly split R/B content: the chroma cap rejects
    # the band and the whole color pass reproduces the percentile method.
    log = _ramp()
    dense = slice(int(0.90 * _H), _H)
    log[dense, 0::2, 0] -= 0.65
    log[dense, 0::2, 2] += 0.65
    log[dense, 1::2, 0] += 0.65
    log[dense, 1::2, 2] -= 0.65

    bounds = analyze_log_exposure_bounds_from_log(log, color_clip=1.0)
    pct_floors, pct_ceils = _percentile_recombined(log)
    np.testing.assert_allclose(bounds.floors, pct_floors, atol=1e-9)
    np.testing.assert_allclose(bounds.ceils, pct_ceils, atol=1e-9)


def test_thin_end_stays_percentile_based():
    # The ceiling color refs are the percentile ones (base is anchored); the dense refs reach
    # the ceiling only through the span that scales its luma reach.
    log = _ramp()
    block = slice(0, int(0.08 * _W))
    log[:, block, 0] = _BASE[0] - 0.95
    bounds = analyze_log_exposure_bounds_from_log(log, color_clip=1.0)
    _, pct_ceils = _percentile_recombined(log)
    np.testing.assert_allclose(bounds.ceils, pct_ceils, atol=5e-3)


def _graded(gammas: tuple) -> np.ndarray:
    """One gray scale per channel at its own density range, with speculars past it."""
    E = np.linspace(0.0, 1.0, _H, dtype=np.float32)
    log = np.empty((_H, _W, 3), np.float32)
    for ch in range(3):
        log[:, :, ch] = (_BASE[ch] - gammas[ch] * E)[:, None]
        log[-1, :8, ch] = _BASE[ch] - gammas[ch] * 1.6
    return log


def test_unequal_channel_ranges_keep_a_gray_scale_neutral():
    # A film whose channels differ in density range: every gray step must normalize to the
    # same value in all three channels.
    log = _graded((0.3, 0.65, 0.2))
    bounds = analyze_log_exposure_bounds_from_log(log, color_clip=1.0)
    norm = np.stack(
        [(log[: _H - 1, 0, ch] - bounds.floors[ch]) / (bounds.ceils[ch] - bounds.floors[ch]) for ch in range(3)],
        axis=-1,
    )
    assert np.ptp(norm, axis=1).max() < 5e-3


def test_headroom_margin_widens_every_channel_alike():
    # A negative Luma Range Clip widens each window by one fraction of its own span, and by the
    # margin on average in log units, even with one channel near flat.
    log = _graded((0.6, 0.65, 0.05))
    margin = 0.1
    at_zero = analyze_log_exposure_bounds_from_log(log, percentile_clip=0.0, color_clip=1.0)
    widened = analyze_log_exposure_bounds_from_log(log, percentile_clip=-margin, color_clip=1.0)
    w0 = np.subtract(at_zero.ceils, at_zero.floors)
    w1 = np.subtract(widened.ceils, widened.floors)
    growth = w1 / w0
    assert np.ptp(growth) < 1e-6
    assert np.mean(w1 - w0) == pytest.approx(2 * margin, abs=1e-6)
