"""Same-pixel colour floor refs: the colour bounds pass must read the dense
(highlight) endpoint from one shared, chroma-gated pixel set instead of
independent per-channel percentiles — coloured highlight content (sunset sky,
red car) is scene content, not film cast. The thin end stays percentile-based:
nothing on real film is thinner than base, so per-channel ceils are physically
anchored. When the dense end holds no trustworthy neutrals, the pass falls back
to the percentile method bit-exact.

Images stay under the analysis grid so the block-median prefilter is a no-op and
in-test replications of the percentile pass are exact.
"""

import numpy as np

from negpy.features.exposure.normalization import (
    _sample_log_bounds,
    analyze_log_exposure_bounds_from_log,
)
from negpy.features.process.models import ProcessMode

_H, _W = 400, 300
_BASE = (-0.10, -0.22, -0.32)
_GAMMA = 0.7


def _ramp() -> np.ndarray:
    E = np.linspace(0.0, 1.0, _H, dtype=np.float32)
    log = np.empty((_H, _W, 3), np.float32)
    for ch in range(3):
        log[:, :, ch] = (_BASE[ch] - _GAMMA * E)[:, None]
    return log


def _old_colour_recombined(img_log: np.ndarray, mode: str = ProcessMode.C41, color_clip: float = 1.0):
    """The pre-change recombination (independent per-channel percentiles)."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    floors, ceils = _sample_log_bounds(img_log, 0.0, float(EXPOSURE_CONSTANTS["base_luma_clip"]), mode, True)
    c_floors, c_ceils = _sample_log_bounds(img_log, color_clip, 0.0, mode, True)
    mean_lf, mean_lc = sum(floors) / 3.0, sum(ceils) / 3.0
    mean_cf, mean_cc = sorted(c_floors)[1], sorted(c_ceils)[1]
    return (
        [mean_lf + (c_floors[ch] - mean_cf) for ch in range(3)],
        [mean_lc + (c_ceils[ch] - mean_cc) for ch in range(3)],
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

    old_floors, _ = _old_colour_recombined(log)
    old_err = np.abs(_dev(old_floors) - truth).max()
    assert old_err > 0.10, "fixture must actually break the percentile pass"


def test_neutral_frame_matches_percentile_pass():
    # With neutral extremes both estimators must agree (no invented tint).
    log = _ramp()
    bounds = analyze_log_exposure_bounds_from_log(log, color_clip=1.0)
    old_floors, _ = _old_colour_recombined(log)
    np.testing.assert_allclose(_dev(bounds.floors), _dev(old_floors), atol=5e-3)


def test_no_neutral_dense_end_falls_back_bit_exact():
    # Dense end made ONLY of strongly split R/B content: the chroma cap rejects
    # the band and the whole colour pass reproduces the percentile method.
    log = _ramp()
    dense = slice(int(0.90 * _H), _H)
    log[dense, 0::2, 0] -= 0.65
    log[dense, 0::2, 2] += 0.65
    log[dense, 1::2, 0] += 0.65
    log[dense, 1::2, 2] -= 0.65

    bounds = analyze_log_exposure_bounds_from_log(log, color_clip=1.0)
    old_floors, old_ceils = _old_colour_recombined(log)
    np.testing.assert_allclose(bounds.floors, old_floors, atol=1e-9)
    np.testing.assert_allclose(bounds.ceils, old_ceils, atol=1e-9)


def test_e6_gated_to_percentile_pass():
    log = _ramp()
    block = slice(0, int(0.08 * _W))
    log[:, block, 0] = _BASE[0] - 0.95
    bounds = analyze_log_exposure_bounds_from_log(log, process_mode=ProcessMode.E6, e6_normalize=True, color_clip=1.0)
    old_floors, old_ceils = _old_colour_recombined(log, mode=ProcessMode.E6)
    np.testing.assert_allclose(bounds.floors, old_floors, atol=1e-9)
    np.testing.assert_allclose(bounds.ceils, old_ceils, atol=1e-9)


def test_thin_end_stays_percentile_based():
    # Ceiling refs must be untouched by the same-pixel change (base is anchored).
    log = _ramp()
    block = slice(0, int(0.08 * _W))
    log[:, block, 0] = _BASE[0] - 0.95
    bounds = analyze_log_exposure_bounds_from_log(log, color_clip=1.0)
    _, old_ceils = _old_colour_recombined(log)
    np.testing.assert_allclose(bounds.ceils, old_ceils, atol=1e-9)
