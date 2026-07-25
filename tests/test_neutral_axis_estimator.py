"""Neutral-axis estimator: hue-uniform chroma ranking, strong-cast recovery via
the correction pass, and the composite confidence model.

Fixtures are built in normalized-val space against fixed bounds (floor -3, ceil 0,
so val = (log + 3) / 3) with tone regions sized to land inside the estimator's
highlight/mid/shadow luma bands. 256px images keep the block-median prefilter a
no-op, so band populations are exact.
"""

import numpy as np

from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.normalization import LogNegativeBounds, measure_neutral_axis_from_log

_H, _W = 258, 256
_BOUNDS = LogNegativeBounds((-3.0, -3.0, -3.0), (0.0, 0.0, 0.0))


def _val_img(highlight=0.17, mid=0.47, shadow=0.79) -> np.ndarray:
    v = np.empty((_H, _W, 3), np.float32)
    third = _H // 3
    v[:third] = highlight
    v[third : 2 * third] = mid
    v[2 * third :] = shadow
    return v


def _to_log(val_img: np.ndarray) -> np.ndarray:
    return (-3.0 + 3.0 * val_img).astype(np.float32)


def _val_of(ref: float) -> float:
    return (ref + 3.0) / 3.0


def _dev(refs: tuple[float, float, float], ch: int) -> float:
    return _val_of(refs[ch]) - _val_of(refs[1])


def test_rms_ranking_prefers_true_neutrals():
    # Mid band split: opposed R/B deviation ±d (true distance from the gray axis
    # sqrt(2)·d) vs a single-channel deviation 1.9d (true distance ~1.55d, i.e.
    # farther). max-min ranks them the other way round (2d vs 1.9d).
    d = 0.04
    v = _val_img()
    third = _H // 3
    mid_rows = slice(third, 2 * third)
    v[mid_rows, 0::2, 0] = 0.47 + d
    v[mid_rows, 0::2, 2] = 0.47 - d
    v[mid_rows, 1::2, 0] = 0.47 + 1.9 * d

    res = measure_neutral_axis_from_log(_to_log(v), _BOUNDS)
    assert res is not None
    mid_refs = res[0]
    assert abs(_dev(mid_refs, 0) - d) < 2e-3, "mid refs must come from the opposed (nearer-neutral) set"
    assert abs(_dev(mid_refs, 2) + d) < 2e-3


def test_strong_cast_recovered_by_correction_pass():
    # A uniform +0.45 red cast puts every neutral's chroma above the strict cap —
    # the single-pass estimator rejected the bands and fell back to the shadow tie.
    # The correction pass re-ranks with the provisional cast removed, so the
    # neutrals pass the (corrected) cap and the full axis is recovered.
    cast = 0.45
    cap = float(EXPOSURE_CONSTANTS["neutral_axis_chroma_cap"])
    assert np.sqrt(2.0 * cast * cast / 3.0) > cap  # fixture genuinely beyond the strict cap
    v = _val_img(highlight=0.15, mid=0.45, shadow=0.75)
    v[:, :, 0] += cast

    res = measure_neutral_axis_from_log(_to_log(v), _BOUNDS)
    assert res is not None, "correctable cast must not collapse to the shadow-tie fallback"
    mid_refs, shadow_refs, _, confidence = res
    assert abs(_dev(mid_refs, 0) - cast) < 2e-3
    assert abs(_dev(shadow_refs, 0) - cast) < 2e-3
    assert confidence > 0.8  # clean, large, consistent grey sets


def test_saturated_content_still_rejected():
    # A mid band made ONLY of strongly coloured content (beyond the first-pass cap)
    # must not fabricate an axis by "correcting" the content to grey. Blue-tinted
    # so the low luma weight keeps the content inside the mid band.
    v = _val_img()
    third = _H // 3
    v[third : 2 * third, :, 2] = 0.47 + 0.75
    assert measure_neutral_axis_from_log(_to_log(v), _BOUNDS) is None


def test_confidence_drops_with_small_neutral_set():
    big = measure_neutral_axis_from_log(_to_log(_val_img()), _BOUNDS)
    # Shrink the mid-band population to a small patch; fill the rest of that
    # region with highlight-band tones so only the patch feeds the mid refs.
    v = _val_img()
    third = _H // 3
    v[third : 2 * third] = 0.17
    v[third : third + 2, :] = 0.47  # 512 px -> ~154 selected
    small = measure_neutral_axis_from_log(_to_log(v), _BOUNDS)
    assert big is not None and small is not None
    assert small[3] < big[3] * 0.75


def test_confidence_drops_on_contradictory_bands():
    consistent = _val_img()
    consistent[:, :, 0] += 0.12
    contradictory = _val_img()
    third = _H // 3
    contradictory[third : 2 * third, :, 0] += 0.12
    contradictory[2 * third :, :, 0] -= 0.12
    res_c = measure_neutral_axis_from_log(_to_log(consistent), _BOUNDS)
    res_x = measure_neutral_axis_from_log(_to_log(contradictory), _BOUNDS)
    assert res_c is not None and res_x is not None
    assert res_x[3] < res_c[3] * 0.5


def test_pure_neutral_full_confidence_and_exact_refs():
    res = measure_neutral_axis_from_log(_to_log(_val_img()), _BOUNDS)
    assert res is not None
    mid_refs, shadow_refs, highlight_refs, confidence = res
    for refs, v in ((mid_refs, 0.47), (shadow_refs, 0.79), (highlight_refs, 0.17)):
        assert refs is not None
        for ch in range(3):
            assert abs(_val_of(refs[ch]) - v) < 1e-3
    assert confidence > 0.9
