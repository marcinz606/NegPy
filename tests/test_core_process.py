"""The process statistics, chart reads and mask overlap against the NumPy code they replace."""

import numpy as np
import pytest
from core_oracle import cv2_oracle

import negpy_core
from negpy.features.local.logic import _OVERLAP_ALPHA
from negpy.features.process import chart
from negpy.features.process.capture_color import _LIGHTBOX_PCT
from negpy.kernel.image.logic import working_oetf_decode


def _scan(seed, shape=(380, 251)):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.2, (*shape, 3)).astype(np.float32) * np.array([0.9, 0.5, 0.3], dtype=np.float32)


@cv2_oracle
@pytest.mark.parametrize("matrix", [None, [[1.1, -0.05, -0.05], [-0.1, 1.2, -0.1], [0.02, -0.3, 1.28]]])
def test_lightbox_white_matches_numpy(matrix):
    img = _scan(1)
    flat = img.reshape(-1, 3)
    step = 7
    sample = flat[::step].astype(np.float64)
    if matrix is not None:
        sample = sample @ np.asarray(matrix, dtype=np.float64).T
    want = float(np.percentile(sample, _LIGHTBOX_PCT))
    assert negpy_core.lightbox_white(img, step, matrix, _LIGHTBOX_PCT) == want


def _corr(a, b):
    a = a.ravel() - float(a.mean())
    b = b.ravel() - float(b.mean())
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b))) + 1e-12
    return float(np.sum(a * b) / denom)


@pytest.mark.parametrize("seed", [2, 3])
def test_process_mode_stats_match_numpy(seed):
    img = np.clip(_scan(seed), 0.0, 1.0)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    want_corr = min(_corr(r, g), _corr(g, b), _corr(r, b))
    want = [float(np.mean(r)), float(np.mean(b))]
    want += [float(np.percentile(r, 25)), float(np.percentile(b, 25))]
    want += [float(np.percentile(c, 98)) for c in (r, g, b)]
    got_corr, got = negpy_core.process_mode_stats(img)
    assert got_corr == want_corr
    assert list(got) == want


def test_chart_patches_match_numpy():
    encoded = _scan(4) - 0.05
    corners = np.array([[20.0, 30.0], [230.0, 28.0], [232.0, 350.0], [18.0, 352.0]])
    h, w = encoded.shape[:2]
    linear = working_oetf_decode(np.clip(encoded, 0.0, 1.0))
    r = max(1, int(round(chart.sample_radius(corners))))
    want = np.empty((24, 3))
    for i, (x, y) in enumerate(chart.patch_centers(corners)):
        x0, x1 = int(np.clip(round(x) - r, 0, w - 1)), int(np.clip(round(x) + r + 1, 1, w))
        y0, y1 = int(np.clip(round(y) - r, 0, h - 1)), int(np.clip(round(y) + r + 1, 1, h))
        want[i] = np.median(linear[y0:y1, x0:x1, :3].reshape(-1, 3), axis=0)
    np.testing.assert_array_equal(chart._patch_medians(encoded, corners, float(np.float32(563.0 / 256.0))), want)
    np.testing.assert_array_equal(chart.sample_patches(linear, corners), want)


def test_alphas_overlap_matches_numpy():
    rng = np.random.default_rng(5)
    for _ in range(20):
        a = rng.uniform(0, 0.3, (96, 96)).astype(np.float32)
        b = rng.uniform(0, 0.27, (96, 96)).astype(np.float32)
        want = bool(((a > _OVERLAP_ALPHA) & (b > _OVERLAP_ALPHA)).any())
        assert negpy_core.alphas_overlap(a, b, _OVERLAP_ALPHA) == want
