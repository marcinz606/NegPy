import numpy as np

from negpy.features.exposure.logic import apply_flat_curve

_IMG = np.random.default_rng(4).random((61, 89, 3), dtype=np.float32) * 1.4 - 0.2
_M = np.array([[0.8, 0.15, 0.05], [0.1, 0.85, 0.05], [0.02, 0.08, 0.9]], np.float32)


def _numpy(img, gain, lift, offsets, m):
    val = img + np.asarray(offsets, dtype=np.float32)
    if m is not None:
        val = val @ m.T
    return np.clip(lift + gain * (1.0 - val), 0.0, 1.0).astype(np.float32)


def test_flat_curve_without_a_stock_is_numpys_arithmetic():
    out = apply_flat_curve(_IMG, 0.65, 0.10, (0.013, -0.02, 0.0))
    assert np.array_equal(out, _numpy(_IMG, 0.65, 0.10, (0.013, -0.02, 0.0), None))


def test_flat_curve_with_a_stock_matches_the_matrix_product():
    out = apply_flat_curve(_IMG, 0.65, 0.10, (0.0, 0.01, -0.01), _M)
    np.testing.assert_allclose(out, _numpy(_IMG, 0.65, 0.10, (0.0, 0.01, -0.01), _M), atol=2e-7)


def test_flat_curve_keeps_nan_and_clips():
    img = np.array([[[np.nan, -5.0, 5.0]]], np.float32)
    out = apply_flat_curve(img, 0.65, 0.10)
    assert np.isnan(out[0, 0, 0]) and out[0, 0, 1] == 1.0 and out[0, 0, 2] == 0.0
