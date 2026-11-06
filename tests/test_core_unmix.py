"""The log-density unmix (`unmix_log_image`): a NumPy oracle in its summation order, the
`np.einsum` it replaced, and a hash pin."""

import negpy_core
import numpy as np
import pytest
from core_pin import digest, linux_pin

from negpy.features.exposure.normalization import unmix_log_image

_M = np.array([[1.3, -0.2, -0.1], [-0.05, 1.4, -0.35], [0.0, -0.3, 1.3]], dtype=np.float64)


def _log(seed, shape=(301, 457)):
    return -np.random.default_rng(seed).uniform(0.0, 4.0, (*shape, 3)).astype(np.float32)


def _left_to_right(img, m):
    m = m.astype(np.float32)
    return np.stack([(img[..., 0] * m[k, 0] + img[..., 1] * m[k, 1]) + img[..., 2] * m[k, 2] for k in range(3)], axis=-1)


@pytest.mark.parametrize("shape", [(1, 1), (7, 5), (301, 457)])
def test_matches_float32_sums_in_channel_order(shape):
    img = _log(shape[0], shape)
    np.testing.assert_array_equal(unmix_log_image(img, _M), _left_to_right(img, _M))


@linux_pin
def test_matches_the_einsum_it_replaced():
    img = _log(2)
    expected = np.einsum("hwc,kc->hwk", img, _M.astype(np.float32))
    np.testing.assert_array_equal(unmix_log_image(img, _M), expected)


def test_keeps_negative_densities_and_takes_a_strided_view():
    img = _log(3)
    view = img[::2, 1::3]
    out = unmix_log_image(view, np.eye(3))
    np.testing.assert_array_equal(out, view)
    assert out.min() < 0.0


def test_no_matrix_is_the_identity():
    img = _log(4)
    assert unmix_log_image(img, None) is img


def test_rejects_a_non_rgb_image():
    with pytest.raises(ValueError):
        negpy_core.unmix_log(np.zeros((4, 4, 2), np.float32), np.eye(3, dtype=np.float32))


@linux_pin
def test_unmix_log_is_pinned():
    img = _log(5)
    assert (
        digest(unmix_log_image(img, _M), unmix_log_image(img[:, ::-1], _M.T))
        == "caa22ec667cea3f0c0c21fb28547fc1d4c8bac5ffffc405eefebd608899818ee"
    )


_ANCHOR = (-0.4, -0.7, -0.55)


def _anchored_oracle(img, m, anchor):
    a = np.asarray(anchor, np.float32)
    excess = np.maximum(np.min(img - a, axis=-1, keepdims=True), np.float32(0.0))
    q = np.minimum(img, a)
    m = m.astype(np.float32)
    return np.stack([((q[..., 0] * m[k, 0] + q[..., 1] * m[k, 1]) + q[..., 2] * m[k, 2]) + excess[..., 0] for k in range(3)], axis=-1)


@pytest.mark.parametrize("shape", [(1, 1), (7, 5), (301, 457)])
def test_the_anchored_unmix_matches_its_oracle(shape):
    img = _log(shape[0], shape) * np.float32(0.3)
    np.testing.assert_array_equal(unmix_log_image(img, _M, _ANCHOR), _anchored_oracle(img, _M, _ANCHOR))


def test_the_anchored_unmix_is_the_unmix_below_the_anchor():
    img = _log(6) - np.float32(1.0)
    np.testing.assert_array_equal(unmix_log_image(img, _M, _ANCHOR), unmix_log_image(img, _M))


def test_light_past_the_anchor_passes_as_a_neutral():
    """A pixel above the anchor in every channel keeps that excess unscaled and equal in each channel."""
    img = np.full((1, 1, 3), 0.0, np.float32)
    out = unmix_log_image(img, _M, _ANCHOR)[0, 0]
    at_anchor = unmix_log_image(np.asarray(_ANCHOR, np.float32).reshape(1, 1, 3), _M)[0, 0]
    np.testing.assert_allclose(out - at_anchor, 0.4, atol=1e-6)


@linux_pin
def test_unmix_log_anchored_is_pinned():
    img = _log(7) * np.float32(0.3)
    assert (
        digest(unmix_log_image(img, _M, _ANCHOR), unmix_log_image(img[:, ::-1], _M.T, _ANCHOR))
        == "ca6f43af70981d41416e79f3da8a1f40278f00daa089b67ccf9021603398be2a"
    )
