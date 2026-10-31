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
