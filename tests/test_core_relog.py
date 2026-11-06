"""The linear-light matrix on a log image (`relog_matrix_image`): a float64 oracle, the identity,
the log floor, and a hash pin."""

import negpy_core
import numpy as np
import pytest
from core_pin import digest, linux_pin

from negpy.features.exposure.normalization import relog_matrix_image

_M = np.array([[0.7152, 0.2848, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0411, 0.9589]], dtype=np.float64)


def _log(seed, shape=(301, 457)):
    return -np.random.default_rng(seed).uniform(0.0, 4.0, (*shape, 3)).astype(np.float32)


def _oracle(img, m):
    lin = 10.0 ** img.astype(np.float64)
    return np.log10(np.maximum(lin @ m.astype(np.float32).astype(np.float64).T, 1e-6))


@pytest.mark.parametrize("shape", [(1, 1), (7, 5), (301, 457)])
def test_matches_a_float64_oracle(shape):
    img = _log(shape[0], shape)
    np.testing.assert_allclose(relog_matrix_image(img, _M), _oracle(img, _M), atol=2e-6)


def test_the_identity_keeps_the_densities():
    img = _log(2)
    np.testing.assert_allclose(relog_matrix_image(img, np.eye(3)), img, atol=1e-6)


def test_a_mix_below_zero_stops_at_the_log_floor():
    img = np.full((2, 2, 3), -1.0, np.float32)
    out = relog_matrix_image(img, np.array([[1.0, -2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    np.testing.assert_allclose(out[..., 0], -6.0, atol=1e-6)


def test_no_matrix_is_the_identity():
    img = _log(4)
    assert relog_matrix_image(img, None) is img


def test_rejects_a_non_rgb_image():
    with pytest.raises(ValueError):
        negpy_core.relog_matrix(np.zeros((4, 4, 2), np.float32), np.eye(3, dtype=np.float32))


@linux_pin
def test_relog_matrix_is_pinned():
    img = _log(5)
    assert (
        digest(relog_matrix_image(img, _M), relog_matrix_image(img[:, ::-1], _M.T))
        == "7bf89a974d8dea2088c65273e29b163cdd8682559743571044c0c9b6367fca1d"
    )
