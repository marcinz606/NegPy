import numpy as np
import pytest

import negpy_core


def percentiles(values, qs):
    return negpy_core.percentiles(np.ascontiguousarray(values).ravel(), [float(q) for q in qs])


_RNG = np.random.default_rng(3)
_ARRAYS = {
    "len1": _RNG.random(1, dtype=np.float32),
    "len2": _RNG.random(2, dtype=np.float32),
    "len7": _RNG.random(7, dtype=np.float32),
    "len8": _RNG.random(8, dtype=np.float32),
    "len10001": _RNG.random(10_001, dtype=np.float32),
    "repeats": np.round(_RNG.random(5_000, dtype=np.float32), 2),
    "image": _RNG.random((37, 53), dtype=np.float32) ** 2.2,
    "large_signed": (_RNG.standard_normal(1_000_003) * 3).astype(np.float32),
    "large_repeats": np.round(_RNG.random(200_000, dtype=np.float32) * 8) / 8 - 0.5,
    "strided": (_RNG.random((300, 400), dtype=np.float32))[::2, 1::3],
}


@pytest.mark.parametrize("name", sorted(_ARRAYS))
@pytest.mark.parametrize("qs", [(1, 99), (80, 20, 60), (50,), (0, 100), (99.5, 0.5)])
def test_percentiles_match_numpy_bit_for_bit(name, qs):
    a = _ARRAYS[name]
    got = percentiles(a, qs)
    for value, q in zip(got, qs):
        want = np.percentile(a, q)
        assert np.float32(value) == want
        assert want.dtype == np.float32


def test_percentiles_leave_the_input_unchanged():
    a = _ARRAYS["len10001"].copy()
    percentiles(a, (1, 99))
    np.testing.assert_array_equal(a, _ARRAYS["len10001"])


def test_percentiles_reject_other_dtypes():
    with pytest.raises(TypeError):
        percentiles(np.arange(10, dtype=np.float64), (50,))


def test_percentiles_empty_raises():
    with pytest.raises(IndexError):
        np.percentile(np.array([], dtype=np.float32), 50)
    with pytest.raises(IndexError):
        percentiles(np.array([], dtype=np.float32), (50,))
