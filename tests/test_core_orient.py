import negpy_core
import numpy as np
import pytest

from negpy.kernel.image.logic import apply_exif_orientation

_NUMPY = {
    2: np.fliplr,
    3: lambda a: np.rot90(a, 2),
    4: np.flipud,
    5: lambda a: np.swapaxes(a, 0, 1),
    6: lambda a: np.rot90(a, 3),
    7: lambda a: np.rot90(np.swapaxes(a, 0, 1), 2),
    8: lambda a: np.rot90(a, 1),
}


@pytest.mark.parametrize("orientation", sorted(_NUMPY))
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((67, 131, 3), np.float32),
        ((131, 67), np.float32),
        ((70, 90, 3), np.uint8),
        ((65, 129, 3), np.uint16),
        ((33, 47), np.bool_),
        ((9, 11, 5), np.uint8),
    ],
)
def test_orientation_matches_numpy(orientation, shape, dtype):
    rng = np.random.default_rng(orientation)
    arr = (rng.random(shape) * 200).astype(dtype)
    out = apply_exif_orientation(arr, orientation)
    expected = _NUMPY[orientation](arr)
    assert out.dtype == arr.dtype and out.shape == expected.shape
    assert out.flags.c_contiguous
    assert np.array_equal(out, expected)


def test_orientation_of_a_strided_view():
    arr = np.arange(40 * 60 * 3, dtype=np.float32).reshape(40, 60, 3)[5:30, 10:50]
    assert np.array_equal(apply_exif_orientation(arr, 6), np.rot90(arr, 3))


def test_orient_rejects_bad_input():
    with pytest.raises(ValueError):
        negpy_core.orient(np.zeros((2, 2, 1), np.uint8), 9)
    with pytest.raises(ValueError):
        negpy_core.orient(np.zeros((2, 2), np.uint8), 6)


def test_identity_and_unknown_orientations_return_the_input():
    arr = np.zeros((3, 4), np.float32)
    for orientation in (None, 0, 1, 9):
        assert apply_exif_orientation(arr, orientation) is arr
