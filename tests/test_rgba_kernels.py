import numpy as np

from negpy.kernel.image.logic import rgb_to_rgba_into, rgba_to_rgb_into


def test_rgb_to_rgba_matches_numpy():
    rng = np.random.default_rng(1)
    src = rng.random((37, 53, 3), dtype=np.float32)
    dst = np.empty((37, 53, 4), dtype=np.float32)
    dst[:, :, 3] = 1.0
    rgb_to_rgba_into(src, dst)
    assert np.array_equal(dst[:, :, :3], src)
    assert np.all(dst[:, :, 3] == 1.0)


def test_rgba_to_rgb_crops_like_numpy():
    rng = np.random.default_rng(2)
    src = rng.random((40, 60, 4), dtype=np.float32)
    dst = np.empty((30, 45, 3), dtype=np.float32)
    rgba_to_rgb_into(src, dst, 5, 7)
    assert np.array_equal(dst, src[5:35, 7:52, :3])
