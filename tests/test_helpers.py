import numpy as np
import pytest
from src.helpers import (
    ensure_rgb,
    get_luminance,
    ensure_array,
    calculate_file_hash,
    transform_point,
)


def test_ensure_array_valid():
    arr = np.zeros((5, 5))
    assert ensure_array(arr) is arr


def test_ensure_array_invalid():
    with pytest.raises(TypeError):
        ensure_array([1, 2, 3])  # type: ignore


def test_ensure_rgb_2d():
    img_2d = np.zeros((10, 10))
    img_rgb = ensure_rgb(img_2d)
    assert img_rgb.shape == (10, 10, 3)


def test_get_luminance_rgb():
    img = np.zeros((1, 1, 3))
    img[0, 0] = [1.0, 1.0, 1.0]
    assert np.isclose(get_luminance(img)[0, 0], 1.0)


def test_calculate_file_hash(tmp_path):
    # Create a dummy file
    d = tmp_path / "test.raw"
    content = b"darkroom" * 1000
    d.write_bytes(content)

    h1 = calculate_file_hash(str(d))
    h2 = calculate_file_hash(str(d))
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 length


def test_transform_point_rotations():
    # Mock params object with .get method
    class Params:
        def __init__(self, rot):
            self.rot = rot

        def get(self, key, default):
            return self.rot if key == "rotation" else default

    # Point (0.1, 0.2)
    # 0 deg: (0.1, 0.2)
    p0 = transform_point(0.1, 0.2, Params(0), 100, 100)
    assert p0 == (0.1, 0.2)

    # 90 deg (1): (1-y, x) -> (0.8, 0.1)
    p1 = transform_point(0.1, 0.2, Params(1), 100, 100)
    assert np.allclose(p1, (0.8, 0.1))

    # 180 deg (2): (1-x, 1-y) -> (0.9, 0.8)
    p2 = transform_point(0.1, 0.2, Params(2), 100, 100)
    assert np.allclose(p2, (0.9, 0.8))

    # 270 deg (3): (y, 1-x) -> (0.2, 0.9)
    p3 = transform_point(0.1, 0.2, Params(3), 100, 100)
    assert np.allclose(p3, (0.2, 0.9))
