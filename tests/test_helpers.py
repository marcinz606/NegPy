import numpy as np
import pytest
from src.helpers import (
    ensure_rgb,
    get_luminance,
    calculate_file_hash,
)
from src.core.validation import ensure_image


def test_ensure_image_valid():
    arr = np.zeros((5, 5), dtype=np.float32)
    assert ensure_image(arr) is arr


def test_ensure_image_conversion():
    arr = np.zeros((5, 5), dtype=np.uint8)
    res = ensure_image(arr)
    assert res.dtype == np.float32


def test_ensure_image_invalid():
    with pytest.raises(TypeError):
        ensure_image([1, 2, 3])  # type: ignore


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
