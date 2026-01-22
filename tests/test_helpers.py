import numpy as np
import pytest
from src.kernel.image.logic import (
    ensure_rgb,
    get_luminance,
    calculate_file_hash,
    float_to_uint8,
    float_to_uint16,
    uint8_to_float32,
    uint16_to_float32,
    float_to_uint_luma,
)
from src.kernel.image.validation import ensure_image


def test_float_to_uint8() -> None:
    img = np.array([[0.0, 0.5, 1.0], [2.0, -1.0, 0.5]], dtype=np.float32)
    res = float_to_uint8(img)
    assert res.dtype == np.uint8
    assert res[0, 0] == 0
    assert res[0, 1] == 127
    assert res[0, 2] == 255
    assert res[1, 0] == 255  # Clamped
    assert res[1, 1] == 0  # Clamped


def test_float_to_uint16() -> None:
    img = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    res = float_to_uint16(img)
    assert res.dtype == np.uint16
    assert res[0, 0] == 0
    assert res[0, 1] == 32767
    assert res[0, 2] == 65535


def test_uint8_to_float32() -> None:
    img = np.array([[[0, 127, 255]]], dtype=np.uint8)
    res = uint8_to_float32(img)
    assert res.dtype == np.float32
    assert np.isclose(res[0, 0, 0], 0.0)
    assert np.isclose(res[0, 0, 1], 127 / 255.0)
    assert np.isclose(res[0, 0, 2], 1.0)


def test_uint16_to_float32() -> None:
    img = np.array([[[0, 32767, 65535]]], dtype=np.uint16)
    res = uint16_to_float32(img)
    assert res.dtype == np.float32
    assert np.isclose(res[0, 0, 0], 0.0)
    assert np.isclose(res[0, 0, 1], 32767 / 65535.0)
    assert np.isclose(res[0, 0, 2], 1.0)


def test_float_to_uint_luma() -> None:
    img = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
    res8 = float_to_uint_luma(img, bit_depth=8)
    assert res8.dtype == np.uint8
    assert res8[0, 0] == 255

    res16 = float_to_uint_luma(img, bit_depth=16)
    assert res16.dtype == np.uint16
    assert res16[0, 0] == 65535


def test_prepare_thumbnail() -> None:
    from PIL import Image
    from src.kernel.image.logic import prepare_thumbnail

    img = Image.new("RGB", (200, 100), (255, 0, 0))
    thumb = prepare_thumbnail(img, 50)
    assert thumb.size == (50, 50)
    # Background should be (14, 17, 23)
    assert thumb.getpixel((0, 0)) == (14, 17, 23)
    # Center should be red
    assert thumb.getpixel((25, 25)) == (255, 0, 0)


def test_ensure_image_valid() -> None:
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
