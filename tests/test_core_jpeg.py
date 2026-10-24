"""negpy_core's baseline JPEG against PIL's: the same bytes without restart markers, the same
decoded pixels with them, and the same pixels through the export encoder."""

import io

import numpy as np
import pytest
from PIL import Image, ImageFile

import negpy_core
from negpy.features.metadata.resolution import Resolution
from negpy.services.export.encoders import encode_jpeg


def _pil(arr, quality=90, icc=None, dpi=None):
    kwargs = {"format": "JPEG", "quality": quality, "subsampling": 0, "optimize": True}
    if icc:
        kwargs["icc_profile"] = icc
    if dpi:
        kwargs["dpi"] = dpi
    buf = io.BytesIO()
    prev = ImageFile.MAXBLOCK
    ImageFile.MAXBLOCK = max(prev, 3 * arr.shape[0] * arr.shape[1])
    try:
        Image.fromarray(arr).save(buf, **kwargs)
    finally:
        ImageFile.MAXBLOCK = prev
    return buf.getvalue()


def _pixels(data):
    return np.asarray(Image.open(io.BytesIO(data)))


def _photo(h, w):
    """Smooth gradients with edges and grain, so every coefficient band carries signal."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(3)
    base = np.stack([128 + 90 * np.sin(x / 23 + c) * np.cos(y / 17 - c) for c in range(3)], axis=-1)
    base += (x[..., None] > w / 2) * 40 + rng.normal(0, 6, (h, w, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


CASES = {
    "photo": _photo(203, 317),
    "noise": (np.random.default_rng(0).random((37, 53, 3)) * 255).astype(np.uint8),
    "gray": _photo(121, 89)[..., 1].copy(),
    "flat": np.full((16, 24, 3), 77, np.uint8),
}


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("quality", [10, 75, 90, 100])
def test_without_restarts_the_bytes_are_pils(name, quality):
    arr = CASES[name]
    assert negpy_core.encode_jpeg(arr, quality, None, None, 0) == _pil(arr, quality)


def test_icc_chunks_and_density_are_written_as_pil_writes_them():
    arr = CASES["photo"]
    icc = bytes(range(256)) * 300
    assert negpy_core.encode_jpeg(arr, 90, icc, (300, 240), 0) == _pil(arr, icc=icc, dpi=(300, 240))


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rows", [1, 3, None])
def test_restart_markers_keep_every_pixel(name, rows):
    arr = CASES[name]
    np.testing.assert_array_equal(_pixels(negpy_core.encode_jpeg(arr, 90, None, None, rows)), _pixels(_pil(arr)))


def test_the_export_encoder_keeps_pils_pixels_and_density():
    arr = CASES["photo"]
    data = encode_jpeg(arr, icc=b"\x00" * 128, resolution=Resolution.from_dpi(300), quality=88)
    np.testing.assert_array_equal(_pixels(data), _pixels(_pil(arr, 88)))
    img = Image.open(io.BytesIO(data))
    assert img.info["dpi"] == (300, 300)
    assert img.info["icc_profile"] == b"\x00" * 128


def test_a_progressive_export_stays_on_pil():
    arr = CASES["photo"]
    assert Image.open(io.BytesIO(encode_jpeg(arr, progressive=True))).info.get("progressive")
