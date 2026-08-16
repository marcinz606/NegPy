"""Regression: a BITMAP embedded thumb must never be read.

rawpy shapes a BITMAP thumb (h, w, 3) from libraw's hardcoded colors=3 while the
allocation holds only data_size bytes, and exposes no data_size. A grayscale thumb is
then read three times past its end, which segfaulted the app at random while a folder's
thumbnails generated. The preview now comes from the file's own TIFF preview page.
"""

import io

import numpy as np
import tifffile
from PIL import Image

import rawpy
from negpy.infrastructure.loaders.helpers import embedded_preview


class _Thumb:
    """A rawpy Thumbnail stand-in whose data raises unless a value was given."""

    def __init__(self, fmt, data=None):
        self.format = fmt
        self._data = data

    @property
    def data(self):
        if self._data is None:
            raise AssertionError("BITMAP thumb data must never be read")
        return self._data


class _Raw:
    def __init__(self, thumb=None, error=None):
        self._thumb = thumb
        self._error = error

    def extract_thumb(self):
        if self._error is not None:
            raise self._error
        return self._thumb


def _jpeg_bytes(w: int = 16, h: int = 10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def _dng_with_preview(path, preview: np.ndarray) -> str:
    """A DNG-shaped file: reduced-resolution page 0 with the full-res data in a SubIFD."""
    with tifffile.TiffWriter(path) as tw:
        tw.write(preview, subifds=1, subfiletype=1)
        tw.write(np.zeros((40, 60), dtype=np.uint16), subfiletype=0)
    return str(path)


def test_jpeg_thumb_is_used():
    img = embedded_preview(_Raw(_Thumb(rawpy.ThumbFormat.JPEG, _jpeg_bytes())), "missing.nef")
    assert img is not None
    assert img.size == (16, 10)


def test_bitmap_thumb_data_is_never_read(tmp_path):
    """No preview page to fall back on -> None, and the unsafe buffer stays untouched."""
    assert embedded_preview(_Raw(_Thumb(rawpy.ThumbFormat.BITMAP)), str(tmp_path / "gone.dng")) is None


def test_bitmap_thumb_falls_back_to_tiff_preview_page(tmp_path):
    """A grayscale preview page is broadened to RGB, at its own size."""
    path = _dng_with_preview(tmp_path / "scan.dng", np.full((10, 16), 128, dtype=np.uint8))
    img = embedded_preview(_Raw(_Thumb(rawpy.ThumbFormat.BITMAP)), path)
    assert img is not None
    assert img.size == (16, 10)
    assert np.asarray(img).shape == (10, 16, 3)


def test_16_bit_preview_page_is_scaled_down(tmp_path):
    """Scanner DNGs write page 0 as uint16; rejecting it sent a whole library through the
    half-size demosaic instead, at about a gigabyte of transients per worker."""
    path = _dng_with_preview(tmp_path / "scan16.dng", np.full((10, 16), 0x8000, dtype=np.uint16))
    img = embedded_preview(_Raw(_Thumb(rawpy.ThumbFormat.BITMAP)), path)
    assert img is not None
    arr = np.asarray(img)
    assert arr.shape == (10, 16, 3) and arr.dtype == np.uint8
    assert arr.max() == 0x80


def test_plain_tiff_without_subifds_has_no_preview_page(tmp_path):
    """Page 0 is the image itself, not a preview, so nothing is offered."""
    path = str(tmp_path / "scan.tif")
    tifffile.imwrite(path, np.zeros((10, 16), dtype=np.uint8))
    assert embedded_preview(_Raw(_Thumb(rawpy.ThumbFormat.BITMAP)), path) is None


def test_thumb_extraction_failure_returns_none():
    assert embedded_preview(_Raw(error=RuntimeError("no thumbnail")), "missing.nef") is None


def test_object_without_extract_thumb_returns_none():
    assert embedded_preview(object(), "missing.dng") is None
