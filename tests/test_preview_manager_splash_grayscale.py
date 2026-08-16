"""Regression: the splash must not read a BITMAP thumb's buffer.

Silverfast-scanned Nikon Coolscan 5000/9000ED DNGs embed a single-channel grayscale
thumbnail, which libraw still reports as 3-channel. Reading it ran past the allocation
and segfaulted at random. The splash comes from the file's TIFF preview page instead,
and is skipped when there is none.
"""

from unittest.mock import MagicMock

import numpy as np
import tifffile

import rawpy
from negpy.services.rendering.preview_manager import PreviewManager


def _raw_with_bitmap_thumb(w: int = 64, h: int = 48) -> MagicMock:
    thumb = MagicMock()
    thumb.format = rawpy.ThumbFormat.BITMAP
    type(thumb).data = property(lambda _self: (_ for _ in ()).throw(AssertionError("BITMAP thumb data must never be read")))
    raw = MagicMock()
    raw.sizes = MagicMock(iheight=h, iwidth=w)
    raw.extract_thumb.return_value = thumb
    return raw


def test_bitmap_thumb_splashes_from_the_preview_page(tmp_path):
    """The grayscale preview page is broadened to RGB, without touching the thumb."""
    path = str(tmp_path / "scan.dng")
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.full((48, 64), 128, dtype=np.uint8), subifds=1, subfiletype=1)
        tw.write(np.zeros((480, 640), dtype=np.uint16), subfiletype=0)

    result = PreviewManager._try_splash_from_open_raw(_raw_with_bitmap_thumb(), path)
    assert result is not None
    buf, _dims = result
    assert buf.ndim == 3 and buf.shape[2] == 3


def test_bitmap_thumb_without_preview_page_skips_the_splash(tmp_path):
    """No safe preview to read -> no splash, and no crash."""
    assert PreviewManager._try_splash_from_open_raw(_raw_with_bitmap_thumb(), str(tmp_path / "gone.dng")) is None


def test_malformed_thumb_returns_none_not_raises():
    """Any thumb extraction failure falls back to None (skip splash), never raises."""
    raw = MagicMock()
    raw.sizes = MagicMock(iheight=48, iwidth=64)
    raw.extract_thumb.side_effect = RuntimeError("no thumbnail")
    assert PreviewManager._try_splash_from_open_raw(raw, "scan.dng") is None
