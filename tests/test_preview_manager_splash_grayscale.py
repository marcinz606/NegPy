"""Regression: grayscale (H,W,1) BITMAP thumbnails must not break splash/load.

Silverfast-scanned Nikon Coolscan 5000/9000ED DNGs embed a single-channel
grayscale thumbnail. PIL's Image.fromarray cannot map (H,W,1) uint8, which used
to abort the whole file load with "cannot handle this data type : (1, 1, 1) |u1".
"""

from unittest.mock import MagicMock

import numpy as np
import rawpy

from negpy.services.rendering.preview_manager import PreviewManager


def _make_thumb(channels: int, w: int = 64, h: int = 48) -> MagicMock:
    thumb = MagicMock()
    thumb.format = rawpy.ThumbFormat.BITMAP
    thumb.data = np.zeros((h, w, channels), dtype=np.uint8)
    raw = MagicMock()
    raw.sizes = MagicMock(iheight=h, iwidth=w)
    raw.extract_thumb.return_value = thumb
    return raw


def test_grayscale_bitmap_thumb_yields_splash():
    """A single-channel thumb is broadened to RGB instead of crashing."""
    raw = _make_thumb(channels=1)
    result = PreviewManager._try_splash_from_open_raw(raw, "scan.dng")
    assert result is not None
    buf, _dims = result
    assert buf.ndim == 3 and buf.shape[2] == 3


def test_malformed_thumb_returns_none_not_raises():
    """Any thumb conversion failure falls back to None (skip splash), never raises."""
    raw = MagicMock()
    raw.sizes = MagicMock(iheight=48, iwidth=64)
    raw.extract_thumb.side_effect = None
    bad = MagicMock()
    bad.format = rawpy.ThumbFormat.BITMAP
    bad.data = object()  # not array-like -> fromarray raises
    raw.extract_thumb.return_value = bad
    assert PreviewManager._try_splash_from_open_raw(raw, "scan.dng") is None
