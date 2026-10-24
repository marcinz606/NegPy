"""inpaint_ns against cv2.inpaint on synthetic tiles, and a hash pin of the hair inpaint."""

import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle
from core_pin import digest, wave

from negpy.features.retouch import logic
from negpy.features.retouch.logic import apply_hair_inpaint

pytestmark = cv2_oracle

_RNG = np.random.default_rng(23)


def _textured(h, w):
    img = cv2.GaussianBlur(_RNG.random((h, w, 3), dtype=np.float32), (0, 0), 2.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img += (0.3 * wave(xx / 9.0) * wave(yy / 13.0 + 0.5))[..., None]
    return np.ascontiguousarray((img - img.min()) / (img.max() - img.min()), dtype=np.float32)


def _encode(crop, lo, inv_span):
    enc = np.empty(crop.shape, np.uint8)
    negpy_core.hair_encode(np.ascontiguousarray(crop), np.float32(lo), np.float32(inv_span), np.float32(1 / 2.2), enc)
    return enc


def _strokes(h, w, n, width):
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        pts = np.cumsum(_RNG.normal(0, max(2.0, min(h, w) / 12), (40, 2)), 0) + (_RNG.random(2) * (w, h))
        cv2.polylines(m, [pts.astype(np.int32)], False, 1, width)
    return m


def _masks(h, w):
    return [
        (_RNG.random((h, w)) < 0.15).astype(np.uint8),
        _strokes(h, w, 3, 1),
        _strokes(h, w, 2, 4),
        cv2.dilate(_strokes(h, w, 1, 2), np.ones((3, 3), np.uint8)),
    ]


def _gate(want, got, mask):
    d = np.abs(want.astype(np.int16) - got.astype(np.int16))
    assert np.array_equal(want[mask == 0], got[mask == 0])
    if mask.any():
        inside = d[mask > 0]
        assert inside.mean() <= 1.0 and inside.max() <= 4, (inside.mean(), inside.max())


def _tiles():
    return [_encode(_textured(h, w), 0.0, 1.0) for h, w in ((64, 80), (2, 2), (5, 300), (257, 190))]


@pytest.mark.parametrize("radius", [1, 3, 7])
def test_inpaint_ns_matches_cv2(radius):
    # A one-pixel-wide image is left out: OpenCV reads past the image there.
    for enc in _tiles():
        h, w = enc.shape[:2]
        for mask in _masks(h, w):
            want = cv2.inpaint(enc, mask, radius, cv2.INPAINT_NS)
            _gate(want, negpy_core.inpaint_ns(enc, mask, float(radius)), mask)


def test_inpaint_ns_without_mask_is_identity():
    enc = _encode(_textured(30, 40), 0.0, 1.0)
    assert np.array_equal(negpy_core.inpaint_ns(enc, np.zeros((30, 40), np.uint8), 3.0), enc)


def test_hair_inpaint_pin(monkeypatch):
    monkeypatch.setitem(globals(), "_RNG", np.random.default_rng(23))
    # A small tile makes the larger frames run the tiled path with its halo.
    monkeypatch.setattr(logic, "_HAIR_TILE_PX", 96)
    got = []
    for img in (_textured(220, 340), _textured(90, 2100)):
        h, w = img.shape[:2]
        for hair in (_strokes(h, w, 2, 1), _strokes(h // 2, w // 2, 3, 2)):
            got.append(np.asarray(apply_hair_inpaint(img, [hair])))
    assert digest(got) == "ce305e209c674492faf140178163368125dba16675eaf372c4105148fded3ee2"
