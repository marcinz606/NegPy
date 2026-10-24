"""negpy_core's line segment detector against cv2's LSD on synthetic frames: long segments must
land on cv2's. The angle modes of `_coarse_angles` are pinned by hash."""

import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle
from core_pin import digest

from negpy.features.geometry import skew

pytestmark = cv2_oracle


def _to_u8(lum):
    lo, hi = np.percentile(lum, (1.0, 99.0))
    return (np.clip((lum - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)


def _matched_fraction(lum):
    """Fraction of cv2's segments longer than `_MIN_SEGMENT` of the long side that a negpy_core
    segment matches with both endpoints within 1 px, in either direction."""
    u8 = _to_u8(lum)
    ref = cv2.createLineSegmentDetector().detect(u8)[0]
    ref = np.empty((0, 4), np.float32) if ref is None else ref.reshape(-1, 4)
    got = negpy_core.line_segments(u8)
    long = ref[np.hypot(ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]) >= skew._MIN_SEGMENT * max(u8.shape)]
    if long.size == 0:
        return 1.0
    if got.size == 0:
        return 0.0

    def far(a, b):
        return np.maximum(np.hypot(got[:, 0] - a[0], got[:, 1] - a[1]), np.hypot(got[:, 2] - b[0], got[:, 3] - b[1]))

    hits = [min(far(s[:2], s[2:]).min(), far(s[2:], s[:2]).min()) <= 1.0 for s in long]
    return float(np.mean(hits))


def _synthetic(angle, seed):
    """A light bed, a dark film frame and a picture with a few hard lines, turned by `angle`."""
    rng = np.random.default_rng(seed)
    h, w = 900, 1300
    img = np.full((h, w), 0.92, np.float32)
    img[150:750, 180:1120] = 0.08
    img[190:710, 230:1070] = 0.55
    img[300:330, 300:1000] = 0.3
    img[420:640, 600:615] = 0.8
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return img + rng.normal(0.0, 0.02, img.shape).astype(np.float32)


_ANGLES = [-9.5, -4.2, -0.7, 0.0, 0.3, 2.6, 6.1, 9.8]


@pytest.mark.parametrize("angle", _ANGLES)
def test_synthetic_frames_match_cv2(angle):
    assert _matched_fraction(_synthetic(angle, int(abs(angle) * 10))) >= 0.95


def test_coarse_angles_pin():
    modes = [skew._coarse_angles(_synthetic(angle, int(abs(angle) * 10))) for angle in _ANGLES]
    assert all(modes)
    assert digest(modes) == "0ef50afd7114fe37a77d6c718a527d8b7ddbb82be2c38b921a34ef46b47ed476"


def test_flat_and_tiny_planes():
    flat = np.full((64, 80), 0.5, np.float32)
    assert negpy_core.line_segments(_to_u8(flat)).shape == (0, 4)
    assert skew._coarse_angles(flat) == []
    assert negpy_core.line_segments(np.zeros((1, 1), np.uint8)).shape == (0, 4)
