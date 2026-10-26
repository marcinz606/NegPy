"""Quality gate for negpy_core.phase_correlate, the float64 port of cv2.phaseCorrelate."""

import cv2
import numpy as np
import pytest
from core_oracle import cv2_oracle

import negpy_core

SHIFTS = [(3.0, -2.0), (0.4, 1.7), (-12.3, 5.5), (0.0, 0.0), (7.75, -0.25)]


def _texture(seed, shape):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.uniform(0, 1, shape).astype(np.float32), (0, 0), 2.5)


def _shifted(img, dx, dy):
    h, w = img.shape
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("shape", [(300, 400), (257, 361), (512, 1024)])
@pytest.mark.parametrize("dx,dy", SHIFTS)
def test_recovers_the_true_shift(shape, dx, dy):
    """Within the bias of the 5x5 peak centroid, which cv2 shares."""
    base = _texture(sum(shape), shape)
    (gx, gy), response = negpy_core.phase_correlate(base, _shifted(base, dx, dy))
    assert abs(gx - dx) < 0.2 and abs(gy - dy) < 0.2
    assert response > 0.5


@pytest.mark.parametrize("shape", [(257, 361), (300, 401), (375, 270)])
def test_an_odd_padded_size_reads_no_shift_as_zero(shape):
    base = _texture(3, shape)
    (gx, gy), _ = negpy_core.phase_correlate(base, base.copy())
    assert abs(gx) < 1e-6 and abs(gy) < 1e-6


@cv2_oracle
@pytest.mark.parametrize("shape", [(300, 400), (257, 361)])
@pytest.mark.parametrize("dx,dy", SHIFTS)
def test_stays_near_cv2(shape, dx, dy):
    """cv2 measures from n / 2.0, half a pixel off along an odd padded axis."""
    h, w = shape
    base = _texture(7, shape)
    mov = _shifted(base, dx, dy)
    (cx, cy), _ = cv2.phaseCorrelate(base, mov, cv2.createHanningWindow((w, h), cv2.CV_32F))
    cx -= 0.5 * (cv2.getOptimalDFTSize(w) % 2)
    cy -= 0.5 * (cv2.getOptimalDFTSize(h) % 2)
    (gx, gy), _ = negpy_core.phase_correlate(base, mov)
    assert abs(gx - cx) < 0.05 and abs(gy - cy) < 0.05


def test_recovers_the_shift_at_16_bit_scale():
    base = _texture(11, (684, 1024)) * 65535
    (gx, gy), _ = negpy_core.phase_correlate(base, _shifted(base, 3.0, -2.0))
    assert abs(gx - 3.0) < 0.5 and abs(gy + 2.0) < 0.5


def test_rejects_mismatched_planes():
    with pytest.raises(ValueError):
        negpy_core.phase_correlate(np.zeros((4, 4), np.float32), np.zeros((4, 5), np.float32))
