import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

pytestmark = cv2_oracle

_RNG = np.random.default_rng(13)
_SHAPES = [(31, 281), (15, 243), (6, 297), (1, 77), (40, 1), (2, 2)]


@pytest.fixture
def one_cv2_thread():
    # cv2's parallel uint8 5x5 GaussianBlur can return different bytes on repeated calls;
    # on one thread it is deterministic and negpy_core matches it.
    before = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(before)


def _gray(shape):
    return cv2.GaussianBlur(_RNG.integers(0, 256, shape).astype(np.uint8), (0, 0), 1.5)


@pytest.mark.parametrize("shape", _SHAPES)
def test_bgr_to_gray_matches_cv2(shape):
    img = _RNG.integers(0, 256, (*shape, 3)).astype(np.uint8)
    assert np.array_equal(negpy_core.bgr_to_gray(img), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize(
    "ksize, sigma, border",
    [
        ((5, 5), 0, 4),
        ((7, 7), 0, 4),
        ((3, 3), 0, 4),
        ((0, 0), 1.5, 4),
        ((0, 0), 2.5, 1),
        ((9, 1), 0, 4),
        ((0, 0), 1.0, 0),
        ((11, 11), 0, 2),
    ],
)
def test_gaussian_blur_u8_matches_cv2(one_cv2_thread, shape, ksize, sigma, border):
    img = _gray(shape)
    want = cv2.GaussianBlur(img, ksize, sigma, borderType=border)
    assert np.array_equal(negpy_core.gaussian_blur_u8(img, ksize, sigma, 0.0, border), want)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("invert", [False, True])
def test_threshold_otsu_matches_cv2(shape, invert):
    img = _gray(shape)
    flag = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    want_t, want = cv2.threshold(img, 0, 255, flag)
    got_t, got = negpy_core.threshold_otsu(img, 255, invert)
    assert got_t == want_t
    assert np.array_equal(got, want)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("low, high", [(40, 160), (10, 30), (100, 50)])
def test_canny_matches_cv2(shape, low, high):
    img = _gray(shape)
    assert np.array_equal(negpy_core.canny(img, low, high), cv2.Canny(img, low, high))


@pytest.mark.parametrize("shape", [(1, 1), (1, 30), (30, 1), (5, 5), (37, 53), (130, 140)])
@pytest.mark.parametrize("ksize", [1, 3, 5, 7, 9, 25, 33])
@pytest.mark.parametrize("kind", ["noise", "smooth", "levels"])
def test_median_blur_matches_cv2(shape, ksize, kind):
    rng = np.random.default_rng(ksize * 7 + shape[0])
    img = rng.integers(0, 256, shape, dtype=np.uint8)
    if kind == "smooth":
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif kind == "levels":
        img = (img // 64 * 64).astype(np.uint8)
    assert negpy_core.median_blur(img, ksize).tobytes() == cv2.medianBlur(img, ksize).tobytes()
