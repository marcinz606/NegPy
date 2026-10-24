import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

pytestmark = cv2_oracle

_RNG = np.random.default_rng(11)
_SHAPES = [(41, 83, 1), (37, 53, 3), (57, 61, 4), (1, 301, 1), (300, 1, 1), (13, 9, 1)]
_BORDERS = [cv2.BORDER_REFLECT_101, cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT]


def _image(h, w, cn):
    return (_RNG.random((h, w, cn) if cn > 1 else (h, w), dtype=np.float32) - 0.3) * 3


def _symmetric(n):
    half = _RNG.random(n // 2 + 1).astype(np.float32)
    return np.concatenate([half[:-1], half[-1:], half[:-1][::-1]]).astype(np.float32)


@pytest.mark.parametrize("n, sigma", [(1, 0), (3, 0), (5, 0), (7, 0), (9, 0), (11, 0), (15, 0), (17, 2.0), (13, 1.5), (33, 4.0), (5, 0.8)])
def test_gaussian_kernel_matches_cv2(n, sigma):
    want = cv2.getGaussianKernel(n, sigma, cv2.CV_32F).ravel()
    assert np.array(negpy_core.gaussian_kernel(n, sigma), np.float32).tobytes() == want.tobytes()


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("border", _BORDERS)
@pytest.mark.parametrize(
    "ksize, sigma_x, sigma_y",
    [((5, 5), 0, 0), ((7, 7), 0, 0), ((11, 1), 0, 0), ((0, 0), 2.0, 0), ((0, 0), 0.7, 0), ((1, 0), 0, 2.5), ((3, 3), 0, 0), ((9, 9), 0, 0)],
)
def test_gaussian_blur_matches_cv2_bit_for_bit(shape, border, ksize, sigma_x, sigma_y):
    img = _image(*shape)
    want = cv2.GaussianBlur(img, ksize, sigma_x, sigmaY=sigma_y, borderType=border)
    got = negpy_core.gaussian_blur(img, ksize, sigma_x, sigma_y, border)
    assert got.tobytes() == want.tobytes()


_KERNELS = {
    "sym3": (_symmetric(3), _symmetric(3)),
    "sym5": (_symmetric(5), _symmetric(5)),
    "sym9x3": (_symmetric(9), _symmetric(3)),
    "sym11": (_symmetric(11), _symmetric(11)),
    "sobel_x": (np.array([-1, 0, 1], np.float32), np.array([1, 2, 1], np.float32)),
    "sobel_y": (np.array([1, 2, 1], np.float32), np.array([-1, 0, 1], np.float32)),
    "scaled_derivative": (np.array([-0.5, 0, 0.5], np.float32), np.array([0.25, 0.5, 0.25], np.float32)),
    "second_derivative": (np.array([1, -2, 1], np.float32), np.array([1, -2, 1], np.float32)),
    "row5_special": (np.array([1, 0, -2, 0, 1], np.float32), np.array([1], np.float32)),
}


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("border", _BORDERS[:3])
@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_sep_filter2d_matches_cv2_bit_for_bit(shape, border, name):
    kx, ky = _KERNELS[name]
    img = _image(*shape)
    want = cv2.sepFilter2D(img, -1, kx, ky, borderType=border)
    assert negpy_core.sep_filter2d(img, kx, ky, border).tobytes() == want.tobytes()


def test_sep_filter2d_rejects_a_kernel_it_does_not_cover():
    with pytest.raises(ValueError):
        negpy_core.sep_filter2d(_image(9, 9, 1), np.array([0.2, 0.5, 0.3], np.float32), np.array([1], np.float32))


@pytest.mark.parametrize("shape", _SHAPES + [(9, 700, 1)])
@pytest.mark.parametrize("border", _BORDERS)
@pytest.mark.parametrize("ksize", [(1, 1), (3, 3), (5, 5), (2, 4), (4, 2), (7, 7), (3, 9), (9, 3), (15, 15), (6, 11), (31, 1), (1, 31)])
@pytest.mark.parametrize("normalize", [True, False])
def test_box_filter_matches_cv2_bit_for_bit(shape, border, ksize, normalize):
    img = (_image(*shape) * _RNG.choice([1.0, 1e3])).astype(np.float32)
    want = cv2.boxFilter(img, -1, ksize, normalize=normalize, borderType=border)
    got = negpy_core.box_filter(img, ksize, normalize, border)
    assert got.tobytes() == want.tobytes()
