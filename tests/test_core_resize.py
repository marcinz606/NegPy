import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

pytestmark = cv2_oracle

_RNG = np.random.default_rng(5)


def _image(h, w, cn):
    return _RNG.random((h, w, cn) if cn > 1 else (h, w), dtype=np.float32) ** 2.2


@pytest.mark.parametrize("cn", [1, 3, 4])
@pytest.mark.parametrize(
    "shape, dsize",
    [
        ((394, 566), (160, 111)),  # the detection downscale, at a tenth of its size
        ((1600, 1059), (1059, 701)),
        ((37, 53), (17, 11)),
        ((100, 98), (49, 40)),  # integer scale on one axis only
        ((64, 64), (32, 32)),  # 2x2: SIMD order
        ((100, 98), (49, 50)),
        ((40, 74), (37, 20)),  # 2x2 with a scalar tail column
        ((90, 90), (30, 30)),  # 3x3
        ((80, 80), (20, 20)),  # 4x4
        ((80, 60), (60, 80)),  # same size
    ],
)
def test_resize_area_matches_cv2_bit_for_bit(shape, dsize, cn):
    img = _image(*shape, cn)
    want = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
    got = negpy_core.resize_area(img, dsize)
    assert got.shape == want.shape
    assert got.tobytes() == want.tobytes()


_SIZES = [
    ((60, 90), (130, 200)),
    ((61, 91), (177, 300)),
    ((100, 140), (37, 51)),
    ((7, 5), (40, 64)),
    ((300, 517), (900, 1100)),
    ((80, 120), (40, 60)),  # an exact halving
    ((80, 120), (160, 50)),  # one axis grows, one shrinks
    ((1, 9), (4, 30)),
    ((9, 1), (30, 4)),
]


@pytest.fixture
def no_ipp():
    # opencv-python runs IPP's float32 bilinear resize on x86, whose bits depend on the CPU;
    # negpy_core copies OpenCV's own code, which a build without IPP runs.
    cv2.ipp.setUseIPP(False)
    yield
    cv2.ipp.setUseIPP(True)


@pytest.mark.parametrize("cn", [1, 2, 3, 4])
@pytest.mark.parametrize("shape, dshape", _SIZES)
def test_resize_linear_matches_cv2_without_ipp(shape, dshape, cn, no_ipp):
    img = _image(*shape, cn)
    dsize = (dshape[1], dshape[0])
    if cn == 2 and dshape[0] * 2 == shape[0] and dshape[1] * 2 == shape[1]:
        with pytest.raises(ValueError):
            negpy_core.resize_linear(img, dsize)
        return
    want = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
    assert negpy_core.resize_linear(img, dsize).tobytes() == want.tobytes()


@pytest.mark.parametrize("cn", [1, 3, 4])
@pytest.mark.parametrize("shape, dshape", [s for s in _SIZES if s[1][0] > s[0][0] or s[1][1] > s[0][1]])
def test_resize_area_upscale_matches_cv2(shape, dshape, cn):
    img = _image(*shape, cn)
    dsize = (dshape[1], dshape[0])
    want = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
    assert negpy_core.resize_area(img, dsize).tobytes() == want.tobytes()


def test_resize_area_rejects_strided_input():
    img = _image(20, 20, 3)
    with pytest.raises(ValueError):
        negpy_core.resize_area(img[:, ::2], (5, 10))


@pytest.mark.parametrize("cn", [1, 3, 4])
@pytest.mark.parametrize("shape, dshape", _SIZES)
def test_resize_nearest_matches_cv2(shape, dshape, cn):
    img = (_image(*shape, cn) * 255).astype(np.uint8)
    dsize = (dshape[1], dshape[0])
    want = cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)
    assert negpy_core.resize_nearest(img, dsize).tobytes() == want.tobytes()
