"""Quality gate for negpy_core.resize_kernel against cv2's INTER_LANCZOS4 and INTER_CUBIC."""

import cv2
import numpy as np
import pytest
from core_oracle import cv2_oracle

import negpy_core

pytestmark = cv2_oracle

CASES = [
    ((120, 170, 3), (400, 290)),
    ((300, 400, 3), (1003, 751)),
    ((64, 80), (97, 61)),
    ((200, 300, 4), (301, 201)),
    ((50, 70, 3), (35, 25)),
]


@pytest.mark.parametrize("shape,dsize", CASES)
@pytest.mark.parametrize("lanczos,flag,tol", [(True, cv2.INTER_LANCZOS4, 1e-6), (False, cv2.INTER_CUBIC, 1e-5)])
def test_resize_kernel_stays_near_cv2(shape, dsize, lanczos, flag, tol):
    rng = np.random.default_rng(sum(shape))
    src = cv2.GaussianBlur(rng.uniform(0, 1, shape).astype(np.float32), (0, 0), 1.2)
    want = cv2.resize(src, dsize, interpolation=flag)
    got = negpy_core.resize_kernel(np.ascontiguousarray(src), dsize, lanczos=lanczos)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=tol)


U8_CASES = [
    ((20, 30, 3), (45, 31)),
    ((17, 23), (40, 33)),
    ((200, 300, 3), (450, 300)),
    ((300, 200, 4), (500, 333)),
    ((64, 64, 3), (48, 40)),
    ((9, 7, 3), (31, 29)),
]


@pytest.fixture
def no_ipp():
    cv2.ipp.setUseIPP(False)
    yield
    cv2.ipp.setUseIPP(True)


@pytest.mark.parametrize("shape,dsize", U8_CASES)
def test_uint8_cubic_matches_opencv_without_ipp(shape, dsize, no_ipp):
    src = np.random.default_rng(sum(shape)).integers(0, 256, shape).astype(np.uint8)
    np.testing.assert_array_equal(negpy_core.resize_u8(src, dsize, cubic=True), cv2.resize(src, dsize, interpolation=cv2.INTER_CUBIC))


@pytest.mark.parametrize("shape,dsize", [((400, 600, 3), (240, 160)), ((600, 900, 3), (300, 200)), ((101, 77, 3), (40, 31))])
def test_uint8_area_matches_cv2(shape, dsize):
    src = np.random.default_rng(sum(shape)).integers(0, 256, shape).astype(np.uint8)
    np.testing.assert_array_equal(negpy_core.resize_u8(src, dsize), cv2.resize(src, dsize, interpolation=cv2.INTER_AREA))


@pytest.mark.parametrize("dtype,top", [(np.uint8, 256), (np.uint16, 65536)])
@pytest.mark.parametrize("shape,dh", [((100, 37, 3), 173), ((50, 64), 101), ((40, 40, 4), 77)])
def test_row_linear_matches_opencv_without_ipp(dtype, top, shape, dh, no_ipp):
    src = np.random.default_rng(dh).integers(0, top, shape).astype(dtype)
    want = cv2.resize(src, (shape[1], dh), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(negpy_core.resize_rows_linear(src, dh), want)
