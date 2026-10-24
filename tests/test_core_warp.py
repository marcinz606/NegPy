import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

from negpy.features.geometry.logic import keystone_matrix

pytestmark = cv2_oracle

_RNG = np.random.default_rng(11)
_REPLICATE = {"flags": cv2.INTER_LINEAR, "borderMode": cv2.BORDER_REPLICATE}
_INTERP = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
_BORDER = {"replicate": cv2.BORDER_REPLICATE, "constant": cv2.BORDER_CONSTANT}
_MODES = [(i, b) for i in ("linear", "cubic") for b in _BORDER]


def _image(h, w, cn):
    return _RNG.random((h, w, cn) if cn > 1 else (h, w), dtype=np.float32) * 3 - 1


@pytest.mark.parametrize("mode", [*_MODES, ("nearest", "replicate"), ("nearest", "constant")])
@pytest.mark.parametrize("cn", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("shape, dshape", [((97, 131), (80, 150)), ((1, 1), (3, 4)), ((3, 2), (9, 9)), ((300, 517), (300, 517))])
def test_remap_matches_cv2(cn, shape, dshape, mode):
    if mode[0] == "cubic" and cn > 4:
        pytest.skip("cv2 interpolates bicubic up to 4 channels")
    img = _image(*shape, cn)
    mx = (_RNG.random(dshape) * (shape[1] + 20) - 10).astype(np.float32)
    my = (_RNG.random(dshape) * (shape[0] + 20) - 10).astype(np.float32)
    want = cv2.remap(img, mx, my, _INTERP[mode[0]], borderMode=_BORDER[mode[1]])
    assert negpy_core.remap(img, mx, my, *mode).tobytes() == want.tobytes()


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("cn", [1, 3, 4])
@pytest.mark.parametrize("angle", [0.7, -3.3, 12.0])
@pytest.mark.parametrize("shape", [(97, 131), (2, 3)])
def test_warp_affine_modes_match_cv2(cn, angle, shape, mode):
    img = _image(*shape, cn)
    h, w = shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 0.9).astype(np.float32).astype(np.float64)
    m[:, 2] += (5.5, -3.25)
    want = cv2.warpAffine(img, m, (w + 9, h + 4), flags=_INTERP[mode[0]], borderMode=_BORDER[mode[1]])
    got = negpy_core.warp_affine(img, m, (w + 9, h + 4), interpolation=mode[0], border=mode[1])
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("cn", [1, 3, 4])
@pytest.mark.parametrize("angle", [0.7, -3.3, 12.0, 90.0])
@pytest.mark.parametrize("shape", [(97, 131), (513, 260), (2, 3)])
def test_warp_affine_matches_cv2(cn, angle, shape):
    img = _image(*shape, cn)
    h, w = shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    for dsize in ((w, h), (w + 7, h - 1)):
        want = cv2.warpAffine(img, m, dsize, **_REPLICATE)
        assert negpy_core.warp_affine(img, m, dsize).tobytes() == want.tobytes()
    inv = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    assert negpy_core.warp_affine(img, m, (w, h), inverse_map=True).tobytes() == inv.tobytes()


@pytest.mark.parametrize("cn", [1, 3])
@pytest.mark.parametrize("converge", [(3.0, -2.0), (-7.5, 1.25), (0.0, 20.0)])
@pytest.mark.parametrize("shape", [(97, 131), (700, 1030), (5, 9)])
def test_warp_perspective_matches_cv2(cn, converge, shape):
    img = _image(*shape, cn)
    h, w = shape
    m = keystone_matrix(*converge, w, h)
    want = cv2.warpPerspective(img, m, (w, h), **_REPLICATE)
    assert negpy_core.warp_perspective(img, m, (w, h)).tobytes() == want.tobytes()


@pytest.mark.parametrize("mode", _MODES)
def test_warp_perspective_modes_match_cv2(mode):
    img = _image(97, 131, 3)
    m = keystone_matrix(5.0, -3.0, 131, 97)
    want = cv2.warpPerspective(img, m, (131, 97), flags=_INTERP[mode[0]], borderMode=_BORDER[mode[1]])
    assert negpy_core.warp_perspective(img, m, (131, 97), interpolation=mode[0], border=mode[1]).tobytes() == want.tobytes()


def test_warps_reject_bad_input():
    img = _image(10, 10, 3)
    with pytest.raises(ValueError):
        negpy_core.warp_affine(img[:, ::2], np.eye(2, 3), (5, 10))
    with pytest.raises(ValueError):
        negpy_core.warp_perspective(img, np.eye(2, 3), (10, 10))
    with pytest.raises(ValueError):
        negpy_core.remap(img, np.zeros((4, 4), np.float32), np.zeros((4, 5), np.float32))
    with pytest.raises(ValueError):
        negpy_core.warp_affine(img, np.eye(2, 3), (10, 10), interpolation="nearest")
    with pytest.raises(ValueError):
        negpy_core.remap(img, np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32), border="reflect")
