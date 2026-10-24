"""Hash pins of the loader kernels on fixed inputs, and checks against single cv2 calls."""

import cv2
import numpy as np
import pytest
from core_oracle import cv2_oracle
from core_pin import digest, linux_pin

import negpy_core
from negpy.infrastructure.loaders.logluv import normalize_linear


@pytest.mark.parametrize(
    "shape,want",
    [
        ((1, 1, 3), "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0"),
        ((7, 5, 3), "870392426c6e48c2e6541e4c8d193a73064648b8dc6a1917d960c95f5b3abe40"),
        ((311, 517, 3), "2f96ccdae2c08d83000ff4193cf5a35830034a0c86e392b5a75cac7475a6ac77"),
        ((40, 30, 4), "cffa44f70d09ca0e66f4af8afbadbd8d75625f705b471f9434f84d7d8e29ccb8"),
    ],
)
@linux_pin
def test_normalize_linear_pin(shape, want):
    rng = np.random.default_rng(sum(shape))
    lin = rng.lognormal(0.0, 1.5, shape) * np.array([0.3, 1.0, 2.0, 1.0][: shape[-1]])
    lin[..., -1] = 0.25 if shape[0] == 7 else lin[..., -1]
    assert digest(normalize_linear(lin)) == want


@cv2_oracle
def test_ellipse_kernel_matches_cv2():
    for w in range(1, 42):
        for h in (1, 2, 3, 6, 9, 17, w):
            want = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))
            np.testing.assert_array_equal(negpy_core.ellipse_kernel((w, h)), want, err_msg=f"{w}x{h}")


def test_constants_match_cv2():
    for name in (
        "BORDER_CONSTANT",
        "BORDER_REPLICATE",
        "BORDER_REFLECT",
        "BORDER_REFLECT_101",
        "MORPH_ERODE",
        "MORPH_DILATE",
        "MORPH_OPEN",
        "MORPH_CLOSE",
        "MORPH_GRADIENT",
        "MORPH_TOPHAT",
        "MORPH_BLACKHAT",
    ):
        assert getattr(negpy_core, name) == getattr(cv2, name), name


@pytest.mark.parametrize(
    "samples,preserve_ir,want",
    [
        (3, False, "fb8fb89f0ae4823a776b85f79c6f6e0649bb6264b55ccb79408488cdf5671cfe"),
        (4, False, "39a04764e022fe88c2f98d60301ecad5ae48544d5683d1a845252421dd36a660"),
        (4, True, "7696eb79268752ef6349e668a915d61288c454115b143f1a94a250c8d7ab07a0"),
    ],
)
@linux_pin
def test_streamed_linearraw_preview_pin(tmp_path, samples, preserve_ir, want):
    import tifffile

    from negpy.infrastructure.loaders import rawpy_loader

    rng = np.random.default_rng(samples)
    data = rng.integers(0, 65535, (300, 470, samples), dtype=np.uint16)
    path = tmp_path / "scan.tif"
    tifffile.imwrite(path, data, tile=(64, 64), photometric="rgb", extrasamples=[0] * (samples - 3))
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        got = rawpy_loader._stream_linearraw_preview(page, page, 100, normalize_tags=False, min_preserve_ir=preserve_ir)
    assert digest(got) == want
