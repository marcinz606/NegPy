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


def _split_rgb_ir_numpy(arr, scale):
    full = np.clip(arr.astype(np.float32) * scale, 0.0, 1.0)
    return np.ascontiguousarray(full[:, :, :3]), np.ascontiguousarray(full[:, :, 3])


def _levels_numpy(arr, black, denom, lut):
    block = arr.astype(np.float64)
    if lut is not None:
        np.clip(block, 0, len(lut) - 1, out=block)
        block = lut[block.astype(np.int64)]
    return np.clip((block - black) / denom, 0.0, 1.0).astype(np.float32)


@pytest.mark.parametrize(
    "dtype,scale",
    [(np.uint16, 1.0 / 65535.0), (np.uint8, 1.0 / 255.0), (np.float32, 1.0)],
)
def test_split_rgb_ir_matches_numpy(dtype, scale):
    rng = np.random.default_rng(4)
    if dtype == np.float32:
        arr = rng.uniform(-0.5, 1.5, (37, 53, 4)).astype(np.float32)
    else:
        arr = rng.integers(0, np.iinfo(dtype).max, (37, 53, 4), endpoint=True, dtype=dtype)
    rgb, ir = negpy_core.split_rgb_ir(arr)
    want_rgb, want_ir = _split_rgb_ir_numpy(arr, scale)
    np.testing.assert_array_equal(rgb, want_rgb)
    np.testing.assert_array_equal(ir, want_ir)


@pytest.mark.parametrize("dtype", [np.uint16, np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("with_lut", [False, True])
def test_levels_to_unit_matches_numpy(dtype, with_lut):
    rng = np.random.default_rng(5)
    if np.issubdtype(dtype, np.integer):
        arr = rng.integers(0, np.iinfo(dtype).max, (41, 29, 3), endpoint=True, dtype=dtype)
    else:
        arr = rng.uniform(-50.0, 70000.0, (41, 29, 3)).astype(dtype)
    black = np.array([256.0, 300.5, 1.0])
    denom = np.maximum(np.array([60000.0, 255.0, 1e-9]) - black, 1e-6)
    lut = np.sqrt(np.arange(4096, dtype=np.float64)) * 977.0 if with_lut else None
    got = negpy_core.levels_to_unit(arr, black.tolist(), denom.tolist(), lut=None if lut is None else lut.tolist())
    np.testing.assert_array_equal(got, _levels_numpy(arr, black, denom, lut))


@linux_pin
def test_dng_peek_kernels_pin():
    rng = np.random.default_rng(6)
    four = rng.integers(0, 65535, (23, 31, 4), endpoint=True, dtype=np.uint16)
    three = rng.integers(0, 65535, (23, 31, 3), endpoint=True, dtype=np.uint16)
    lut = (np.arange(65536, dtype=np.float64) ** 0.8).tolist()
    got = (
        negpy_core.split_rgb_ir(four),
        negpy_core.levels_to_unit(three, [512.0, 600.0, 700.0], [60000.0, 59000.0, 58000.0]),
        negpy_core.levels_to_unit(three, [512.0, 600.0, 700.0], [7000.0, 6900.0, 6800.0], lut=lut),
    )
    assert digest(got) == "fd11d821816f58136a027a1aade4a9213fc589640cc3a64993671f2327d2a8d5"
