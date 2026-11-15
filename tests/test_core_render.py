"""The render-stage helpers: NumPy oracles for one-line rules, hash pins for the mask expansion."""

import numpy as np
import pytest
from core_pin import digest, linux_pin

import negpy_core
from negpy.features.exposure.logic import contrast_mask_ev, expand_mask_plane
from negpy.features.process.capture_color import apply_camera_matrix


def _img(seed, shape=(97, 131, 3)):
    return np.random.default_rng(seed).uniform(-0.2, 1.4, shape).astype(np.float32)


def test_log_density_open_matches_numpy_within_the_ledger():
    img = _img(1)
    img[0, 0] = [np.nan, np.inf, -np.inf]
    want = np.log10(np.clip(np.nan_to_num(img, nan=1e-6, posinf=1.0, neginf=1e-6), 1e-6, None))
    np.testing.assert_allclose(negpy_core.log_density_open(img), want, rtol=4e-7, atol=0)


def test_small_helpers_match_numpy():
    img = _img(2)
    lum = img[..., 1].copy()
    np.testing.assert_array_equal(negpy_core.gray_to_rgb(lum), np.stack([lum, lum, lum], axis=-1))
    np.testing.assert_array_equal(negpy_core.add(lum, img[..., 0].copy()), lum + img[..., 0])
    np.testing.assert_array_equal(negpy_core.clip01(img), np.clip(img, 0, 1))
    zeros = np.zeros((5, 6, 3), np.float32)
    assert not negpy_core.channel_any(zeros, 1)
    zeros[4, 5, 1] = np.nan
    assert negpy_core.channel_any(zeros, 1) == bool(zeros[..., 1].any())
    gains = np.array([1.7, 1.0, 0.6], np.float32)
    np.testing.assert_array_equal(negpy_core.scale_clip_unit(img, list(gains)), np.clip(img * gains, 0, 1))
    nan = _img(3)
    nan[1, 2] = [np.nan, np.inf, -np.inf]
    np.testing.assert_array_equal(negpy_core.nan_to_num(nan, 1e-6, 1.0, 1e-6), np.nan_to_num(nan, nan=1e-6, posinf=1.0, neginf=1e-6))


def test_camera_matrix_matches_numpy():
    img = _img(4)
    m = np.random.default_rng(5).uniform(-0.4, 1.3, (3, 3))
    want = (img.reshape(-1, 3).astype(np.float32) @ np.asarray(m, dtype=np.float32).T).reshape(img.shape)
    np.testing.assert_array_equal(apply_camera_matrix(img, m), want)


@linux_pin
def test_expand_mask_plane_output_is_pinned():
    outs = []
    for plane_shape in [(300, 400), (61, 87)]:
        plane = _img(6, plane_shape)
        for roi in [None, (0, 300, 0, 400), (20, 250, 35, 390), (-5, 320, 10, 500)]:
            outs += [
                expand_mask_plane(plane, (300, 400), roi),
                contrast_mask_ev(plane, (900.0, 30.0, -412.0, 611.0, 2e-5, 3e-5, 1.5), (300, 400), roi),
            ]
    assert digest(*outs) == "f022949c33afdb769e573b99576ccc441988f1cfe3758c310925e90dcd57d436"


def test_expand_mask_plane_of_an_empty_roi_is_none():
    assert expand_mask_plane(_img(6, (61, 87)), (300, 400), (100, 100, 0, 10)) is None


def test_wrapper_linear16_postprocess_is_the_clipped_float():
    from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper

    data = _img(9)
    data[1, 2] = [np.nan, np.inf, -np.inf]
    gains = (1.3, 1.0, 0.8)
    wrapper = NonStandardFileWrapper(data, wb_gains=gains)
    for wb in (False, True):
        for half in (False, True):
            src = data[::2, ::2] if half else data
            got = wrapper.postprocess(gamma=(1, 1), output_bps=16, use_camera_wb=wb, half_size=half)
            finite = np.nan_to_num(src, nan=0.0, posinf=1.0, neginf=0.0)
            want = np.clip(finite * np.array(gains, np.float32) if wb else finite, 0, 1)
            assert np.isfinite(got).all()
            assert got.dtype == np.float32 and not np.shares_memory(got, data)
            np.testing.assert_array_equal(got, want)


def _display_numpy(data, gains, gamma, top):
    """The NumPy display decode the kernel replaced, for values inside 0..1."""
    if gains is not None:
        data = np.clip(data * np.array(gains, np.float32), 0.0, 1.0)
    if gamma:
        data = np.where(data < 0.018, data * 4.5, 1.099 * np.power(np.maximum(data, 0.0), 1.0 / 2.222) - 0.099)
    return (data * top).astype(np.uint16 if top > 255 else np.uint8)


def test_wrapper_display_postprocess_matches_numpy():
    from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper

    data = np.clip(_img(12), 0.0, 1.0)
    data[0, :8] = np.linspace(0.0, 0.03, 8, dtype=np.float32)[:, None]
    gains = (1.3, 1.0, 0.8)
    wrapper = NonStandardFileWrapper(data, wb_gains=gains)
    for wb in (False, True):
        for half in (False, True):
            src = data[::2, ::2] if half else data
            linear8 = wrapper.postprocess(gamma=(1, 1), output_bps=8, use_camera_wb=wb, half_size=half)
            np.testing.assert_array_equal(linear8, _display_numpy(src, gains if wb else None, False, 255.0))
            for bps, top in ((8, 255.0), (16, 65535.0)):
                got = wrapper.postprocess(output_bps=bps, use_camera_wb=wb, half_size=half)
                want = _display_numpy(src, gains if wb else None, True, top)
                assert got.dtype == want.dtype and got.shape == want.shape
                # NumPy's float32 power is not correctly rounded, so a code may differ by one.
                assert np.abs(got.astype(np.int32) - want.astype(np.int32)).max() <= 1
                assert (got != want).mean() < 0.01


def test_display_decode_saturates_outside_the_codes():
    values = np.array([[np.nan, -0.5, 0.0, 1.0, 1.5, np.inf, -np.inf]], np.float32)
    np.testing.assert_array_equal(negpy_core.display_decode(values, None, False, 8), [[0, 0, 0, 255, 255, 255, 0]])
    np.testing.assert_array_equal(negpy_core.display_decode(values, None, True, 16)[0, [0, 1, 2, 4, 5, 6]], [0, 0, 0, 65535, 65535, 0])
    with pytest.raises(ValueError):
        negpy_core.display_decode(values, (1.2, 1.0, 0.9), False, 8)
