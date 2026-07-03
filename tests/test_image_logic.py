import numpy as np
from negpy.features.exposure.logic import (
    apply_characteristic_curve,
)
from negpy.kernel.image.logic import apply_exif_orientation, float_to_uint8, float_to_uint16


def test_apply_exif_orientation_noop():
    arr = np.arange(6).reshape(2, 3).astype(np.float32)
    for o in (1, 0, None):
        assert np.array_equal(apply_exif_orientation(arr, o), arr)


def test_apply_exif_orientation_rotations_match_numpy():
    arr = np.arange(6).reshape(2, 3).astype(np.float32)
    assert np.array_equal(apply_exif_orientation(arr, 3), np.rot90(arr, 2))
    assert np.array_equal(apply_exif_orientation(arr, 6), np.rot90(arr, 3))  # 90 CW
    assert np.array_equal(apply_exif_orientation(arr, 8), np.rot90(arr, 1))  # 90 CCW
    assert np.array_equal(apply_exif_orientation(arr, 2), np.fliplr(arr))
    assert np.array_equal(apply_exif_orientation(arr, 4), np.flipud(arr))


def test_apply_exif_orientation_swaps_dims_for_90deg():
    # (2,3,3) RGB → 90° rotation yields (3,2,3); channel axis preserved
    arr = np.random.rand(2, 3, 3).astype(np.float32)
    for o in (5, 6, 7, 8):
        out = apply_exif_orientation(arr, o)
        assert out.shape == (3, 2, 3)
        assert out.flags["C_CONTIGUOUS"]


def test_apply_film_characteristic_curve_range():
    img = np.array([[[0.1, 0.5, 0.9]]])
    # Params: (pivot, slope)
    params = (-2.5, 1.0)
    res = apply_characteristic_curve(img, params, params, params)
    assert res.shape == img.shape
    assert np.all(res >= 0.0)
    assert np.all(res <= 1.0)


def test_apply_film_characteristic_curve_positive_output():
    # Ensure POSITIVE output (Bright Input -> Dark Output)
    # 0.1 (Highlight) -> Bright Print
    # 0.9 (Shadow) -> Dark Print

    img = np.array([[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]])

    params = (-2.0, 1.0)  # Pivot -2.0, Slope 1.0
    res = apply_characteristic_curve(img, params, params, params)

    val_highlight_input = np.mean(res[0, 0])  # Input 0.1
    val_shadow_input = np.mean(res[0, 1])  # Input 0.9

    # Highlight Input (0.1) should result in Bright Output
    # Shadow Input (0.9) should result in Dark Output

    assert val_highlight_input > val_shadow_input


def test_float_to_uint_dtype_and_layout_variants_match():
    # The no-copy fast path must not change results for f64 or non-contiguous inputs.
    rng = np.random.default_rng(3)
    img = (rng.random((6, 6, 3)) * 1.2 - 0.1).astype(np.float32)  # includes out-of-range

    for fn in (float_to_uint8, float_to_uint16):
        ref = fn(img)
        assert np.array_equal(ref, fn(img.astype(np.float64)))

        wide = np.zeros((6, 12, 3), dtype=np.float32)
        wide[:, ::2, :] = img
        view = wide[:, ::2, :]
        assert not view.flags["C_CONTIGUOUS"]
        assert np.array_equal(ref, fn(view))


def test_float_to_uint_does_not_mutate_input():
    img = np.random.default_rng(5).random((4, 4, 3)).astype(np.float32)
    img_before = img.copy()
    float_to_uint8(img)
    float_to_uint16(img)
    assert np.array_equal(img, img_before)
