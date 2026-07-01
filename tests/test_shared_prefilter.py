import unittest

import numpy as np

from negpy.features.exposure.normalization import (
    analyze_log_exposure_bounds,
    analyze_log_exposure_bounds_from_log,
    measure_anchor,
    measure_anchor_from_log,
    measure_neutral_axis,
    measure_neutral_axis_from_log,
    measure_shadow_log_refs,
    measure_shadow_refs_from_log,
    measure_textural_range,
    measure_textural_range_from_log,
    prefilter_log_grid,
)
from negpy.features.process.models import ProcessMode


def _scene(h: int = 900, w: int = 1200) -> np.ndarray:
    """Coloured linear-space negative with per-channel structure and some outliers."""
    rng = np.random.default_rng(11)
    base = np.linspace(0.02, 0.6, h, dtype=np.float32)[:, None]
    img = np.stack(
        [np.repeat(base * s, w, axis=1) for s in (1.0, 0.85, 0.7)],
        axis=-1,
    ).astype(np.float32)
    img += rng.normal(0.0, 0.01, img.shape).astype(np.float32)
    img = np.clip(img, 1e-4, 1.0)
    img[10:15, 10:15, :] = 1.0  # speculars
    img[500:503, 900:903, :] = 1e-5  # dust
    return img


class TestSharedPrefilter(unittest.TestCase):
    """The shared prefilter fed to the *_from_log meters must be bit-exact vs. the
    per-function linear wrappers that each recompute log10 + block-median."""

    def setUp(self) -> None:
        self.img = _scene()
        self.prefiltered = prefilter_log_grid(self.img, None, 0.0)

    def test_bounds_match(self):
        ref = analyze_log_exposure_bounds(self.img, process_mode=ProcessMode.C41)
        got = analyze_log_exposure_bounds_from_log(self.prefiltered, None, 0.0, process_mode=ProcessMode.C41)
        np.testing.assert_allclose(ref.floors, got.floors, rtol=0, atol=1e-6)
        np.testing.assert_allclose(ref.ceils, got.ceils, rtol=0, atol=1e-6)

    def test_shadow_refs_match(self):
        ref = measure_shadow_log_refs(self.img)
        got = measure_shadow_refs_from_log(self.prefiltered, None, 0.0)
        np.testing.assert_allclose(ref, got, rtol=0, atol=1e-6)

    def test_anchor_and_textural_match(self):
        bounds = analyze_log_exposure_bounds(self.img, process_mode=ProcessMode.C41)
        self.assertAlmostEqual(
            measure_anchor(self.img, bounds),
            measure_anchor_from_log(self.prefiltered, bounds, None, 0.0),
            delta=1e-6,
        )
        self.assertAlmostEqual(
            measure_textural_range(self.img),
            measure_textural_range_from_log(self.prefiltered, None, 0.0),
            delta=1e-6,
        )

    def test_neutral_axis_match(self):
        bounds = analyze_log_exposure_bounds(self.img, process_mode=ProcessMode.C41)
        ref = measure_neutral_axis(self.img, bounds)
        got = measure_neutral_axis_from_log(self.prefiltered, bounds, None, 0.0)
        self.assertEqual(ref is None, got is None)
        if ref is not None and got is not None:
            for a, b in zip(ref[:3], got[:3]):
                if a is None:
                    self.assertIsNone(b)
                else:
                    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
            self.assertAlmostEqual(ref[3], got[3], delta=1e-6)

    def test_block_median_reapply_idempotent(self):
        """Re-block-medianing an already-gridded array is a no-op (b<=1 early return) —
        this is what makes feeding a prefiltered grid to the *_from_log meters bit-exact."""
        from negpy.features.exposure.normalization import _block_median_grid

        np.testing.assert_array_equal(self.prefiltered, _block_median_grid(self.prefiltered))


if __name__ == "__main__":
    unittest.main()
