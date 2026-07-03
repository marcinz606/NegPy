"""Physically-based print model: absolute CC filtration, per-channel paper base,
print-dye coupling."""

import unittest

import numpy as np

from negpy.features.exposure.logic import (
    apply_characteristic_curve,
    calculate_wb_shifts_from_log,
    compute_pivot,
    filtration_offsets,
    grade_to_slope,
    paper_dmin_rgb,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.normalization import LogNegativeBounds
from negpy.features.exposure.papers import PaperProfile, resolve_dye_matrix


def _ramp() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    return np.stack([x, x, x], axis=-1)[None, :, :]


def _curve(img, paper=None, d_min=0.0, **kw):
    slope = grade_to_slope(115.0, 1.3)
    pivot = compute_pivot(slope, 1.0, d_min=d_min, paper=paper)
    sp = (pivot, slope)
    return apply_characteristic_curve(img, sp, sp, sp, d_min=d_min, paper=paper, **kw)


class TestAbsoluteFiltration(unittest.TestCase):
    def test_offset_is_range_invariant_density(self):
        # One slider unit = cmy_max_density absolute D, whatever the stretch range.
        cmy_max = EXPOSURE_CONSTANTS["cmy_max_density"]
        for rng in (0.8, 1.3, 2.2):
            bounds = LogNegativeBounds((-rng, -rng, -rng), (0.0, 0.0, 0.0))
            off = filtration_offsets((1.0, 0.5, 0.0), bounds)
            self.assertAlmostEqual(off[0] * rng, cmy_max, places=6)
            self.assertAlmostEqual(off[1] * rng, 0.5 * cmy_max, places=6)
            self.assertEqual(off[2], 0.0)

    def test_no_bounds_falls_back_to_unit_range(self):
        off = filtration_offsets((1.0, 0.0, 0.0), None)
        self.assertAlmostEqual(off[0], EXPOSURE_CONSTANTS["cmy_max_density"], places=6)

    def test_e6_reversed_bounds_keep_direction(self):
        fwd = filtration_offsets((1.0, 1.0, 1.0), LogNegativeBounds((-1.5, -1.5, -1.5), (0.0, 0.0, 0.0)))
        rev = filtration_offsets((1.0, 1.0, 1.0), LogNegativeBounds((0.0, 0.0, 0.0), (-1.5, -1.5, -1.5)))
        self.assertEqual(fwd, rev)

    def test_wb_pick_matches_applied_offset(self):
        # The picker's slider, run back through filtration_offsets, must cancel
        # the sampled normalized deviation exactly.
        bounds = LogNegativeBounds((-1.7, -1.4, -1.1), (0.0, 0.0, 0.0))
        sampled = np.array([0.5, 0.44, 0.58])
        m, y = calculate_wb_shifts_from_log(sampled, bounds)
        off = filtration_offsets((0.0, m, y), bounds)
        self.assertAlmostEqual(sampled[1] + off[1], sampled[0], places=6)
        self.assertAlmostEqual(sampled[2] + off[2], sampled[0], places=6)


class TestPaperBaseTint(unittest.TestCase):
    def test_dmin_rgb_is_tinted_floor(self):
        paper = PaperProfile(label="t", base_tint_cmy=(0.02, 0.0, -0.01), d_min=0.06)
        np.testing.assert_allclose(paper_dmin_rgb(0.06, paper), (0.08, 0.06, 0.05), atol=1e-12)

    def test_paper_dmin_off_kills_tint(self):
        paper = PaperProfile(label="t", base_tint_cmy=(0.02, 0.0, -0.01))
        self.assertEqual(paper_dmin_rgb(0.0, paper), (0.0, 0.0, 0.0))

    def test_tint_shows_in_highlights_fades_in_shadows(self):
        paper = PaperProfile(label="t", base_tint_cmy=(0.04, 0.0, 0.0), d_min=0.06)
        neutral = PaperProfile(label="n", d_min=0.06)
        tinted = np.asarray(_curve(_ramp(), paper=paper, d_min=0.06))
        plain = np.asarray(_curve(_ramp(), paper=neutral, d_min=0.06))
        hi = np.abs(tinted[0, 0] - plain[0, 0])  # thinnest = paper white
        lo = np.abs(tinted[0, -1] - plain[0, -1])  # densest = paper black
        self.assertGreater(float(hi[0]), 5e-3)  # red channel floor lifted
        self.assertLess(float(lo[0]), float(hi[0]) * 0.2)  # fades toward d_max
        np.testing.assert_allclose(tinted[..., 1], plain[..., 1], atol=1e-6)  # untinted channel


class TestDyeCoupling(unittest.TestCase):
    _M = ((0.9, 0.08, 0.02), (0.1, 0.85, 0.05), (0.05, 0.15, 0.8))

    def test_identity_resolves_to_none(self):
        self.assertIsNone(resolve_dye_matrix(PaperProfile(label="n")))
        self.assertIsNone(resolve_dye_matrix(None))

    def test_rows_normalized(self):
        m = resolve_dye_matrix(PaperProfile(label="d", dye_matrix=self._M))
        assert m is not None
        np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-9)

    def test_neutral_ramp_preserved(self):
        # Row normalization: equal channel densities stay equal after mixing.
        dyed = np.asarray(_curve(_ramp(), paper=PaperProfile(label="d", dye_matrix=self._M)))
        plain = np.asarray(_curve(_ramp()))
        np.testing.assert_allclose(dyed, plain, atol=1e-5)

    def test_coloured_input_coupled(self):
        x = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        img = np.stack([x, np.flip(x), np.full_like(x, 0.5)], axis=-1)[None, :, :]
        dyed = np.asarray(_curve(img, paper=PaperProfile(label="d", dye_matrix=self._M)))
        plain = np.asarray(_curve(img))
        self.assertGreater(float(np.max(np.abs(dyed - plain))), 1e-3)


if __name__ == "__main__":
    unittest.main()
