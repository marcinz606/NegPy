import os
import unittest

import imagecodecs
import numpy as np

from negpy.features.exposure.logic import (
    apply_characteristic_curve,
    compute_pivot,
    grade_to_slope,
    per_channel_curve_params,
    slope_to_grade,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.papers import (
    DEFAULT_PROFILE_KEY,
    PAPER_PROFILES,
    _TONAL_KEYS,
    effective_constants,
    effective_paper_profile,
    profiles_for_mode,
    resolve_paper,
    resolve_print_color,
)
from negpy.features.exposure.print_lut import DyeLut
from negpy.features.process.chart import delta_e2000
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE, ColorSpaceRegistry
from negpy.kernel.image.logic import rgb_to_lab_working

# Stocks baked from an ICC look by scripts/build_print_luts.py, each with its profile in tests/fixtures/.
ICC_STOCKS = {"pakon": "RGBScan.icc"}


class TestPaperProfiles(unittest.TestCase):
    """Paper profiles override curve character; the default reproduces the constants."""

    def _ramp(self) -> np.ndarray:
        # 0..1 grayscale ramp as a 1xN x3 image in normalized log space.
        x = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        return np.stack([x, x, x], axis=-1)[None, :, :]

    def _render(self, profile_key: str, img: np.ndarray | None = None, paper_white: bool = True, bpc: bool = False) -> np.ndarray:
        paper = resolve_paper(profile_key)
        d_min = paper.d_min if paper_white else 0.0
        slopes, pivots, _ = per_channel_curve_params(
            grade=115.0,
            density=1.0,
            auto_normalize_contrast=False,
            strength=0.0,
            lum_range=1.3,
            textural_range=None,
            d_min=d_min,
            paper=paper,
        )
        # Base tint + dye coupling ride the paper profile inside the curve.
        return apply_characteristic_curve(
            self._ramp() if img is None else img,
            (pivots[0], slopes[0]),
            (pivots[1], slopes[1]),
            (pivots[2], slopes[2]),
            d_min=d_min,
            paper=paper,
            bpc=bpc,
        )

    def test_registry_well_formed(self):
        self.assertIn(DEFAULT_PROFILE_KEY, PAPER_PROFILES)
        self.assertEqual(PAPER_PROFILES[DEFAULT_PROFILE_KEY].kind, "default")
        kinds = {p.kind for p in PAPER_PROFILES.values()}
        self.assertTrue({"bw", "ra4"} <= kinds)

    def test_midtone_gamma_leaves_snap_headroom(self):
        # Snap and its per-layer trims add to the profile's gamma below the kernel's |gamma| < 1.
        for key, p in PAPER_PROFILES.items():
            self.assertLessEqual(abs(p.paper_midtone_gamma), 0.25, key)

    def test_default_matches_constants(self):
        # The neutral default must leave every tonal key untouched.
        c = effective_constants(resolve_paper(DEFAULT_PROFILE_KEY))
        self.assertIs(c, EXPOSURE_CONSTANTS)
        for k in _TONAL_KEYS:
            self.assertEqual(c[k], EXPOSURE_CONSTANTS[k])

    def test_config_default_resolves(self):
        # A fresh config (and an old save without the key) lands on the default.
        self.assertEqual(ExposureConfig().paper_profile, DEFAULT_PROFILE_KEY)

    def test_unknown_key_falls_back(self):
        self.assertIs(resolve_paper("does-not-exist"), PAPER_PROFILES[DEFAULT_PROFILE_KEY])

    def test_effective_overrides_tonal(self):
        c = effective_constants(PAPER_PROFILES["ilford_fb_classic"])
        self.assertEqual(c["d_max"], PAPER_PROFILES["ilford_fb_classic"].d_max)
        self.assertNotEqual(c["d_max"], EXPOSURE_CONSTANTS["d_max"])

    def test_default_render_unchanged(self):
        # paper=default must render byte-for-byte like passing no profile.
        d_min = EXPOSURE_CONSTANTS["d_min"]
        slope = grade_to_slope(115.0, 1.3)
        pivot = compute_pivot(slope, 1.0, d_min=d_min)
        sp = (pivot, slope)
        base = apply_characteristic_curve(self._ramp(), sp, sp, sp, d_min=d_min)
        prof = self._render(DEFAULT_PROFILE_KEY)
        np.testing.assert_allclose(prof, base, atol=1e-6)

    def test_profile_changes_curve(self):
        # Chromatic, so a stock that keeps Digital's curve and changes only its dyes counts too.
        ramp = self._ramp() * np.array([1.0, 0.8, 0.6], dtype=np.float32)
        default = self._render(DEFAULT_PROFILE_KEY, ramp)
        for key in PAPER_PROFILES:
            if key == DEFAULT_PROFILE_KEY:
                continue
            out = self._render(key, ramp)
            self.assertGreater(float(np.max(np.abs(out - default))), 1e-3, f"{key} did not change the curve")

    def test_icc_stocks_are_digital_with_their_profile_colors(self):
        # The target is the profile applied to a Digital print at the defaults (Paper White off, BPC on),
        # as the export applies an input ICC.
        rng = np.random.default_rng(0)
        img = rng.uniform(0.0, 1.0, (1, 2000, 3)).astype(np.float32)
        digital = self._render(DEFAULT_PROFILE_KEY, img, paper_white=False, bpc=True)[0]
        adobe = open(ColorSpaceRegistry.get_icc_path(WORKING_COLOR_SPACE), "rb").read()
        enc = np.round(np.clip(digital, 0, 1) ** (256 / 563) * 65535).astype(np.uint16)[:, None, :]
        for key, fixture in ICC_STOCKS.items():
            with self.subTest(key):
                icc = open(os.path.join(os.path.dirname(__file__), "fixtures", fixture), "rb").read()
                out = imagecodecs.cms_transform(enc, icc, adobe, colorspace="RGB", outcolorspace="RGB", intent=1, flags=0x2000)
                target = rgb_to_lab_working((out[:, 0, :] / 65535.0) ** (563 / 256))
                miss = delta_e2000(rgb_to_lab_working(self._render(key, img, paper_white=False, bpc=True)[0]), target).mean()
                self.assertLess(miss, 0.1)

    def test_color_stocks_are_luts(self):
        for key, p in PAPER_PROFILES.items():
            with self.subTest(key):
                if p.kind == "ra4":
                    self.assertIsInstance(p.print_color, DyeLut)
                else:
                    self.assertEqual(resolve_print_color(p), (None, None))

    def test_lut_stock_falls_back_off_mode(self):
        for mode in (ProcessMode.BW, ProcessMode.E6):
            self.assertEqual(resolve_print_color(effective_paper_profile("kodak_endura", mode)), (None, None))

    def test_frontier_gain_matches_digital_midtone_span(self):
        # slope_gain prints L* 90 to 10 over Digital's log E span at the same Grade.
        x = np.linspace(-0.5, 1.5, 2001, dtype=np.float32)
        img = np.stack([x, x, x], axis=-1)[None]

        def span(key):
            y = np.maximum(self._render(key, img)[0].mean(-1), 1e-9)
            lstar = 116.0 * np.cbrt(y) - 16.0
            return float(x[np.argmax(lstar <= 10.0)] - x[np.argmax(lstar <= 90.0)])

        self.assertAlmostEqual(span("frontier") / span(DEFAULT_PROFILE_KEY), 1.0, delta=0.05)

    def test_slope_gain_keeps_a_neutral_film_neutral_under_cast_removal(self):
        # Cast Removal matches R and B to green's slope, so the gain applies once to all three layers.
        p = resolve_paper("frontier")
        axis = ((0.5, 0.5, 0.5), (0.8, 0.8, 0.8), (0.2, 0.2, 0.2))
        for strength in (0.0, 1.0):
            slopes, _, _ = per_channel_curve_params(115.0, 1.0, False, strength, 1.3, None, d_min=p.d_min, paper=p, neutral_axis_norm=axis)
            self.assertAlmostEqual(slopes[0], slopes[1], places=5)
            self.assertAlmostEqual(slopes[2], slopes[1], places=5)
        self.assertAlmostEqual(slopes[1], grade_to_slope(115.0, 1.3) * p.slope_gain, places=5)

    def test_slope_to_grade_inverts_a_gained_slope(self):
        self.assertAlmostEqual(slope_to_grade(grade_to_slope(130.0, 1.2, 2.4), 1.2, 2.4), 130.0, places=6)

    def test_crystal_archive_tones_from_frontier(self):
        self.assertGreater(PAPER_PROFILES["fuji_crystal"].d_max, 2.3)
        self.assertIsInstance(PAPER_PROFILES["fuji_crystal_hd"].print_color, DyeLut)
        keys = list(PAPER_PROFILES)
        self.assertEqual(keys[keys.index("fuji_crystal") + 1], "fuji_crystal_hd")
        # Pearl differs from the standard paper only in its surface, which a print stock cannot show.
        self.assertNotIn("fuji_crystal_pearl", PAPER_PROFILES)

    def test_ra4_introduces_color_divergence(self):
        # Per-channel gamma + tint must make a neutral ramp non-grey on RA4 papers.
        out = self._render("kodak_ultra_endura")
        chan_spread = np.max(np.abs(out[..., 0] - out[..., 2]))
        self.assertGreater(float(chan_spread), 1e-3)

    def test_channel_gamma_scales_slopes(self):
        paper = PAPER_PROFILES["kodak_ultra_endura"]
        slopes, _, _ = per_channel_curve_params(115.0, 1.0, False, False, 1.3, None, paper=paper)
        base, _, _ = per_channel_curve_params(115.0, 1.0, False, False, 1.3, None)
        # Blue channel gamma > 1 → steeper blue slope than the neutral baseline.
        self.assertGreater(slopes[2], base[2])

    def test_effective_paper_profile_mode_gating(self):
        neutral = PAPER_PROFILES[DEFAULT_PROFILE_KEY]
        # Matching mode keeps the paper.
        self.assertEqual(effective_paper_profile("kodak_endura", ProcessMode.C41).label, "Kodak Endura Premier")
        self.assertEqual(effective_paper_profile("ilford_mg_rc", ProcessMode.BW).label, "Ilford Multigrade RC")
        # Cross-mode stored value collapses to neutral.
        self.assertIs(effective_paper_profile("ilford_mg_rc", ProcessMode.C41), neutral)
        self.assertIs(effective_paper_profile("kodak_endura", ProcessMode.BW), neutral)
        # E-6 and None force neutral regardless of stored value.
        self.assertIs(effective_paper_profile("kodak_endura", ProcessMode.E6), neutral)
        self.assertIs(effective_paper_profile("ilford_mg_rc", ProcessMode.E6), neutral)
        self.assertIs(effective_paper_profile("kodak_endura", None), neutral)
        # The neutral default is valid in every mode.
        for m in (ProcessMode.C41, ProcessMode.BW, ProcessMode.E6, None):
            self.assertIs(effective_paper_profile(DEFAULT_PROFILE_KEY, m), neutral)

    def test_profiles_for_mode(self):
        c41 = profiles_for_mode(ProcessMode.C41)
        bw = profiles_for_mode(ProcessMode.BW)
        e6 = profiles_for_mode(ProcessMode.E6)
        # Default is always first.
        for lst in (c41, bw, e6):
            self.assertEqual(lst[0][0], DEFAULT_PROFILE_KEY)
        # C41 lists default + RA4 only; B&W default + B&W only; E-6 default only.
        self.assertTrue(all(p.kind == "ra4" for _, p in c41[1:]))
        self.assertTrue(all(p.kind == "bw" for _, p in bw[1:]))
        self.assertEqual(len(e6), 1)
        self.assertEqual(len(c41), 1 + sum(p.kind == "ra4" for p in PAPER_PROFILES.values()))
        self.assertEqual(len(bw), 1 + sum(p.kind == "bw" for p in PAPER_PROFILES.values()))

    def test_every_stock_has_a_distinct_short_name_for_its_strip_band(self):
        for mode in (ProcessMode.C41, ProcessMode.BW):
            shorts = [p.short for _, p in profiles_for_mode(mode)]
            self.assertTrue(all(shorts), mode)
            self.assertEqual(len(set(shorts)), len(shorts), mode)


if __name__ == "__main__":
    unittest.main()
