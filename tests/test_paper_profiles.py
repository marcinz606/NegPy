import unittest

import numpy as np

from negpy.features.exposure.logic import (
    apply_characteristic_curve,
    compute_pivot,
    grade_to_slope,
    per_channel_curve_params,
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
)
from negpy.features.process.models import ProcessMode


class TestPaperProfiles(unittest.TestCase):
    """Paper profiles override curve character; the default reproduces the constants."""

    def _ramp(self) -> np.ndarray:
        # 0..1 grayscale ramp as a 1xN x3 image in normalized log space.
        x = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        return np.stack([x, x, x], axis=-1)[None, :, :]

    def _render(self, profile_key: str) -> np.ndarray:
        paper = resolve_paper(profile_key)
        d_min = paper.d_min  # paper_dmin on
        slopes, pivots, _ = per_channel_curve_params(
            grade=115.0,
            density=1.0,
            auto_normalize_contrast=False,
            strength=0.0,
            lum_range=1.3,
            shadow_refs_norm=None,
            textural_range=None,
            d_min=d_min,
            paper=paper,
        )
        tint = paper.base_tint_cmy
        return apply_characteristic_curve(
            self._ramp(),
            (pivots[0], slopes[0]),
            (pivots[1], slopes[1]),
            (pivots[2], slopes[2]),
            cmy_offsets=tint,
            d_min=d_min,
            paper=paper,
        )

    def test_registry_well_formed(self):
        self.assertIn(DEFAULT_PROFILE_KEY, PAPER_PROFILES)
        self.assertEqual(PAPER_PROFILES[DEFAULT_PROFILE_KEY].kind, "default")
        kinds = {p.kind for p in PAPER_PROFILES.values()}
        self.assertTrue({"bw", "ra4"} <= kinds)

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
        default = self._render(DEFAULT_PROFILE_KEY)
        for key in PAPER_PROFILES:
            if key == DEFAULT_PROFILE_KEY:
                continue
            out = self._render(key)
            self.assertGreater(float(np.max(np.abs(out - default))), 1e-3, f"{key} did not change the curve")

    def test_ra4_introduces_colour_divergence(self):
        # Per-channel gamma + tint must make a neutral ramp non-grey on RA4 papers.
        out = self._render("fuji_crystal")
        chan_spread = np.max(np.abs(out[..., 0] - out[..., 2]))
        self.assertGreater(float(chan_spread), 1e-3)

    def test_channel_gamma_scales_slopes(self):
        paper = PAPER_PROFILES["fuji_crystal"]
        slopes, _, _ = per_channel_curve_params(115.0, 1.0, False, False, 1.3, None, None, paper=paper)
        base, _, _ = per_channel_curve_params(115.0, 1.0, False, False, 1.3, None, None)
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


if __name__ == "__main__":
    unittest.main()
