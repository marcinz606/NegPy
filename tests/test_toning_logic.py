import unittest
import cv2
import numpy as np
from negpy.features.toning.logic import (
    apply_chemical_toning,
    apply_split_toning,
)


class TestChemicalToning(unittest.TestCase):
    """Density-driven chemical toners on the linear print: selenium converts the
    densest silver first (Dmax boost, cool shadows); sepia bleach-redevelop
    converts the thinnest silver first (warm highlights, shadows hold); gold
    plates the finest grain first (cool blue-black, orange-red over sepia)."""

    @staticmethod
    def _gray(t: float) -> np.ndarray:
        return np.full((4, 4, 3), t, dtype=np.float32)

    @staticmethod
    def _density(res: np.ndarray, ch: int) -> float:
        return float(-np.log10(max(float(res[0, 0, ch]), 1e-6)))

    def test_zero_strength_is_identity(self):
        img = np.random.rand(10, 10, 3).astype(np.float32)
        res = apply_chemical_toning(img, selenium_strength=0.0, sepia_strength=0.0)
        np.testing.assert_array_equal(res, img)

    def test_selenium_deepens_shadows(self):
        """Selenium adds density where silver is dense — blacks get deeper."""
        dark = self._gray(0.05)  # D ~ 1.3
        res = apply_chemical_toning(dark, selenium_strength=1.0, sepia_strength=0.0)
        self.assertLess(float(res.mean()), 0.05)

    def test_selenium_converts_densest_first(self):
        """Density gain grows with input density; highlights barely move."""
        d_dark_in, d_light_in = -np.log10(0.05), -np.log10(0.9)
        res_dark = apply_chemical_toning(self._gray(0.05), selenium_strength=1.0, sepia_strength=0.0)
        res_light = apply_chemical_toning(self._gray(0.9), selenium_strength=1.0, sepia_strength=0.0)
        gain_dark = self._density(res_dark, 1) - d_dark_in
        gain_light = self._density(res_light, 1) - d_light_in
        self.assertGreater(gain_dark, gain_light * 10)
        self.assertAlmostEqual(gain_light, 0.0, places=3)

    def test_selenium_cools_shadows(self):
        """Green gains the most density -> magenta/eggplant cast in the shadows."""
        res = apply_chemical_toning(self._gray(0.05), selenium_strength=1.0, sepia_strength=0.0)
        self.assertLess(float(res[0, 0, 1]), float(res[0, 0, 0]))  # G darker than R
        self.assertLess(float(res[0, 0, 1]), float(res[0, 0, 2]))  # G darker than B

    def test_sepia_warms_highlights(self):
        """Converted silver -> warm sulfide dye: red lifts, blue drops."""
        light = self._gray(0.6)
        res = apply_chemical_toning(light, selenium_strength=0.0, sepia_strength=1.0)
        self.assertGreater(float(res[0, 0, 0]), 0.6)  # R lighter (warm)
        self.assertLess(float(res[0, 0, 2]), 0.6)  # B denser

    def test_sepia_converts_thinnest_first(self):
        """Bleach eats the thinnest silver first — highlights tone, shadows hold
        (the classic split-sepia look at partial strength)."""
        res_light = apply_chemical_toning(self._gray(0.6), selenium_strength=0.0, sepia_strength=1.0)
        res_dark = apply_chemical_toning(self._gray(0.01), selenium_strength=0.0, sepia_strength=1.0)
        warmth_light = float(res_light[0, 0, 0] - res_light[0, 0, 2])
        warmth_dark = float(res_dark[0, 0, 0] - res_dark[0, 0, 2])
        self.assertGreater(warmth_light, 0.01)
        self.assertAlmostEqual(warmth_dark, 0.0, places=3)

    def test_gold_cools_and_deepens_highlights(self):
        """Colloidal gold adds density (R most) -> cool blue-black, slight Dmax boost."""
        light = self._gray(0.6)
        res = apply_chemical_toning(light, gold_strength=1.0)
        self.assertLess(float(res[0, 0, 0]), float(res[0, 0, 2]))  # R darker than B (cool)
        self.assertLess(float(res.mean()), 0.6)  # net intensification

    def test_gold_converts_thinnest_first(self):
        """Gold plates the finest grain first — highlights cool, dense shadows hold."""
        res_light = apply_chemical_toning(self._gray(0.6), gold_strength=1.0)
        res_dark = apply_chemical_toning(self._gray(0.01), gold_strength=1.0)
        cool_light = float(res_light[0, 0, 2] - res_light[0, 0, 0])
        cool_dark = float(res_dark[0, 0, 2] - res_dark[0, 0, 0])
        self.assertGreater(cool_light, 0.005)
        self.assertAlmostEqual(cool_dark, 0.0, places=3)

    def test_gold_over_sepia_shifts_red(self):
        """Gold on sulfide (the classic gold-over-sepia combo) pushes the warm
        sepia hue further toward orange-red instead of cooling it."""
        light = self._gray(0.6)
        res_sep = apply_chemical_toning(light, sepia_strength=1.0)
        res_both = apply_chemical_toning(light, sepia_strength=1.0, gold_strength=1.0)
        warmth_sep = float(res_sep[0, 0, 0] - res_sep[0, 0, 2])
        warmth_both = float(res_both[0, 0, 0] - res_both[0, 0, 2])
        self.assertGreater(warmth_sep, 0.01)
        self.assertGreater(warmth_both, warmth_sep)

    def test_gold_monotone_with_strength(self):
        """Longer gold bath -> deeper print, output stays sane."""
        light = self._gray(0.6)
        res_1 = apply_chemical_toning(light, gold_strength=1.0)
        res_2 = apply_chemical_toning(light, gold_strength=2.0)
        self.assertGreaterEqual(float(res_2.min()), 0.0)
        self.assertLessEqual(float(res_2.max()), 1.0)
        self.assertLessEqual(float(res_2.mean()), float(res_1.mean()))

    def test_paper_white_stays_white(self):
        """No silver at paper white — nothing to tone."""
        white = self._gray(1.0)
        res = apply_chemical_toning(white, selenium_strength=1.0, sepia_strength=1.0, gold_strength=1.0)
        np.testing.assert_allclose(res, white, atol=1e-3)

    def test_output_range_combined(self):
        img = np.random.rand(10, 10, 3).astype(np.float32)
        res = apply_chemical_toning(img, selenium_strength=1.0, sepia_strength=1.0)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)

    def test_blue_zero_strength_is_identity(self):
        img = np.random.rand(10, 10, 3).astype(np.float32)
        res = apply_chemical_toning(img, blue_strength=0.0, copper_strength=0.0)
        np.testing.assert_array_equal(res, img)

    def test_blue_deepens_shadows(self):
        """Iron blue deposits ~3x colouring matter per unit silver — intensification.
        The B channel alone lightens (that's the hue), so assert on luma."""
        dark = self._gray(0.05)
        res = apply_chemical_toning(dark, blue_strength=1.0)
        luma = 0.2126 * res[0, 0, 0] + 0.7152 * res[0, 0, 1] + 0.0722 * res[0, 0, 2]
        self.assertLess(float(luma), 0.05)

    def test_blue_cools_image(self):
        """Prussian blue absorbs red most -> blue cast where silver converted."""
        res = apply_chemical_toning(self._gray(0.05), blue_strength=1.0)
        self.assertGreater(float(res[0, 0, 2]), float(res[0, 0, 0]))  # B lighter than R

    def test_blue_converts_proportionally(self):
        """Conversion tracks silver density — shadows tone hardest, but broader
        than selenium. Probe R, the channel Prussian blue absorbs."""
        d_dark_in, d_light_in = -np.log10(0.05), -np.log10(0.6)
        res_dark = apply_chemical_toning(self._gray(0.05), blue_strength=1.0)
        res_light = apply_chemical_toning(self._gray(0.6), blue_strength=1.0)
        gain_dark = self._density(res_dark, 0) - d_dark_in
        gain_light = self._density(res_light, 0) - d_light_in
        self.assertGreater(gain_dark, gain_light * 10)
        self.assertGreater(gain_light, 0.0)

    def test_blue_monotone_with_strength(self):
        """Longer bath -> more silver converted -> more total density (the B
        channel alone lightens, so the claim is densitometric, not reflectance)."""
        dark = self._gray(0.05)
        res_1 = apply_chemical_toning(dark, blue_strength=1.0)
        res_2 = apply_chemical_toning(dark, blue_strength=2.0)
        self.assertGreaterEqual(float(res_2.min()), 0.0)
        self.assertLessEqual(float(res_2.max()), 1.0)
        d_1 = -np.log10(np.clip(res_1, 1e-6, 1.0)).mean()
        d_2 = -np.log10(np.clip(res_2, 1e-6, 1.0)).mean()
        self.assertGreaterEqual(float(d_2), float(d_1))

    def test_blue_visible_in_midtones(self):
        """A blue bath at normal strength colours the mids, not just the blacks
        — regression: a Dmax-referenced d_ref pushed the colour into the deep
        shadows only."""
        res = apply_chemical_toning(self._gray(0.5), blue_strength=1.0)
        self.assertGreater(float(res[0, 0, 2] - res[0, 0, 0]), 0.03)

    def test_copper_visible_in_midtones(self):
        res = apply_chemical_toning(self._gray(0.5), copper_strength=1.0)
        self.assertGreater(float(res[0, 0, 0] - res[0, 0, 2]), 0.03)

    def test_copper_warms_image(self):
        """Copper ferrocyanide lifts red -> pink/brick-red cast."""
        res = apply_chemical_toning(self._gray(0.3), copper_strength=1.0)
        self.assertGreater(float(res[0, 0, 0]), float(res[0, 0, 2]))  # R lighter than B

    def test_copper_loses_dmax(self):
        """The in-bath ferricyanide bleaches while it tones — blacks weaken."""
        black = self._gray(0.01)
        res = apply_chemical_toning(black, copper_strength=1.0)
        self.assertGreater(float(res.mean()), 0.01)

    def test_copper_monotone_with_strength(self):
        mid = self._gray(0.3)
        res_1 = apply_chemical_toning(mid, copper_strength=1.0)
        res_2 = apply_chemical_toning(mid, copper_strength=2.0)
        self.assertGreaterEqual(float(res_2.min()), 0.0)
        self.assertLessEqual(float(res_2.max()), 1.0)
        warmth_1 = float(res_1[0, 0, 0] - res_1[0, 0, 2])
        warmth_2 = float(res_2[0, 0, 0] - res_2[0, 0, 2])
        self.assertGreaterEqual(warmth_2, warmth_1)

    def test_vanadium_greens_image(self):
        """Vanadium/iron green deposit absorbs red most, spares green."""
        res = apply_chemical_toning(self._gray(0.5), vanadium_strength=1.0)
        self.assertGreater(float(res[0, 0, 1]), float(res[0, 0, 0]))  # G lighter than R
        self.assertGreater(float(res[0, 0, 1]), float(res[0, 0, 2]))  # G lighter than B

    def test_vanadium_visible_in_midtones(self):
        res = apply_chemical_toning(self._gray(0.5), vanadium_strength=1.0)
        self.assertGreater(float(res[0, 0, 1] - res[0, 0, 0]), 0.03)

    def test_vanadium_converts_thinnest_first(self):
        """Bleach-then-tone like sepia — highlights and mids green, deep
        shadows keep their black silver (the green-print-with-black-blacks look)."""
        res_mid = apply_chemical_toning(self._gray(0.5), vanadium_strength=1.0)
        res_dark = apply_chemical_toning(self._gray(0.01), vanadium_strength=1.0)
        green_mid = float(res_mid[0, 0, 1] - res_mid[0, 0, 0])
        green_dark = float(res_dark[0, 0, 1] - res_dark[0, 0, 0])
        self.assertGreater(green_mid, 0.03)
        self.assertAlmostEqual(green_dark, 0.0, places=3)

    def test_vanadium_loses_density(self):
        """The bleach component lifts luma where the toner converts."""
        mid = self._gray(0.5)
        res = apply_chemical_toning(mid, vanadium_strength=1.0)
        luma = 0.2126 * res[0, 0, 0] + 0.7152 * res[0, 0, 1] + 0.0722 * res[0, 0, 2]
        self.assertGreater(float(luma), 0.5)

    def test_vanadium_monotone_with_strength(self):
        mid = self._gray(0.5)
        res_1 = apply_chemical_toning(mid, vanadium_strength=1.0)
        res_2 = apply_chemical_toning(mid, vanadium_strength=2.0)
        self.assertGreaterEqual(float(res_2.min()), 0.0)
        self.assertLessEqual(float(res_2.max()), 1.0)
        green_1 = float(res_1[0, 0, 1] - res_1[0, 0, 0])
        green_2 = float(res_2[0, 0, 1] - res_2[0, 0, 0])
        self.assertGreaterEqual(green_2, green_1)

    def test_vanadium_paper_white_stays_white(self):
        white = self._gray(1.0)
        res = apply_chemical_toning(white, vanadium_strength=1.0)
        np.testing.assert_allclose(res, white, atol=1e-3)

    def test_output_range_all_toners(self):
        img = np.random.rand(10, 10, 3).astype(np.float32)
        res = apply_chemical_toning(
            img,
            selenium_strength=1.0,
            sepia_strength=1.0,
            gold_strength=1.0,
            blue_strength=1.0,
            copper_strength=1.0,
            vanadium_strength=1.0,
        )
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)

    def test_paper_white_stays_white_new_toners(self):
        white = self._gray(1.0)
        res = apply_chemical_toning(white, blue_strength=1.0, copper_strength=1.0)
        np.testing.assert_allclose(res, white, atol=1e-3)

    def test_slider_max_saturates_conversion(self):
        """Sliders go to 2.0 — conversion caps at all-silver-toned, output stays
        sane and monotone with strength."""
        dark = self._gray(0.05)
        res_1 = apply_chemical_toning(dark, selenium_strength=1.0, sepia_strength=0.0)
        res_2 = apply_chemical_toning(dark, selenium_strength=2.0, sepia_strength=0.0)
        self.assertGreaterEqual(float(res_2.min()), 0.0)
        self.assertLessEqual(float(res_2.max()), 1.0)
        self.assertLessEqual(float(res_2.mean()), float(res_1.mean()))  # longer bath, deeper


class TestTonerInteractions(unittest.TestCase):
    """Silver-ledger competition: toners share one metallic-silver reservoir,
    converted silver is locked to later baths (Rudman/Ilford). Interactions
    emerge from depletion, not per-pair cross-terms."""

    @staticmethod
    def _gray(t: float) -> np.ndarray:
        return np.full((4, 4, 3), t, dtype=np.float32)

    @staticmethod
    def _warmth(res: np.ndarray) -> float:
        return float(res[0, 0, 0] - res[0, 0, 2])

    @staticmethod
    def _density_warmth(res: np.ndarray) -> float:
        # Sulfide warmth in density space (immune to overall darkening).
        d = -np.log10(np.clip(res[0, 0].astype(np.float64), 1e-6, 1.0))
        return float(d[2] - d[0])

    def test_selenium_protects_shadows_from_sepia(self):
        """Selenized shadow silver resists the sepia bleach — the classic
        archival split: shadow warmth drops sharply as selenium claims the
        dense silver first."""
        dark = self._gray(0.05)
        warmth_sep_only = self._density_warmth(apply_chemical_toning(dark, sepia_strength=1.0))
        warmth_after_sel = self._density_warmth(apply_chemical_toning(dark, selenium_strength=1.0, sepia_strength=1.0))
        self.assertGreater(warmth_sep_only, 0.0)
        self.assertLess(warmth_after_sel, 0.6 * warmth_sep_only)

    def test_full_sepia_starves_blue_in_highlights(self):
        """Complete sepia leaves no metallic silver in the highlights — a
        following blue bath is a near no-op there."""
        light = self._gray(0.7)
        res_sep = apply_chemical_toning(light, sepia_strength=1.5)
        res_both = apply_chemical_toning(light, sepia_strength=1.5, blue_strength=1.0)
        self.assertLess(float(np.abs(res_both - res_sep).max()), 0.005)

    def test_sepia_blue_green_split(self):
        """Partial sepia + blue = the classic green two-bath: warm sulfide
        highlights, blue shadows (Ilford combination table)."""
        res_light = apply_chemical_toning(self._gray(0.7), sepia_strength=1.0, blue_strength=1.0)
        res_dark = apply_chemical_toning(self._gray(0.03), sepia_strength=1.0, blue_strength=1.0)
        self.assertGreater(self._warmth(res_light), 0.01)  # highlights stay warm
        self.assertLess(self._warmth(res_dark), -0.001)  # shadows go blue

    def test_no_double_conversion_density_bound(self):
        """Two full baths cannot each convert 100% of the same silver: the
        density ratio stays inside the single-toner gain envelope."""
        dark = self._gray(0.03)  # D0 ~ 1.52: both baths saturate
        res = apply_chemical_toning(dark, selenium_strength=2.0, blue_strength=2.0)
        d0 = -np.log10(0.03)
        for ch in range(3):
            ratio = -np.log10(max(float(res[0, 0, ch]), 1e-6)) / d0
            self.assertLessEqual(ratio, 1.30 + 1e-3, f"ch{ch}")
            self.assertGreaterEqual(ratio, 0.80 - 1e-3, f"ch{ch}")

    def test_single_toner_output_unchanged(self):
        """Ledger with one bath collapses to the closed form
        D0·(1−c+c·gain) — pins single-toner behavior across the refactor."""
        from negpy.features.toning.logic import TONING_CONSTANTS as C

        shapes = {
            "selenium": ("shadow", C["sel_d_ref"], C["sel_power"], C["sel_gain"]),
            "sepia": ("highlight", C["sep_d_bleach"], C["sep_power"], C["sep_gain"]),
            "gold": ("highlight", C["gold_d_ref"], C["gold_power"], C["gold_gain"]),
            "blue": ("shadow", C["blue_d_ref"], C["blue_power"], C["blue_gain"]),
            "copper": ("shadow", C["copper_d_ref"], C["copper_power"], C["copper_gain"]),
            "vanadium": ("highlight", C["van_d_ref"], C["van_power"], C["van_gain"]),
        }
        ramp = np.linspace(0.001, 1.0, 256, dtype=np.float32)
        img = np.stack([ramp] * 3, axis=-1)[None, :, :]
        d0 = -np.log10(np.clip(ramp.astype(np.float64), 1e-6, 1.0))
        for name, (kind, d_ref, power, gain) in shapes.items():
            for s in (0.5, 1.0, 2.0):
                res = apply_chemical_toning(img, **{f"{name}_strength": s})
                frac = np.minimum(d0 / d_ref, 1.0)
                c = np.minimum(s * (frac if kind == "shadow" else 1.0 - frac) ** power, 1.0)
                for ch in range(3):
                    expected = np.clip(10.0 ** -(d0 * (1.0 - c + c * gain[ch])), 0.0, 1.0)
                    np.testing.assert_allclose(res[0, :, ch], expected, atol=1e-5, err_msg=f"{name}@{s} ch{ch}")

    def test_multi_toner_monotone_no_reversal(self):
        """All six baths maxed: no tone reversal along a gray ramp."""
        ramp = np.linspace(0.001, 1.0, 512, dtype=np.float32)
        img = np.stack([ramp] * 3, axis=-1)[None, :, :]
        res = apply_chemical_toning(
            img,
            selenium_strength=2.0,
            sepia_strength=2.0,
            gold_strength=2.0,
            blue_strength=2.0,
            copper_strength=2.0,
            vanadium_strength=2.0,
        )
        luma = res[0].mean(axis=-1)
        self.assertGreaterEqual(float(np.diff(luma).min()), -1e-4)


class TestSplitToning(unittest.TestCase):
    def test_noop_at_zero_strength(self):
        """Zero strengths → output identical to input."""
        img = np.random.rand(20, 20, 3).astype(np.float32)
        res = apply_split_toning(img, shadow_hue=195.0, shadow_strength=0.0, highlight_hue=30.0, highlight_strength=0.0)
        np.testing.assert_array_almost_equal(img, res)

    def test_shadow_tint_affects_shadows_more_than_highlights(self):
        """Shadow tint should shift chroma in dark pixels more than bright pixels."""
        # Dark pixel (shadow) vs bright pixel (highlight)
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[0:5, :, :] = 0.05  # shadows
        img[5:10, :, :] = 0.95  # highlights

        res = apply_split_toning(img, shadow_hue=0.0, shadow_strength=1.0, highlight_hue=0.0, highlight_strength=0.0)

        lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)

        chroma_change_shadow = np.mean(np.abs(lab_out[0:5, :, 1:] - lab_in[0:5, :, 1:]))
        chroma_change_highlight = np.mean(np.abs(lab_out[5:10, :, 1:] - lab_in[5:10, :, 1:]))

        self.assertGreater(chroma_change_shadow, chroma_change_highlight)

    def test_highlight_tint_affects_highlights_more_than_shadows(self):
        """Highlight tint should shift chroma in bright pixels more than dark pixels."""
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[0:5, :, :] = 0.05  # shadows
        img[5:10, :, :] = 0.95  # highlights

        res = apply_split_toning(img, shadow_hue=0.0, shadow_strength=0.0, highlight_hue=90.0, highlight_strength=1.0)

        lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)

        chroma_change_shadow = np.mean(np.abs(lab_out[0:5, :, 1:] - lab_in[0:5, :, 1:]))
        chroma_change_highlight = np.mean(np.abs(lab_out[5:10, :, 1:] - lab_in[5:10, :, 1:]))

        self.assertGreater(chroma_change_highlight, chroma_change_shadow)

    def test_shadow_hue_direction(self):
        """Hue 0° pushes a* positive (magenta); hue 180° pushes a* negative (green)."""
        img = np.full((10, 10, 3), 0.1, dtype=np.float32)  # dark shadows

        res_magenta = apply_split_toning(img, shadow_hue=0.0, shadow_strength=1.0)
        res_green = apply_split_toning(img, shadow_hue=180.0, shadow_strength=1.0)

        lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_magenta = cv2.cvtColor(res_magenta, cv2.COLOR_RGB2LAB)
        lab_green = cv2.cvtColor(res_green, cv2.COLOR_RGB2LAB)

        # Hue 0° → a* increases (magenta direction)
        self.assertGreater(float(np.mean(lab_magenta[:, :, 1])), float(np.mean(lab_in[:, :, 1])))
        # Hue 180° → a* decreases (green direction)
        self.assertLess(float(np.mean(lab_green[:, :, 1])), float(np.mean(lab_in[:, :, 1])))

    def test_luminance_preserved(self):
        """Split toning should not significantly alter luminance."""
        img = np.random.rand(20, 20, 3).astype(np.float32)

        res = apply_split_toning(img, shadow_hue=195.0, shadow_strength=1.0, highlight_hue=30.0, highlight_strength=1.0)

        lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)

        # L* change should be small (within 3 Lab units on average)
        mean_L_change = float(np.mean(np.abs(lab_out[:, :, 0] - lab_in[:, :, 0])))
        self.assertLess(mean_L_change, 3.0)

    def test_output_range(self):
        """Output should stay in [0, 1]."""
        img = np.random.rand(20, 20, 3).astype(np.float32)
        res = apply_split_toning(img, shadow_hue=195.0, shadow_strength=1.0, highlight_hue=30.0, highlight_strength=1.0)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)

    def test_bw_image_gets_tinted(self):
        """A neutral gray (B&W) image should acquire chroma after split toning."""
        img = np.full((10, 10, 3), 0.1, dtype=np.float32)  # neutral gray shadow

        res = apply_split_toning(img, shadow_hue=195.0, shadow_strength=0.8)

        lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)

        # Chroma (distance from neutral in a*b* plane) should increase
        chroma_in = np.sqrt(lab_in[:, :, 1] ** 2 + lab_in[:, :, 2] ** 2)
        chroma_out = np.sqrt(lab_out[:, :, 1] ** 2 + lab_out[:, :, 2] ** 2)
        self.assertGreater(float(np.mean(chroma_out)), float(np.mean(chroma_in)))


if __name__ == "__main__":
    unittest.main()
