import unittest
import numpy as np
import cv2
from negpy.kernel.image.logic import _skin_weight, lab_to_rgb_working, rgb_to_lab_working, skin_chroma_rein
from negpy.features.lab.models import LabConfig
from negpy.features.lab.logic import (
    apply_chroma_denoise,
    apply_clahe,
    apply_glow_and_halation,
    apply_output_sharpening,
    apply_rl_sharpening,
    apply_saturation,
    gaussian_kernel_1d,
    rl_iterations,
)


class TestLabLogic(unittest.TestCase):
    def test_spectral_crosstalk(self) -> None:
        """Matrix should mix channels (op now lives in normalization, capture-side)."""
        from negpy.features.exposure.normalization import resolve_crosstalk_matrix, unmix_log_image

        def apply(img, strength, matrix):
            return unmix_log_image(img, resolve_crosstalk_matrix(strength, tuple(matrix)))

        img = np.array([[[1.0, 0.5, 0.0]]], dtype=np.float32)
        # Identity matrix
        matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        res = apply(img, 1.0, matrix)
        assert np.allclose(res, img)

        # Swap R and G
        matrix_swap = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        res_swap = apply(img, 1.0, matrix_swap)
        assert np.allclose(res_swap[0, 0], [0.5, 1.0, 0.0])

    def test_clahe(self) -> None:
        """CLAHE should modify image."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        res = apply_clahe(img, 1.0)
        assert res.shape == img.shape
        # Should be different
        assert not np.allclose(res, img)

    def test_clahe_zero_strength_passthrough(self) -> None:
        img = np.random.rand(32, 32, 3).astype(np.float32)
        assert apply_clahe(img, 0.0) is img

    def test_clahe_flat_image_near_identity(self) -> None:
        """Clipping redistributes a constant image's single-bin mass, so the CDF
        approximates the identity ramp (cdf[b] ≈ (b+1)/256 + limit/total). Needs
        realistic tile sizes: 256x256 → 32x32 tiles, like the 200px preview tiles."""
        img = np.full((256, 256, 3), 0.35, dtype=np.float32)
        res = apply_clahe(img, 1.0)
        np.testing.assert_allclose(res, img, atol=0.02)

    def test_clahe_cdf_invariants(self) -> None:
        """Per-tile CDFs are monotone and end exactly at 1.0 — the excess
        redistribution conserves the tile total, mirroring clahe_cdf.wgsl."""
        from negpy.features.lab.logic import _clahe_cdfs

        rng = np.random.default_rng(7)
        bins = rng.integers(0, 256, (128, 128)).astype(np.int32)
        cdfs = _clahe_cdfs(bins, 2.5)
        self.assertEqual(cdfs.shape, (64, 256))
        self.assertTrue(np.all(np.diff(cdfs, axis=1) >= 0))
        self.assertTrue(np.all(cdfs[:, -1] == 1.0))

    def test_output_sharpening(self) -> None:
        """Sharpening should increase local variance."""
        # Create a simple square
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[25:75, 25:75, :] = 0.5

        res = apply_output_sharpening(img, amount=1.0, scale_factor=1.0)

        # Sharpening should increase variance on edges
        self.assertGreater(np.var(res), np.var(img))

    def test_gaussian_kernel_invariants(self) -> None:
        """Shared CPU/GPU taps: normalised, symmetric, radius = ceil(2.5σ), capped."""
        from negpy.features.lab.logic import gaussian_kernel_1d

        for sigma, expected_r in ((0.5, 2), (1.0, 3), (3.75, 10), (45.0, 113)):
            k = gaussian_kernel_1d(sigma)
            self.assertEqual((len(k) - 1) // 2, expected_r)
            self.assertEqual(k.dtype, np.float32)
            self.assertAlmostEqual(float(k.sum()), 1.0, places=5)
            np.testing.assert_allclose(k, k[::-1])
        self.assertEqual(len(gaussian_kernel_1d(1000.0)), 511)

    def test_sharpen_no_overshoot_on_step(self) -> None:
        """Halo suppression: a hard step must stay within the local range plus
        the (+1 light / -2 dark) overshoot tolerances in L*."""
        img = np.zeros((40, 40, 3), dtype=np.float32)
        img[:, 20:] = 0.8

        res = apply_output_sharpening(img, amount=1.0, scale_factor=1.0)

        l_in = rgb_to_lab_working(img)[..., 0]
        l_out = rgb_to_lab_working(res.astype(np.float32))[..., 0]
        self.assertGreaterEqual(float(l_out.min()), float(l_in.min()) - 2.0 - 0.1)
        self.assertLessEqual(float(l_out.max()), float(l_in.max()) + 1.0 + 0.1)

    def test_sharpen_flat_below_gate_passthrough(self) -> None:
        """L* diffs under the noise gate must not be amplified."""
        rng = np.random.default_rng(3)
        img = np.clip(0.5 + rng.normal(0, 0.001, (64, 64, 3)), 0.0, 1.0).astype(np.float32)

        res = apply_output_sharpening(img, amount=1.0, scale_factor=1.0)

        l_in = rgb_to_lab_working(img)[..., 0]
        l_out = rgb_to_lab_working(res.astype(np.float32))[..., 0]
        np.testing.assert_allclose(l_out, l_in, atol=0.05)

    def test_sharpen_masking_protects_flat_texture(self) -> None:
        """masking=1 suppresses grain amplification in flat areas while the
        strong edge still sharpens."""
        rng = np.random.default_rng(5)
        img = np.zeros((64, 64, 3), dtype=np.float32)
        img[:, :32] = 0.2
        img[:, 32:] = 0.8
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0.0, 1.0).astype(np.float32)

        res_open = apply_output_sharpening(img, amount=1.0, scale_factor=1.0, masking=0.0)
        res_masked = apply_output_sharpening(img, amount=1.0, scale_factor=1.0, masking=1.0)

        l_in = rgb_to_lab_working(img)[..., 0]
        l_open = rgb_to_lab_working(res_open.astype(np.float32))[..., 0]
        l_masked = rgb_to_lab_working(res_masked.astype(np.float32))[..., 0]

        flat = np.s_[8:56, 8:24]
        edge = np.s_[8:56, 30:34]
        self.assertLess(
            float(np.abs(l_masked[flat] - l_in[flat]).mean()),
            float(np.abs(l_open[flat] - l_in[flat]).mean()),
        )
        self.assertGreater(float(np.abs(l_masked[edge] - l_in[edge]).max()), 0.5)

    def test_rl_iterations_bounds(self) -> None:
        """Deterministic iteration count from radius, clamped to [5, 20]."""
        self.assertEqual(rl_iterations(0.5), 5)
        self.assertEqual(rl_iterations(1.0), 10)
        self.assertEqual(rl_iterations(3.0), 20)

    def _luminance(self, img: np.ndarray) -> np.ndarray:
        return img[..., 0] * 0.2973769 + img[..., 1] * 0.6273491 + img[..., 2] * 0.0752741

    def test_rl_recovers_blurred_edge(self) -> None:
        """RL deconvolution of a Gaussian-blurred step moves luminance closer to
        the sharp step than the blurred input."""
        img = np.zeros((40, 40, 3), dtype=np.float32)
        img[:, 20:] = 0.7
        k = gaussian_kernel_1d(1.0)
        blurred = np.stack(
            [cv2.sepFilter2D(img[..., c], -1, k, k, borderType=cv2.BORDER_REFLECT_101) for c in range(3)],
            axis=-1,
        ).astype(np.float32)

        res = apply_rl_sharpening(blurred, amount=1.0, scale_factor=1.0, radius=1.0)

        step_y, blur_y, res_y = self._luminance(img), self._luminance(blurred), self._luminance(res.astype(np.float32))
        self.assertLess(float(np.abs(res_y - step_y).mean()), float(np.abs(blur_y - step_y).mean()))

    def test_rl_masking_protects_flat_texture(self) -> None:
        """masking=1 suppresses grain amplification in flat areas; the edge still sharpens."""
        rng = np.random.default_rng(5)
        img = np.zeros((64, 64, 3), dtype=np.float32)
        img[:, :32] = 0.2
        img[:, 32:] = 0.8
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0.0, 1.0).astype(np.float32)

        res_open = apply_rl_sharpening(img, amount=1.0, scale_factor=1.0, radius=1.0, masking=0.0)
        res_masked = apply_rl_sharpening(img, amount=1.0, scale_factor=1.0, radius=1.0, masking=1.0)

        y_in, y_open, y_masked = (
            self._luminance(img),
            self._luminance(res_open.astype(np.float32)),
            self._luminance(res_masked.astype(np.float32)),
        )
        flat = np.s_[8:56, 8:24]
        self.assertLess(
            float(np.abs(y_masked[flat] - y_in[flat]).mean()),
            float(np.abs(y_open[flat] - y_in[flat]).mean()),
        )

    def test_rl_preserves_chroma(self) -> None:
        """RGB-ratio apply keeps hue: channel cross-products are unchanged."""
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[:, :5] = [0.6, 0.2, 0.1]
        img[:, 5:] = [0.1, 0.5, 0.3]

        res = apply_rl_sharpening(img, amount=1.0, scale_factor=1.0, radius=1.0)

        mask = img.min(axis=-1) > 0.01
        cross = np.abs(res[..., 0] * img[..., 1] - res[..., 1] * img[..., 0])
        self.assertLess(float(cross[mask].max()), 1e-4)

    def test_saturation(self) -> None:
        """Saturation scales chroma in CIELAB — preserves L*, no V-style darkening."""
        # Pure Red (1, 0, 0). L* measured in the working space (Adobe RGB CIELAB).
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[:, :, 0] = 1.0
        l_input = rgb_to_lab_working(img)[0, 0, 0]

        # Desaturate fully → mid-gray (R≈G≈B) at the same L*.
        desat = apply_saturation(img, 0.0)
        r, g, b = float(desat[0, 0, 0]), float(desat[0, 0, 1]), float(desat[0, 0, 2])
        self.assertAlmostEqual(r, g, delta=1e-3)
        self.assertAlmostEqual(g, b, delta=1e-3)
        # Midtone gray, not white. Linear output: pure red's Adobe Y≈0.30.
        self.assertLess(r, 0.5)
        self.assertGreater(r, 0.2)
        l_desat = rgb_to_lab_working(desat)[0, 0, 0]
        self.assertAlmostEqual(float(l_desat), float(l_input), delta=1.0)

        # Saturate pale red (0.8, 0.5, 0.5) × 2.0 → still red-dominant, L* preserved
        # (in-gamut input chosen so the result doesn't hit per-channel sRGB clip).
        img2 = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        img2[:, :, 0] = 0.8
        l_input2 = rgb_to_lab_working(img2)[0, 0, 0]

        sat = apply_saturation(img2, 2.0)
        r2, g2, b2 = float(sat[0, 0, 0]), float(sat[0, 0, 1]), float(sat[0, 0, 2])
        self.assertGreater(r2, g2)
        self.assertGreater(r2, b2)
        l_sat = rgb_to_lab_working(sat)[0, 0, 0]
        self.assertAlmostEqual(float(l_sat), float(l_input2), delta=2.0)

    def test_saturation_does_not_darken_saturated_red(self) -> None:
        """Regression for #193: boosting saturation must not drop perceived lightness L*."""
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[:, :, 0] = 0.9
        img[:, :, 1] = 0.15
        img[:, :, 2] = 0.1

        l_in = float(rgb_to_lab_working(img)[0, 0, 0])
        boosted = apply_saturation(img, 1.5)
        l_out = float(rgb_to_lab_working(boosted)[0, 0, 0])

        # CIELAB preserves L* pre-clip; the gamut-aware knee (see
        # test_saturation_is_gamut_aware below) keeps this red (L*≈67) closer
        # to the gamut edge than a hard per-channel clamp -- measured ~4.7
        # points down, vs. the ~6 a naive flat scale + clip produces.
        self.assertGreaterEqual(l_out, l_in - 5.5)

    def test_saturation_is_gamut_aware(self) -> None:
        """A flat a*/b* scale preserves hue mathematically, but a hard per-channel
        RGB clamp afterward shifts the *actual* hue by changing the R:G:B ratio
        unevenly. The gamut-aware knee should avoid that: pushing hard on a
        deeply saturated color should land much closer to its true (unclipped)
        hue than a naive flat scale + clamp does -- specifically on the pixels
        that actually clip under the naive approach. The other ~90%+ of any
        random population never clips either way and both methods produce
        identical output for it (see test_saturation_unaffected_when_
        comfortably_in_gamut), so mixing that population in just dilutes the
        comparison with ~2.6deg of shared RGB<->Lab round-trip measurement
        noise -- not a real signal either way, and enough to make an
        aggregate-over-everything assertion misleadingly weak."""
        rng = np.random.default_rng(3)
        n = 2000
        angles = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        chroma = rng.uniform(20, 55, n).astype(np.float32)
        l_vals = rng.uniform(20, 80, n).astype(np.float32)
        a = chroma * np.cos(angles)
        b = chroma * np.sin(angles)
        lab = np.stack([l_vals, a, b], axis=-1).astype(np.float32)
        rgb = np.clip(lab_to_rgb_working(lab), 0.0, 1.0).astype(np.float32).reshape(-1, 1, 3)

        def naive_flat_scale(img, saturation):
            lab = rgb_to_lab_working(img)
            res = lab.copy()
            res[..., 1] *= saturation
            res[..., 2] *= saturation
            raw = lab_to_rgb_working(res)
            clipped = ((raw < -1e-4) | (raw > 1.0 + 1e-4)).any(axis=-1).reshape(-1)
            return np.clip(raw, 0.0, 1.0), clipped

        def hue_error(out, a_in, b_in):
            # out is (n, 1, 3); flatten the middle axis so this lines up elementwise
            # against the (n,) a_in/b_in instead of broadcasting into an (n, n) mess.
            lab_out = rgb_to_lab_working(out.astype(np.float32)).reshape(-1, 3)
            hue_actual = np.arctan2(lab_out[:, 2], lab_out[:, 1])
            hue_intended = np.arctan2(b_in, a_in)
            return np.degrees(np.abs(np.angle(np.exp(1j * (hue_actual - hue_intended)))))

        saturation = 1.6
        gamut_aware_out = apply_saturation(rgb, saturation)
        naive_out, naive_clipped = naive_flat_scale(rgb, saturation)
        self.assertGreater(naive_clipped.sum(), 50)  # sanity: this population must actually exercise clipping

        gamut_aware_err = hue_error(gamut_aware_out, a, b)[naive_clipped].mean()
        naive_err = hue_error(naive_out, a, b)[naive_clipped].mean()

        # On pixels that actually clip under the naive approach, the gamut-aware
        # version must be meaningfully better, not just different -- measured
        # ~1.5deg vs. ~7.3deg (about 5x) on this seed; assert a safe fraction
        # of that margin rather than the exact measured ratio.
        self.assertLess(gamut_aware_err, naive_err / 3.0)

    def test_saturation_unaffected_when_comfortably_in_gamut(self) -> None:
        """A gentle push on a color nowhere near the gamut edge -- and nowhere
        near the skin-protection hue band -- should behave identically to a
        flat a*/b* scale -- the knee should only engage near the boundary, not
        change everyday, moderate saturation adjustments."""
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        img[:, :, 2] = 0.55  # mild, safely in-gamut blue bias -- hue ~291deg,
        # ~121deg from the skin-protection center (52deg), so its protection
        # weight is ~1e-5, negligible enough that this isolates gamut-awareness
        # cleanly, same as before skin protection existed.

        gamut_aware = apply_saturation(img, 1.1)

        lab = rgb_to_lab_working(img)
        lab[..., 1] *= 1.1
        lab[..., 2] *= 1.1
        naive = np.clip(lab_to_rgb_working(lab), 0.0, 1.0)

        np.testing.assert_allclose(gamut_aware, naive, atol=1e-4)

    def test_saturation_below_one_unchanged(self) -> None:
        """Desaturating (saturation < 1.0) can't push a pixel further out of
        gamut than it already was, so the knee only applies above 1.0 -- this
        must stay a plain flat scale, unchanged from before."""
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[:, :, 0] = 0.9
        img[:, :, 1] = 0.15
        img[:, :, 2] = 0.1

        gamut_aware = apply_saturation(img, 0.3)

        lab = rgb_to_lab_working(img)
        lab[..., 1] *= 0.3
        lab[..., 2] *= 0.3
        naive = np.clip(lab_to_rgb_working(lab), 0.0, 1.0)

        np.testing.assert_allclose(gamut_aware, naive, atol=1e-5)

    def test_chroma_denoise(self) -> None:
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        lab = rgb_to_lab_working(img)
        lab[:, :, 1] += np.random.normal(0, 5, (100, 100)).astype(np.float32)
        img_noisy = lab_to_rgb_working(lab)

        res = apply_chroma_denoise(img_noisy, radius=2.0)
        res_lab = rgb_to_lab_working(res)

        np.testing.assert_array_almost_equal(lab[:, :, 0], res_lab[:, :, 0], decimal=0)
        self.assertLess(float(np.var(res_lab[:, :, 1])), float(np.var(lab[:, :, 1])))

    def test_chroma_denoise_does_not_bleed_across_edge(self) -> None:
        """A saturated region must not tint its neighbours — an isotropic blur haloed."""
        rng = np.random.default_rng(0)
        img = np.full((120, 120, 3), 0.5, dtype=np.float32)
        lab = rgb_to_lab_working(img)
        lab[:, :, 1] += rng.normal(0, 5, (120, 120)).astype(np.float32)
        lab[:, 60:, 1] += 40.0  # saturated block on the right half
        noisy = lab_to_rgb_working(lab)

        res_lab = rgb_to_lab_working(apply_chroma_denoise(noisy, radius=5.0))

        # Clean side, right up against the edge: must stay near its own a*, not pick up
        # the block's +40. The old GaussianBlur contaminated this band by ~18 units.
        halo = float(np.abs(res_lab[:, 48:60, 1].mean(axis=0) - lab[:, 48:60, 1].mean(axis=0)).max())
        self.assertLess(halo, 3.0)
        # Still denoises well away from the edge.
        self.assertLess(float(np.var(res_lab[:, :40, 1])), float(np.var(lab[:, :40, 1])))


def _lab(l_val: float, chroma: float, hue_deg: float) -> np.ndarray:
    rad = np.radians(hue_deg)
    return np.array([[l_val, chroma * np.cos(rad), chroma * np.sin(rad)]], dtype=np.float32)


def _chroma(lab: np.ndarray) -> float:
    return float(np.hypot(lab[0, 1], lab[0, 2]))


class TestSkinMask(unittest.TestCase):
    def test_skin_patch_scores_high(self) -> None:
        """Mid-lightness, moderate-chroma warm hue -- the middle of the skin locus."""
        self.assertGreater(_skin_weight(65.0, *_lab(65.0, 28.0, 52.0)[0, 1:]), 0.9)

    def test_saturated_red_scores_near_zero(self) -> None:
        """The regression the hue-only mask couldn't catch: pure red sits ~40deg
        in this working space, close enough for a hue Gaussian to damp it like a
        face, but at chroma ~104 it is far outside the skin locus."""
        lab = rgb_to_lab_working(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))[0, 0]
        self.assertGreater(np.hypot(lab[1], lab[2]), 85.0)
        self.assertLess(_skin_weight(float(lab[0]), float(lab[1]), float(lab[2])), 0.01)

    def test_deep_skin_scores_high(self) -> None:
        """Deep skin is low-chroma and dark but sits at the same hue -- the chroma
        window must not be tightened to where it drops out."""
        self.assertGreater(_skin_weight(27.0, *_lab(27.0, 22.0, 53.0)[0, 1:]), 0.9)

    def test_saturated_warm_objects_score_low(self) -> None:
        """Sunset, terracotta, brick and autumn colour all sit inside the skin hue
        band -- only the chroma window keeps them out. The loose window this
        replaced scored them 0.98 / 0.96 / 0.89 / 0.47, reining a sunset harder
        than a face; they now measure 0.04 / 0.18 / 0.25 / 0.00."""
        for name, (l_val, chroma, hue) in {
            "sunset": (71.0, 57.0, 55.0),
            "terracotta": (55.0, 53.0, 45.0),
            "brick": (39.0, 51.0, 40.0),
            "autumn leaf": (53.0, 71.0, 54.0),
            "rust": (44.0, 69.0, 48.0),
        }.items():
            with self.subTest(name):
                self.assertLess(_skin_weight(l_val, *_lab(l_val, chroma, hue)[0, 1:]), 0.3)

    def test_neutral_and_shadow_score_zero(self) -> None:
        self.assertEqual(_skin_weight(50.0, 0.3, 0.4), 0.0)
        self.assertEqual(_skin_weight(0.0, 20.0, 25.0), 0.0)

    def test_cool_hue_scores_zero(self) -> None:
        self.assertLess(_skin_weight(65.0, *_lab(65.0, 28.0, 250.0)[0, 1:]), 1e-3)


class TestSkinChromaRein(unittest.TestCase):
    def test_zero_strength_is_identity(self) -> None:
        lab = _lab(60.0, 70.0, 52.0)
        np.testing.assert_array_equal(skin_chroma_rein(lab, 0.0), lab)

    def test_never_raises_chroma(self) -> None:
        for chroma in (5.0, 20.0, 40.0, 60.0, 80.0, 100.0):
            for hue in (0.0, 52.0, 120.0, 250.0):
                lab = _lab(60.0, chroma, hue)
                self.assertLessEqual(_chroma(skin_chroma_rein(lab, 1.0)), chroma + 1e-4)

    def test_below_the_knee_untouched(self) -> None:
        """Ordinary skin chroma passes through: at strength 0.5 the ceiling is
        44 and the knee starts at 26.4, so a C* of 20 is left alone."""
        lab = _lab(65.0, 20.0, 52.0)
        np.testing.assert_allclose(skin_chroma_rein(lab, 0.5), lab, atol=1e-5)

    def test_excessive_skin_chroma_is_pulled_down(self) -> None:
        lab = _lab(65.0, 45.0, 52.0)
        self.assertLess(_chroma(skin_chroma_rein(lab, 0.5)), 43.0)

    def test_hue_and_lightness_preserved(self) -> None:
        lab = _lab(65.0, 45.0, 52.0)
        out = skin_chroma_rein(lab, 0.8)
        self.assertAlmostEqual(float(out[0, 0]), 65.0, places=4)
        self.assertAlmostEqual(float(np.degrees(np.arctan2(out[0, 2], out[0, 1]))), 52.0, places=3)

    def test_stronger_reins_harder(self) -> None:
        lab = _lab(65.0, 45.0, 52.0)
        chromas = [_chroma(skin_chroma_rein(lab, s)) for s in (0.2, 0.5, 0.8, 1.0)]
        self.assertEqual(chromas, sorted(chromas, reverse=True))


class TestSkinProtectionInSaturation(unittest.TestCase):
    def test_acts_at_chroma_one(self) -> None:
        """The point of the control: protection works with Chroma at 1.0, where
        the scale itself is a no-op."""
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[:, :, 0] = 0.53
        img[:, :, 1] = 0.27
        img[:, :, 2] = 0.16  # skin at L* 65, C* 35, 52deg -- above the knee, mask weight 1.0

        before = _chroma(rgb_to_lab_working(img)[0, :1])
        after = _chroma(rgb_to_lab_working(apply_saturation(img, 1.0, 0.8))[0, :1])

        self.assertLess(after, before - 1.0)

    def test_on_by_default(self) -> None:
        """Ships on at half strength -- a look decision, so pin it."""
        self.assertEqual(LabConfig().skin_protection, 0.5)

        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[:, :, 0] = 0.53
        img[:, :, 1] = 0.27
        img[:, :, 2] = 0.16  # skin, C* 35 -- above the knee at strength 0.5

        before = _chroma(rgb_to_lab_working(img)[0, :1])
        after = _chroma(rgb_to_lab_working(apply_saturation(img, 1.0, LabConfig().skin_protection))[0, :1])

        self.assertLess(after, before - 1.0)

    def test_off_is_identity_at_chroma_one(self) -> None:
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        img[:, :, 0] = 0.85
        np.testing.assert_array_equal(apply_saturation(img, 1.0, 0.0), img)

    def test_desaturation_still_reaches_grey(self) -> None:
        """Protection only ever removes chroma, so asking for zero still means
        zero -- no special case needed, unlike the boost-damping it replaced."""
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[:, :, 0] = 1.0  # pure red, hue ~40deg -- inside the skin hue band

        desat = apply_saturation(img, 0.0, 1.0)
        r, g, b = float(desat[0, 0, 0]), float(desat[0, 0, 1]), float(desat[0, 0, 2])
        self.assertAlmostEqual(r, g, delta=1e-3)
        self.assertAlmostEqual(g, b, delta=1e-3)


class TestGlowAndHalation(unittest.TestCase):
    def _highlight_image(self) -> np.ndarray:
        """100x100 image with a bright white spot in the centre on a dark background."""
        img = np.full((100, 100, 3), 0.1, dtype=np.float32)
        img[40:60, 40:60, :] = 1.0
        return img

    def test_noop_when_both_zero(self) -> None:
        """No change when both amounts are 0.0."""
        img = self._highlight_image()
        res = apply_glow_and_halation(img, glow_amount=0.0, halation_strength=0.0)
        np.testing.assert_array_equal(res, img)

    def test_output_shape_and_range(self) -> None:
        """Output keeps the same shape and stays in [0, 1]."""
        img = self._highlight_image()
        for glow, hal in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            res = apply_glow_and_halation(img, glow, hal)
            self.assertEqual(res.shape, img.shape)
            self.assertGreaterEqual(float(res.min()), 0.0)
            self.assertLessEqual(float(res.max()), 1.0)

    def test_glow_brightens_dark_area_near_highlight(self) -> None:
        """Glow should increase brightness in the dark area neighbouring the highlight."""
        img = self._highlight_image()
        res = apply_glow_and_halation(img, glow_amount=1.0, halation_strength=0.0)
        # Dark border just outside the bright spot should be brighter after glow
        dark_before = float(img[35, 35, 0])
        dark_after = float(res[35, 35, 0])
        self.assertGreater(dark_after, dark_before)

    def test_glow_all_channels_equally(self) -> None:
        """Glow bloom should be approximately equal across R, G, B channels."""
        img = self._highlight_image()
        res = apply_glow_and_halation(img, glow_amount=1.0, halation_strength=0.0)
        # Check a dark pixel near the highlight
        delta = res[30, 50] - img[30, 50]
        # All three channels should have gained roughly the same amount
        self.assertAlmostEqual(float(delta[0]), float(delta[1]), delta=0.05)
        self.assertAlmostEqual(float(delta[1]), float(delta[2]), delta=0.05)

    def test_halation_red_dominant(self) -> None:
        """Halation scatter should add more red than blue to dark pixels near highlights."""
        img = self._highlight_image()
        res = apply_glow_and_halation(img, glow_amount=0.0, halation_strength=1.0)
        delta = res[30, 50] - img[30, 50]
        self.assertGreater(float(delta[0]), float(delta[2]))

    def test_scale_factor_affects_spread(self) -> None:
        """A larger scale factor should spread the bloom further from the highlight."""
        img = self._highlight_image()
        res_small = apply_glow_and_halation(img, glow_amount=1.0, halation_strength=0.0, scale_factor=0.5)
        res_large = apply_glow_and_halation(img, glow_amount=1.0, halation_strength=0.0, scale_factor=2.0)
        # scale=0.5 → kernel radius ~7px; scale=2.0 → kernel radius ~30px.
        # Pixel at row 28 is ~12px above the highlight edge (row 40), so it should
        # receive bloom with scale=2.0 but not with scale=0.5.
        far_small = float(res_small[28, 50, 0])
        far_large = float(res_large[28, 50, 0])
        self.assertGreater(far_large, far_small)

    def test_halation_ignores_midtones(self) -> None:
        """Mid-gray (0.5 linear) must not halate: the mask thresholds linear
        reflectance (0.65), not display code — the old encoded-domain mask lit up
        anything above ~0.29 linear and moved with grade/density."""
        img = np.full((64, 64, 3), 0.5, dtype=np.float32)
        res = apply_glow_and_halation(img, glow_amount=0.0, halation_strength=1.0)
        np.testing.assert_allclose(res, img, atol=1e-6)

    def test_halation_energy_conserving(self) -> None:
        """Additive scatter cannot add more light than the masked highlight source."""
        from negpy.features.lab.logic import HALATION_THRESHOLD_LINEAR as t

        img = self._highlight_image()
        res = apply_glow_and_halation(img, glow_amount=0.0, halation_strength=1.0)
        lin_luma = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
        mask = np.clip((lin_luma - t) / (1.0 - t), 0.0, 1.0) ** 2
        source_energy = float(np.sum(img[:, :, 0] * mask)) * (1.0 + 0.3 + 0.05)
        added = float(np.sum(res - img))
        self.assertLessEqual(added, source_energy + 1e-3)
        self.assertGreater(added, 0.0)

    def test_combined_brighter_than_individual(self) -> None:
        """Applying both glow and halation should be at least as bright as either alone."""
        img = self._highlight_image()
        res_glow = apply_glow_and_halation(img, glow_amount=0.5, halation_strength=0.0)
        res_hal = apply_glow_and_halation(img, glow_amount=0.0, halation_strength=0.5)
        res_both = apply_glow_and_halation(img, glow_amount=0.5, halation_strength=0.5)
        self.assertGreaterEqual(float(res_both[30, 50, 0]), float(res_glow[30, 50, 0]))
        self.assertGreaterEqual(float(res_both[30, 50, 0]), float(res_hal[30, 50, 0]))


if __name__ == "__main__":
    unittest.main()
