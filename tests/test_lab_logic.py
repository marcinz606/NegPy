import unittest
import numpy as np
import cv2
from negpy.kernel.image.logic import lab_to_rgb_working, rgb_to_lab_working
from negpy.features.lab.logic import (
    apply_chroma_denoise,
    apply_clahe,
    apply_glow_and_halation,
    apply_output_sharpening,
    apply_rl_sharpening,
    apply_saturation,
    apply_vibrance,
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

        # CIELAB preserves L* pre-clip; in linear ProPhoto this red (L*≈67) clips
        # toward the gamut edge, ~6 points down — far less than the HSV path.
        self.assertGreaterEqual(l_out, l_in - 8.0)

    def test_vibrance(self) -> None:
        """Vibrance should increase saturation of pale colors more than vibrant ones."""
        # Pale color
        img_pale = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        img_pale[:, :, 0] = 0.6

        # Vibrant color
        img_vibrant = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        img_vibrant[:, :, 0] = 1.0

        res_pale = apply_vibrance(img_pale, 1.5)
        res_vibrant = apply_vibrance(img_vibrant, 1.5)

        # Calculate saturation increase
        def get_sat(rgb):
            c = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            return np.mean(c[:, :, 1])

        sat_gain_pale = get_sat(res_pale) - get_sat(img_pale)
        sat_gain_vibrant = get_sat(res_vibrant) - get_sat(img_vibrant)

        self.assertGreater(sat_gain_pale, sat_gain_vibrant)

    def test_chroma_denoise(self) -> None:
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        lab = rgb_to_lab_working(img)
        lab[:, :, 1] += np.random.normal(0, 5, (100, 100)).astype(np.float32)
        img_noisy = lab_to_rgb_working(lab)

        res = apply_chroma_denoise(img_noisy, radius=2.0)
        res_lab = rgb_to_lab_working(res)

        np.testing.assert_array_almost_equal(lab[:, :, 0], res_lab[:, :, 0], decimal=0)
        self.assertLess(float(np.var(res_lab[:, :, 1])), float(np.var(lab[:, :, 1])))


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
