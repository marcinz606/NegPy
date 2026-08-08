"""The preview soft proof rides a cached 3D LUT instead of a per-frame littleCMS transform.

The LUT is built by pushing the identity grid through ``soft_proof_preview`` itself,
so it cannot drift from the branch (print profile / export space / GRAY) it stands in
for. Export is unaffected — it keeps the exact per-pixel transform.
"""

import unittest

import numpy as np

from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry, WORKING_COLOR_SPACE
from negpy.infrastructure.display.icc_lut import apply_lut_f32
from negpy.services.rendering.image_processor import PROOF_LUT_SIZE, ImageProcessor

# Fine enough to stand in for the continuous transform when scoring the shipped grid.
_TRUTH_LUT_SIZE = 129


def _srgb_path() -> str:
    return ColorSpaceRegistry.get_icc_path("sRGB")


def _proof_lut(out_path, size, monitor=None):
    ImageProcessor.soft_proof_lut.cache_clear()
    return ImageProcessor.soft_proof_lut(WORKING_COLOR_SPACE, None, out_path, monitor, size)


class TestSoftProofLut(unittest.TestCase):
    def setUp(self):
        ImageProcessor.soft_proof_lut.cache_clear()
        rng = np.random.default_rng(0)
        self.img = rng.random((48, 64, 3), dtype=np.float32)

    def _per_pixel(self, img, out_path, monitor=None):
        from PIL import Image

        u8 = np.clip(img, 0.0, 1.0)
        pil = Image.fromarray((u8 * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        proofed = ImageProcessor.soft_proof_preview(pil, WORKING_COLOR_SPACE, None, out_path, monitor)
        return np.asarray(proofed, dtype=np.float32) / 255.0

    def test_shape_and_range(self):
        lut = ImageProcessor.soft_proof_lut(WORKING_COLOR_SPACE, None, _srgb_path())
        self.assertIsNotNone(lut)
        n = PROOF_LUT_SIZE
        self.assertEqual(lut.shape, (n, n, n, 3))
        self.assertEqual(lut.dtype, np.float32)
        self.assertGreaterEqual(lut.min(), 0.0)
        self.assertLessEqual(lut.max(), 1.0)

    def test_more_accurate_than_the_path_it_replaces(self):
        """The old path quantized the float buffer to 8 bits before transforming.

        Scored against a 129³ stand-in for the continuous transform, the shipped grid
        must beat that round-trip on every bundled output space — otherwise this is a
        fidelity regression, not a speed-up. An absolute bound would be arbitrary:
        wide-gamut destinations clip harder at the boundary, and both paths err more
        there.
        """
        for space in ("sRGB", "Adobe RGB", "ProPhoto RGB", "P3 D65", "Rec 2020", "Greyscale"):
            path = ColorSpaceRegistry.get_icc_path(space)
            self.assertIsNotNone(path, f"{space} should be a bundled export profile")
            with self.subTest(space=space):
                truth = apply_lut_f32(self.img, _proof_lut(path, _TRUTH_LUT_SIZE))
                lut_err = np.abs(apply_lut_f32(self.img, _proof_lut(path, PROOF_LUT_SIZE)) - truth)
                old_err = np.abs(self._per_pixel(self.img, path) - truth)
                self.assertLess(
                    lut_err.max(),
                    old_err.max(),
                    f"{space}: worst-case {lut_err.max() * 255:.2f}/255 vs the old path's {old_err.max() * 255:.2f}/255",
                )
                self.assertLess(lut_err.mean(), old_err.mean(), f"{space}: mean error regressed")

    def test_different_output_spaces_give_different_luts(self):
        srgb = ImageProcessor.soft_proof_lut(WORKING_COLOR_SPACE, None, ColorSpaceRegistry.get_icc_path("sRGB"))
        prophoto = ImageProcessor.soft_proof_lut(WORKING_COLOR_SPACE, None, ColorSpaceRegistry.get_icc_path("ProPhoto RGB"))
        self.assertFalse(np.array_equal(srgb, prophoto), "the proof must depend on the output space")

    def test_built_once_per_profile_combination(self):
        out_path = _srgb_path()
        ImageProcessor.soft_proof_lut.cache_clear()
        for _ in range(5):
            ImageProcessor.soft_proof_lut(WORKING_COLOR_SPACE, None, out_path)
        self.assertEqual(ImageProcessor.soft_proof_lut.cache_info().misses, 1)
        self.assertEqual(ImageProcessor.soft_proof_lut.cache_info().hits, 4)

    def test_monitor_profile_is_part_of_the_key(self):
        out_path = _srgb_path()
        from negpy.infrastructure.display.color_mgmt import icc_bytes_for_space

        monitor = icc_bytes_for_space("P3 D65")
        self.assertIsNotNone(monitor, "P3 D65 should be a bundled profile")
        plain = ImageProcessor.soft_proof_lut(WORKING_COLOR_SPACE, None, out_path)
        on_p3 = ImageProcessor.soft_proof_lut(WORKING_COLOR_SPACE, None, out_path, monitor)
        self.assertFalse(np.array_equal(plain, on_p3), "the proof must land on the monitor profile")


class TestDisplayTransformCarriesTheProof(unittest.TestCase):
    """The proof reaches pixels through the display transform, not the render worker."""

    def setUp(self):
        from negpy.infrastructure.display.color_mgmt import get_display_lut

        get_display_lut.cache_clear()
        ImageProcessor.soft_proof_lut.cache_clear()
        rng = np.random.default_rng(1)
        self.img = np.ascontiguousarray(rng.random((32, 48, 3), dtype=np.float32))

    def test_get_display_lut_returns_the_proof_lut(self):
        from negpy.infrastructure.display.color_mgmt import get_display_lut

        out_path = _srgb_path()
        proofed = get_display_lut(WORKING_COLOR_SPACE, None, (None, out_path))
        plain = get_display_lut(WORKING_COLOR_SPACE, None, None)
        self.assertIsNotNone(proofed)
        self.assertEqual(proofed.shape, (PROOF_LUT_SIZE, PROOF_LUT_SIZE, PROOF_LUT_SIZE, 3))
        self.assertFalse(plain is not None and plain.shape == proofed.shape and np.array_equal(plain, proofed))

    def test_apply_display_transform_proofs(self):
        from negpy.infrastructure.display.color_mgmt import apply_display_transform

        out_path = _srgb_path()
        proofed = apply_display_transform(self.img, WORKING_COLOR_SPACE, None, (None, out_path))
        plain = apply_display_transform(self.img, WORKING_COLOR_SPACE, None, None)
        self.assertFalse(np.array_equal(proofed, plain), "the proof must change the displayed pixels")
        expected = apply_lut_f32(self.img, ImageProcessor.soft_proof_lut(WORKING_COLOR_SPACE, None, out_path, None))
        self.assertTrue(np.array_equal(proofed, expected))

    def test_the_render_worker_no_longer_bakes_a_proof(self):
        """Baking would strand the buffer on the host and break the zero-copy path."""
        from negpy.desktop.workers.render import RenderWorker

        self.assertFalse(hasattr(RenderWorker, "_soft_proof"))


if __name__ == "__main__":
    unittest.main()
