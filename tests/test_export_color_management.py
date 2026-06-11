import unittest

import numpy as np
from PIL import ImageCms

from negpy.domain.models import ColorSpace
from negpy.infrastructure.display.color_mgmt import ColorService, apply_display_transform, get_display_lut
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE, ColorSpaceRegistry
from negpy.infrastructure.display.icc_lut import apply_icc_u16_rgb
from negpy.services.rendering.image_processor import ImageProcessor


def _open(cs_name: str):
    path = ColorSpaceRegistry.get_icc_path(cs_name)
    return ImageCms.getOpenProfile(path)


def _decode_to_srgb_u16(img_u16: np.ndarray, src_cs: str) -> np.ndarray:
    """Render a buffer (tagged `src_cs`) into sRGB, as a color-managed viewer would."""
    return apply_icc_u16_rgb(
        img_u16,
        _open(src_cs),
        ImageCms.createProfile("sRGB"),
        ImageCms.Intent.RELATIVE_COLORIMETRIC,
        ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )


class TestExportColorManagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proc = ImageProcessor()

    def test_export_is_appearance_preserving_across_spaces(self):
        """Exporting to sRGB / Adobe / ProPhoto must look the same in a CM viewer.

        Working space is Adobe RGB. A real working→target conversion preserves the
        in-gamut appearance, so decoding each export back through its embedded
        profile yields ~identical sRGB pixels. (The old tag-only behaviour diverged.)
        """
        # Mid patch well inside every gamut so no clipping masks differences.
        patch = np.array([[[0.50, 0.40, 0.30]]], dtype=np.float32)
        img_u16 = (patch * 65535.0 + 0.5).astype(np.uint16)

        decoded = {}
        for target in (ColorSpace.SRGB.value, ColorSpace.ADOBE_RGB.value, ColorSpace.PROPHOTO.value):
            out, _ = self.proc._apply_color_management_u16_rgb(img_u16, WORKING_COLOR_SPACE, target, None, None)
            decoded[target] = _decode_to_srgb_u16(out, target).astype(np.float32) / 65535.0

        ref = decoded[ColorSpace.SRGB.value]
        for target, arr in decoded.items():
            self.assertTrue(
                np.allclose(arr, ref, atol=0.02),
                msg=f"{target} export diverges from sRGB export in CM view: {arr.ravel()} vs {ref.ravel()}",
            )

    def test_same_space_export_is_noop(self):
        """working == target with no custom profile leaves pixels untouched."""
        img_u16 = np.random.randint(0, 65535, size=(4, 4, 3), dtype=np.uint16)
        out, icc = self.proc._apply_color_management_u16_rgb(img_u16, WORKING_COLOR_SPACE, WORKING_COLOR_SPACE, None, None)
        np.testing.assert_array_equal(out, img_u16)
        self.assertIsNotNone(icc)  # target profile still embedded


class TestDisplayTransform(unittest.TestCase):
    def test_srgb_working_is_identity(self):
        img = np.random.rand(4, 4, 3).astype(np.float32)
        out = apply_display_transform(img, ColorSpace.SRGB.value)
        np.testing.assert_array_equal(out, img)
        self.assertIsNone(get_display_lut(ColorSpace.SRGB.value))

    def test_display_matches_simulate_on_srgb(self):
        """The float display LUT must agree with PIL's simulate_on_srgb (Adobe→sRGB)."""
        from PIL import Image

        patch = np.array([[[0.80, 0.30, 0.20]]], dtype=np.float32)
        lut_out = apply_display_transform(patch, WORKING_COLOR_SPACE)[0, 0]

        u8 = (patch * 255.0 + 0.5).astype(np.uint8)
        sim = ColorService.simulate_on_srgb(Image.fromarray(u8, mode="RGB"), WORKING_COLOR_SPACE)
        sim_arr = np.asarray(sim, dtype=np.float32)[0, 0] / 255.0

        self.assertTrue(
            np.allclose(lut_out, sim_arr, atol=0.02),
            msg=f"display LUT {lut_out} != simulate_on_srgb {sim_arr}",
        )

    def test_non_rgb_buffer_passthrough(self):
        grey = np.random.rand(4, 4).astype(np.float32)
        out = apply_display_transform(grey, WORKING_COLOR_SPACE)
        np.testing.assert_array_equal(out, grey)


if __name__ == "__main__":
    unittest.main()
