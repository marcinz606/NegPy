import unittest

import numpy as np
from PIL import ImageCms

from negpy.domain.models import ColorSpace
from negpy.infrastructure.display.color_mgmt import (
    ColorService,
    apply_display_transform,
    get_display_lut,
    icc_bytes_for_space,
    open_profile_from_bytes,
    profile_description,
)
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

    def test_greyscale_export_matches_gamma22_tag(self):
        """B&W luma must be re-encoded from the working TRC (ProPhoto ROMM 1.8) to the
        gamma 2.2 expected by the GrayGamma2.2 tag, so a viewer recovers the linear luma."""
        from negpy.kernel.image.logic import working_oetf_encode

        lin = np.linspace(0.05, 0.9, 6, dtype=np.float32)
        enc = np.asarray(working_oetf_encode(lin))  # engine-output luma in the working TRC
        u16 = np.clip(enc * 65535.0 + 0.5, 0, 65535).astype(np.uint16).reshape(1, -1)

        out, _icc = self.proc._apply_color_management_u16_greyscale(u16, WORKING_COLOR_SPACE, ColorSpace.GREYSCALE.value, None)

        decoded = (out.astype(np.float32) / 65535.0) ** 2.2  # as a Gamma2.2 viewer would decode
        np.testing.assert_allclose(decoded.reshape(-1), lin, atol=0.01)

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


def _icc_bytes(cs_name: str) -> bytes:
    with open(ColorSpaceRegistry.get_icc_path(cs_name), "rb") as f:
        return f.read()


class TestMonitorDisplayProfile(unittest.TestCase):
    """The display transform must target the monitor profile, not always sRGB."""

    def test_no_monitor_is_legacy_srgb(self):
        # Default (no monitor profile) keeps the sRGB-display behaviour: sRGB working
        # stays identity, so existing callers are unaffected.
        self.assertIsNone(get_display_lut(ColorSpace.SRGB.value))
        self.assertIsNone(get_display_lut(ColorSpace.SRGB.value, None))

    def test_monitor_profile_makes_srgb_working_nonidentity(self):
        # On a P3 display even an sRGB working space needs a transform.
        lut = get_display_lut(ColorSpace.SRGB.value, _icc_bytes(ColorSpace.P3_D65.value))
        self.assertIsNotNone(lut)

    def test_monitor_display_differs_from_srgb_display(self):
        p3 = _icc_bytes(ColorSpace.P3_D65.value)
        patch = np.array([[[0.80, 0.30, 0.20]]], dtype=np.float32)
        srgb_disp = apply_display_transform(patch, WORKING_COLOR_SPACE)[0, 0]
        p3_disp = apply_display_transform(patch, WORKING_COLOR_SPACE, p3)[0, 0]
        self.assertFalse(
            np.allclose(srgb_disp, p3_disp, atol=0.02),
            msg=f"P3 display {p3_disp} should differ from sRGB display {srgb_disp}",
        )


class TestDisplayProfileOverrideHelpers(unittest.TestCase):
    """Helpers backing the manual Display-profile override dropdown."""

    def test_icc_bytes_for_space_opens_as_profile(self):
        data = icc_bytes_for_space(ColorSpace.P3_D65.value)
        self.assertTrue(data)
        # Must open as a usable profile (drives apply_display_transform).
        self.assertIsNotNone(open_profile_from_bytes(data))

    def test_icc_bytes_for_space_unknown_is_none(self):
        self.assertIsNone(icc_bytes_for_space("NotARealSpace"))

    def test_profile_description_none_is_srgb_fallback(self):
        self.assertEqual(profile_description(None), "sRGB fallback")

    def test_profile_description_reads_name(self):
        desc = profile_description(icc_bytes_for_space(ColorSpace.P3_D65.value))
        self.assertTrue(desc and desc != "sRGB fallback")

    def test_override_bytes_match_apply_transform(self):
        # An override to Display P3 must produce the same display LUT as feeding those
        # bytes directly — i.e. the dropdown selection drives the real transform.
        p3 = icc_bytes_for_space(ColorSpace.P3_D65.value)
        patch = np.array([[[0.80, 0.30, 0.20]]], dtype=np.float32)
        via_override = apply_display_transform(patch, WORKING_COLOR_SPACE, p3)
        via_direct = apply_display_transform(patch, WORKING_COLOR_SPACE, _icc_bytes(ColorSpace.P3_D65.value))
        np.testing.assert_array_equal(via_override, via_direct)


class TestPrintProfileDetection(unittest.TestCase):
    """`_is_print_profile` routes paper/printer profiles to the paper-white proof."""

    class _Stub:
        def __init__(self, device_class="", xcolor_space="RGB "):
            self.profile = type("P", (), {"device_class": device_class, "xcolor_space": xcolor_space})()

    def test_printer_class_is_print(self):
        self.assertTrue(ImageProcessor._is_print_profile(self._Stub(device_class="prtr ")))

    def test_cmyk_space_is_print(self):
        self.assertTrue(ImageProcessor._is_print_profile(self._Stub(xcolor_space="CMYK")))

    def test_bundled_display_spaces_not_print(self):
        for cs in (ColorSpace.SRGB.value, ColorSpace.ADOBE_RGB.value, ColorSpace.P3_D65.value):
            prof = ImageCms.getOpenProfile(ColorSpaceRegistry.get_icc_path(cs))
            self.assertFalse(ImageProcessor._is_print_profile(prof), msg=f"{cs} should not be a print profile")


class TestSoftProofToMonitor(unittest.TestCase):
    """Soft proof must chain working→output→monitor when a display profile is set."""

    @classmethod
    def setUpClass(cls):
        cls.proc = ImageProcessor()

    def test_proof_unchanged_without_monitor(self):
        from PIL import Image

        u8 = (np.array([[[0.70, 0.40, 0.25]]], dtype=np.float32) * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(u8, mode="RGB")
        out_path = ColorSpaceRegistry.get_icc_path(ColorSpace.SRGB.value)
        a = np.asarray(self.proc.soft_proof_preview(img, WORKING_COLOR_SPACE, None, out_path), dtype=np.float32)
        b = np.asarray(self.proc.soft_proof_preview(img, WORKING_COLOR_SPACE, None, out_path, None), dtype=np.float32)
        np.testing.assert_array_equal(a, b)

    def test_proof_in_gamut_stable_across_output_spaces(self):
        """An in-gamut proof must look the same regardless of export space (#243).

        The output→display step always runs (sRGB display here), so source→output→
        display cancels to source→display for in-gamut colors — the intermediate
        export space drops out. Only out-of-gamut colors should differ.
        """
        from PIL import Image

        u8 = (np.array([[[0.55, 0.42, 0.30]]], dtype=np.float32) * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(u8, mode="RGB")
        spaces = (
            ColorSpace.SRGB.value,
            ColorSpace.ADOBE_RGB.value,
            ColorSpace.REC2020.value,
            ColorSpace.PROPHOTO.value,
        )
        outs = [
            np.asarray(
                self.proc.soft_proof_preview(img, WORKING_COLOR_SPACE, None, ColorSpaceRegistry.get_icc_path(s)),
                dtype=np.float32,
            )
            for s in spaces
        ]
        ref = outs[0]
        for s, arr in zip(spaces, outs):
            # ±2/255: only 8-bit LUT rounding, not a per-space appearance shift.
            self.assertTrue(np.allclose(arr, ref, atol=2.0), msg=f"{s} proof diverges: {arr.ravel()} vs {ref.ravel()}")

    def test_proof_to_monitor_matches_explicit_chain(self):
        from PIL import Image

        p3 = _icc_bytes(ColorSpace.P3_D65.value)
        out_path = ColorSpaceRegistry.get_icc_path(ColorSpace.SRGB.value)
        u8 = (np.array([[[0.70, 0.40, 0.25]]], dtype=np.float32) * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(u8, mode="RGB")

        no_mon = np.asarray(self.proc.soft_proof_preview(img, WORKING_COLOR_SPACE, None, out_path), dtype=np.float32)
        with_mon = np.asarray(self.proc.soft_proof_preview(img, WORKING_COLOR_SPACE, None, out_path, p3), dtype=np.float32)
        # The output→monitor step changes the displayed pixels.
        self.assertFalse(np.allclose(no_mon, with_mon, atol=1.0))

        # ...and equals an explicit working→output→monitor chain (relative + BPC).
        p_work = _open(WORKING_COLOR_SPACE)
        p_out = _open(ColorSpace.SRGB.value)
        p_mon = open_profile_from_bytes(p3)
        step1 = ImageCms.profileToProfile(
            img,
            p_work,
            p_out,
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
        step2 = ImageCms.profileToProfile(
            step1,
            p_out,
            p_mon,
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
        chain = np.asarray(step2, dtype=np.float32)
        np.testing.assert_allclose(with_mon, chain, atol=1.0)


if __name__ == "__main__":
    unittest.main()
