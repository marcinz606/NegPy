"""Exported files must carry an ICC profile that matches their pixel encoding."""

import io

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from negpy.domain.models import EXPORT_COLOR_SPACES, ColorSpace, ExportConfig, ExportFormat
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE, ColorSpaceRegistry
from negpy.kernel.image.logic import working_oetf_decode
from negpy.services.rendering.image_processor import ImageProcessor


@pytest.fixture(scope="module")
def proc():
    return ImageProcessor()


def _profile_desc(icc_bytes):
    return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))).strip()


def _pil_icc(bits):
    im = Image.open(io.BytesIO(bits))
    im.load()
    return im.info.get("icc_profile")


def _srgb_oetf(lin: float) -> float:
    return lin * 12.92 if lin <= 0.0031308 else 1.055 * lin ** (1 / 2.4) - 0.055


def _encode(proc, fmt, cs, buf, **cfg):
    settings = ExportConfig(export_fmt=fmt, export_color_space=cs, **cfg)
    return proc._encode_export(buf, settings, cs, working_color_space=WORKING_COLOR_SPACE)


def test_greyscale_jpeg_is_tagged_and_matches_tiff(proc):
    """Regression: greyscale JPEG fed an L image into an RGB CMS transform, which
    failed silently and shipped untagged working-TRC pixels — inconsistent with the
    greyscale TIFF of the same edit."""
    buf = np.full((8, 8, 3), 0.42, dtype=np.float32)

    jpg_bits, _ = _encode(proc, ExportFormat.JPEG, ColorSpace.GREYSCALE.value, buf.copy())
    icc = _pil_icc(jpg_bits)
    assert icc, "greyscale JPEG must embed the grey profile"
    assert "Gray" in _profile_desc(icc)

    tif_bits, _ = _encode(proc, ExportFormat.TIFF, ColorSpace.GREYSCALE.value, buf.copy())
    tif_px = tifffile.imread(io.BytesIO(tif_bits))
    jpg_px = np.asarray(Image.open(io.BytesIO(jpg_bits)))
    assert jpg_px.ndim == 2  # single-channel L, not RGB
    # Same edit, same TRC across formats (JPEG loss allows a small tolerance).
    assert abs(int(jpg_px[4, 4]) - int(tif_px[4, 4]) / 257.0) < 2.5


def test_greyscale_webp_is_tagged(proc):
    buf = np.full((8, 8, 3), 0.42, dtype=np.float32)
    bits, _ = _encode(proc, ExportFormat.WEBP, ColorSpace.GREYSCALE.value, buf.copy(), webp_lossless=True)
    icc = _pil_icc(bits)
    assert icc, "greyscale WebP must embed the grey profile"
    assert "Gray" in _profile_desc(icc)


def test_greyscale_encode_uses_srgb_trc_not_pure_2_2(proc):
    """The bundled GrayGamma2.2.icc actually holds the sRGB TRC (see _JXL_COLOR in
    image_processor); a pure-2.2 encode mistags the shadows. Probe a deep shadow
    where the two curves differ by ~3x."""
    v = 0.05  # working-encoded; linear ~0.0014, inside sRGB's linear toe segment
    buf = np.full((4, 4, 3), v, dtype=np.float32)
    bits, _ = _encode(proc, ExportFormat.TIFF, ColorSpace.GREYSCALE.value, buf)
    px = tifffile.imread(io.BytesIO(bits))

    lin = float(working_oetf_decode(np.float32(v)))
    expected_srgb = round(_srgb_oetf(lin) * 65535)
    legacy_2_2 = round((lin ** (1 / 2.2)) * 65535)
    assert abs(int(px[2, 2]) - expected_srgb) <= 2
    assert abs(int(px[2, 2]) - legacy_2_2) > 100  # proves we're not on the old curve


def test_unmapped_space_falls_back_to_tagged_working_space(proc):
    """Regression: ACES/XYZ have no ICC profile; export used to skip CMS and ship
    untagged working-space pixels. It must instead export the working space with
    its own profile embedded."""
    buf = np.random.default_rng(5).random((8, 8, 3)).astype(np.float32)

    aces_bits, _ = _encode(proc, ExportFormat.TIFF, ColorSpace.ACES.value, buf.copy())
    working_bits, _ = _encode(proc, ExportFormat.TIFF, WORKING_COLOR_SPACE, buf.copy())

    with tifffile.TiffFile(io.BytesIO(aces_bits)) as tf:
        icc = tf.pages[0].iccprofile
        px = tf.pages[0].asarray()
    assert icc, "fallback export must still be tagged"
    with open(ColorSpaceRegistry.get_icc_path(WORKING_COLOR_SPACE), "rb") as f:
        assert icc == f.read()
    with tifffile.TiffFile(io.BytesIO(working_bits)) as tf:
        np.testing.assert_array_equal(px, tf.pages[0].asarray())


def test_export_color_spaces_exclude_unmappable():
    """The export UI must not offer spaces the encoder can only fall back on."""
    assert ColorSpace.ACES.value not in EXPORT_COLOR_SPACES
    assert ColorSpace.XYZ.value not in EXPORT_COLOR_SPACES
    for cs in EXPORT_COLOR_SPACES:
        if cs == ColorSpace.SAME_AS_SOURCE.value:
            continue  # resolved to a concrete space at export time
        assert ColorSpaceRegistry.get_icc_path(cs) is not None, f"{cs} offered but unmappable"


def test_contact_sheet_srgb_bytes_available():
    from negpy.desktop.workers.export import _srgb_icc_bytes

    icc = _srgb_icc_bytes()
    assert icc
    assert "sRGB" in _profile_desc(icc)
