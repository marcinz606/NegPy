"""Guards the resolution tags on exported files.

tifffile writes XResolution (1, 1) with ResolutionUnit NONE when ``resolution`` is
unset, and readers report that as 1 DPI rather than falling back to 72. PIL reads a
PNG's pHYs into ``info`` but writes it only from a save kwarg, so a metadata rewrite
drops it. Both need an explicit value at every write.
"""

import io

import numpy as np
import pytest
import tifffile
from PIL import Image

from negpy.domain.models import ColorSpace, ExportConfig, ExportFormat, ExportResolutionMode, preset_from_export_config
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.writer import embed_metadata
from negpy.services.export.linear_output import NOMINAL_DPI, _write_tiff
from negpy.services.rendering.image_processor import ImageProcessor

# 25.4 cm of paper is 10 inches, so a 2000 px long edge is exactly 200 DPI.
_PRINT_SIZE_CM = 25.4
_TARGET_PX = 2000
_DERIVED_DPI = 200


@pytest.fixture(scope="module")
def proc():
    return ImageProcessor()


def _buf() -> np.ndarray:
    return np.random.default_rng(0).random((16, 24, 3), dtype=np.float32)


def _tiff_resolution(bits: bytes) -> tuple:
    with tifffile.TiffFile(io.BytesIO(bits)) as tf:
        tags = tf.pages[0].tags
        return tags["XResolution"].value, tags["YResolution"].value, int(tags["ResolutionUnit"].value)


def _png_dpi(bits: bytes) -> tuple:
    with Image.open(io.BytesIO(bits)) as img:
        return img.info.get("dpi")


def _print_settings(fmt: ExportFormat, dpi: int = 300) -> ExportConfig:
    return ExportConfig(export_fmt=fmt, export_dpi=dpi, export_resolution_mode=ExportResolutionMode.PRINT.value)


def test_tiff_tags_inch_resolution(proc):
    """Not the unit-less (1, 1) tifffile writes by default."""
    bits, ext = proc._encode_export(_buf(), _print_settings(ExportFormat.TIFF), ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    assert ext == "tiff"
    assert _tiff_resolution(bits) == ((300, 1), (300, 1), 2)  # RESUNIT.INCH


def test_tiff_greyscale_tags_inch_resolution(proc):
    """The greyscale branch shares the write, so it carries the same tags."""
    settings = _print_settings(ExportFormat.TIFF, dpi=600)

    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.GREYSCALE.value, ColorSpace.SRGB.value)

    assert _tiff_resolution(bits) == ((600, 1), (600, 1), 2)


def test_png_carries_phys(proc):
    """pHYs stores pixels per metre, so the round trip lands just short of the integer."""
    bits, ext = proc._encode_export(_buf(), _print_settings(ExportFormat.PNG), ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    assert ext == "png"
    assert _png_dpi(bits) == pytest.approx((300, 300), abs=0.01)


@pytest.mark.parametrize(
    ("fmt", "read_dpi"),
    [
        (ExportFormat.TIFF, lambda bits: _tiff_resolution(bits)[0][0]),
        (ExportFormat.JPEG, lambda bits: _png_dpi(bits)[0]),
    ],
)
def test_target_px_mode_tags_the_derived_dpi(proc, fmt, read_dpi):
    """The DPI field is Print-only in the UI, so a Pixels export must tag the DPI its
    own long edge implies, not the value left behind in the field."""
    settings = ExportConfig(
        export_fmt=fmt,
        export_dpi=300,
        export_resolution_mode=ExportResolutionMode.TARGET_PX.value,
        export_print_size=_PRINT_SIZE_CM,
        export_target_long_edge_px=_TARGET_PX,
    )

    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    assert read_dpi(bits) == _DERIVED_DPI


def test_metadata_embed_keeps_tiff_resolution(proc):
    """The embed decodes and re-encodes the file; the tags must survive it."""
    bits, _ = proc._encode_export(_buf(), _print_settings(ExportFormat.TIFF), ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    out = embed_metadata(bits, MetadataConfig(), None)

    assert _tiff_resolution(out) == ((300, 1), (300, 1), 2)


def test_metadata_embed_keeps_png_dpi(proc):
    bits, _ = proc._encode_export(_buf(), _print_settings(ExportFormat.PNG), ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    out = embed_metadata(bits, MetadataConfig(), None)

    assert _png_dpi(out) == pytest.approx((300, 300), abs=0.01)


def test_linear_output_tiff_tags_nominal_resolution():
    """A linear master has no print intent but still needs an absolute unit."""
    dest = io.BytesIO()

    _write_tiff(_buf(), dest, "scan.nef")

    assert _tiff_resolution(dest.getvalue()) == ((NOMINAL_DPI, 1), (NOMINAL_DPI, 1), 2)


def test_preset_export_tags_resolution(proc):
    """The export worker encodes from an ExportPreset, which mirrors the sizing fields."""
    preset = preset_from_export_config(ExportConfig(export_fmt=ExportFormat.TIFF, export_dpi=360))

    bits, _ = proc._encode_export(_buf(), preset, ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    assert _tiff_resolution(bits) == ((360, 1), (360, 1), 2)


def test_zero_dpi_config_still_tags_a_positive_resolution(proc):
    """The tag asserts inches, so a 0 would be a division by zero downstream. The
    spinbox cannot produce one; a hand-edited sidecar can."""
    settings = ExportConfig(export_fmt=ExportFormat.TIFF, export_dpi=0, export_resolution_mode=ExportResolutionMode.PRINT.value)

    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    assert _tiff_resolution(bits) == ((1, 1), (1, 1), 2)
