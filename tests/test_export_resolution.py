"""Guards the resolution tags on exported files.

tifffile writes XResolution (1, 1) with ResolutionUnit NONE when ``resolution`` is
unset, and readers report that as 1 DPI rather than falling back to 72. PIL reads a
PNG's pHYs into ``info`` but writes it only from a save kwarg, so a metadata rewrite
drops it. Both need an explicit value at every write.
"""

import io

import numpy as np
import piexif
import pytest
import tifffile
from PIL import Image

from negpy.domain.models import ColorSpace, ExportConfig, ExportFormat, ExportResolutionMode, preset_from_export_config
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.writer import embed_metadata, export_embed_plan, preserve_source_metadata
from negpy.infrastructure.loaders.jxl_boxes import read_jxl_exif
from negpy.services.export.linear_output import NOMINAL_DPI, _write_tiff
from negpy.services.export.print import PrintService
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


# --- Source EXIF resolution must not survive into the export -------------------
#
# A scan's XResolution describes the scanner, not the resized file NegPy writes.
# JPEG and JPEG XL keep their only resolution record in EXIF, and PNG's eXIf chunk
# sits beside the pHYs the encoder writes, so all three have to be rewritten.

_SCANNER_DPI = 3600


def _source_exif() -> dict:
    return {
        "0th": {
            piexif.ImageIFD.Make: b"Nikon",
            piexif.ImageIFD.XResolution: (_SCANNER_DPI, 1),
            piexif.ImageIFD.YResolution: (_SCANNER_DPI, 1),
            piexif.ImageIFD.ResolutionUnit: 2,
        },
        "Exif": {},
        "GPS": {},
        "Interop": {},
        "1st": {},
    }


def _pixels_settings(fmt: ExportFormat) -> ExportConfig:
    return ExportConfig(
        export_fmt=fmt,
        export_dpi=300,
        export_resolution_mode=ExportResolutionMode.TARGET_PX.value,
        export_print_size=_PRINT_SIZE_CM,
        export_target_long_edge_px=_TARGET_PX,
    )


def _exif_dpi(bits: bytes, fmt: ExportFormat):
    """XResolution numerator from whichever EXIF carrier the format uses."""
    if fmt == ExportFormat.JXL:
        raw = read_jxl_exif(bits)
        return None if raw is None else piexif.load(raw)["0th"].get(piexif.ImageIFD.XResolution, (None,))[0]
    if fmt == ExportFormat.PNG:
        with Image.open(io.BytesIO(bits)) as img:
            raw = img.info.get("exif")
        return None if raw is None else piexif.load(raw)["0th"].get(piexif.ImageIFD.XResolution, (None,))[0]
    return piexif.load(bits)["0th"].get(piexif.ImageIFD.XResolution, (None,))[0]


@pytest.mark.parametrize("fmt", [ExportFormat.JPEG, ExportFormat.JXL, ExportFormat.PNG])
def test_embed_replaces_stale_source_resolution(proc, fmt):
    settings = _pixels_settings(fmt)
    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    out = embed_metadata(bits, MetadataConfig(), _source_exif(), dpi=PrintService.resolution_tag_dpi(settings))

    assert _exif_dpi(out, fmt) == _DERIVED_DPI


@pytest.mark.parametrize("fmt", [ExportFormat.JPEG, ExportFormat.JXL, ExportFormat.PNG])
def test_preserve_keeps_the_source_resolution(proc, fmt):
    """Protect original metadata means untouched, resolution included. An export that
    was not resampled still has the source's sampling density, so overwriting it with
    the Print-only spinbox value would replace a right answer with a guess."""
    settings = _pixels_settings(fmt)
    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    out = preserve_source_metadata(bits, "unused.nef", _source_exif())

    assert _exif_dpi(out, fmt) == _SCANNER_DPI


def _protect_plan(settings):
    return export_embed_plan(
        MetadataConfig(protect_original_metadata=True),
        _source_exif(),
        "unused.nef",
        dpi=PrintService.resolution_tag_dpi(settings),
    )


def test_embed_plan_under_protect_keeps_the_source_resolution(proc):
    """The first-encode path honours the toggle too. PNG's eXIf therefore keeps the
    scanner's DPI while its pHYs describes the export: two records, by design, because
    only one of them is copied metadata."""
    settings = _pixels_settings(ExportFormat.PNG)

    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value, embed_plan=_protect_plan(settings))

    assert _exif_dpi(bits, ExportFormat.PNG) == _SCANNER_DPI
    assert _png_dpi(bits) == pytest.approx((_DERIVED_DPI, _DERIVED_DPI), abs=0.01)


def test_tiff_baseline_describes_the_export_even_under_protect(proc):
    """TIFF has no separate EXIF resolution: 282/283/296 are filtered out of the
    extratags, so the baseline tag is the file's own record and always describes it."""
    settings = _pixels_settings(ExportFormat.TIFF)

    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value, embed_plan=_protect_plan(settings))

    assert _tiff_resolution(bits)[0] == (_DERIVED_DPI, 1)


@pytest.mark.parametrize("fmt", [ExportFormat.TIFF, ExportFormat.PNG])
def test_embed_plan_path_carries_the_export_resolution(proc, fmt):
    """TIFF and PNG take metadata at the first encode instead of a post-hoc rewrite."""
    settings = _pixels_settings(fmt)
    plan = export_embed_plan(MetadataConfig(), _source_exif(), "unused.nef", dpi=PrintService.resolution_tag_dpi(settings))

    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value, embed_plan=plan)

    if fmt == ExportFormat.TIFF:
        assert _tiff_resolution(bits)[0] == (_DERIVED_DPI, 1)
    else:
        assert _png_dpi(bits) == pytest.approx((_DERIVED_DPI, _DERIVED_DPI), abs=0.01)
    assert _exif_dpi(bits, fmt) == _DERIVED_DPI


def test_png_exif_agrees_with_phys(proc):
    """Two resolution records in one file must not disagree."""
    settings = _pixels_settings(ExportFormat.PNG)
    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    out = embed_metadata(bits, MetadataConfig(), _source_exif(), dpi=PrintService.resolution_tag_dpi(settings))

    assert _png_dpi(out) == pytest.approx((_DERIVED_DPI, _DERIVED_DPI), abs=0.01)
    assert _exif_dpi(out, ExportFormat.PNG) == _DERIVED_DPI


def test_webp_keeps_the_source_resolution(proc):
    """Deliberate: WebP has no resolution field of its own and browsers ignore the
    EXIF one, so it is left as the source wrote it rather than rewritten."""
    settings = _pixels_settings(ExportFormat.WEBP)
    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    out = embed_metadata(bits, MetadataConfig(), _source_exif(), dpi=PrintService.resolution_tag_dpi(settings))

    assert _exif_dpi(out, ExportFormat.WEBP) == _SCANNER_DPI


def test_embed_without_a_dpi_leaves_exif_alone(proc):
    """The override is opt-in, so callers with no export settings are unaffected."""
    settings = _pixels_settings(ExportFormat.JPEG)
    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    out = embed_metadata(bits, MetadataConfig(), _source_exif())

    assert _exif_dpi(out, ExportFormat.JPEG) == _SCANNER_DPI
