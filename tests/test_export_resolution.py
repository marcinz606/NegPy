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
from negpy.desktop.workers.export import _export_resolution
from negpy.features.metadata.writer import (
    embed_metadata,
    export_embed_plan,
    preserve_source_metadata,
)
from negpy.features.metadata.resolution import Resolution, read_source
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

    out = embed_metadata(bits, MetadataConfig(), _source_exif(), resolution=Resolution.from_dpi(PrintService.resolution_tag_dpi(settings)))

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
        resolution=Resolution.from_dpi(PrintService.resolution_tag_dpi(settings)),
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
    plan = export_embed_plan(
        MetadataConfig(), _source_exif(), "unused.nef", resolution=Resolution.from_dpi(PrintService.resolution_tag_dpi(settings))
    )

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

    out = embed_metadata(bits, MetadataConfig(), _source_exif(), resolution=Resolution.from_dpi(PrintService.resolution_tag_dpi(settings)))

    assert _png_dpi(out) == pytest.approx((_DERIVED_DPI, _DERIVED_DPI), abs=0.01)
    assert _exif_dpi(out, ExportFormat.PNG) == _DERIVED_DPI


def test_webp_carries_the_export_resolution(proc):
    """WebP has no resolution field of its own, so EXIF is the only carrier."""
    settings = _pixels_settings(ExportFormat.WEBP)
    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value)

    out = embed_metadata(bits, MetadataConfig(), _source_exif(), resolution=Resolution.from_dpi(PrintService.resolution_tag_dpi(settings)))

    assert _exif_dpi(out, ExportFormat.WEBP) == _DERIVED_DPI


# --- Which DPI wins ------------------------------------------------------------


def _exif_at(dpi, unit=2):
    zeroth = (
        {}
        if dpi is None
        else {
            piexif.ImageIFD.XResolution: (dpi, 1),
            piexif.ImageIFD.YResolution: (dpi, 1),
            piexif.ImageIFD.ResolutionUnit: unit,
        }
    )
    return {"0th": zeroth, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}


@pytest.mark.parametrize(
    ("dpi", "unit", "expected"),
    [
        (3600, 2, Resolution((3600, 1), (3600, 1), 2)),
        (1417, 3, Resolution((1417, 1), (1417, 1), 3)),  # centimetres, kept as centimetres
        (3600, 1, None),  # no absolute unit: an aspect ratio, not a resolution
        (None, 2, None),
    ],
)
def test_read_source_from_exif(dpi, unit, expected):
    assert read_source(None, _exif_at(dpi, unit)) == expected


def test_read_source_keeps_asymmetric_axes():
    exif = {
        "0th": {
            piexif.ImageIFD.XResolution: (600, 1),
            piexif.ImageIFD.YResolution: (300, 1),
            piexif.ImageIFD.ResolutionUnit: 2,
        }
    }
    assert read_source(None, exif) == Resolution((600, 1), (300, 1), 2)


def test_read_source_falls_back_to_the_container(tmp_path):
    """A lab JPEG commonly carries its density in JFIF and no EXIF at all."""
    path = tmp_path / "lab.jpg"
    Image.new("RGB", (8, 8)).save(path, "JPEG", dpi=(72, 72))

    assert piexif.load(path.read_bytes())["0th"] == {}  # nothing in EXIF
    assert read_source(str(path), None) == Resolution((72, 1), (72, 1), 2)


def test_read_source_reads_the_file_when_the_cache_is_empty(tmp_path):
    """source_exif is only cached for files the user selected, so a batch export of
    untouched frames has to reach the file itself."""
    path = tmp_path / "scan.tif"
    tifffile.imwrite(path, np.zeros((8, 8, 3), np.uint16), photometric="rgb", resolution=(600, 600), resolutionunit="INCH")

    assert read_source(str(path), None) == Resolution((600, 1), (600, 1), 2)


class _Task:
    """The fields _export_resolution reads."""

    def __init__(self, settings, source_exif, protect=False, path="unused.nef"):
        self.file_info = {"path": path}
        self.export_settings = settings
        self.source_exif = source_exif
        self.metadata_config = MetadataConfig(protect_original_metadata=protect)


def _original_settings(dpi=300):
    return ExportConfig(export_fmt=ExportFormat.TIFF, export_dpi=dpi, export_resolution_mode=ExportResolutionMode.ORIGINAL.value)


def _print_settings_at(dpi):
    return ExportConfig(
        export_fmt=ExportFormat.TIFF,
        export_dpi=dpi,
        export_resolution_mode=ExportResolutionMode.PRINT.value,
        export_print_size=_PRINT_SIZE_CM,
    )


def test_original_mode_keeps_the_source_resolution():
    """Nothing is resampled, so the source's sampling density still describes the pixels."""
    assert _export_resolution(_Task(_original_settings(), _exif_at(_SCANNER_DPI))) == Resolution((_SCANNER_DPI, 1), (_SCANNER_DPI, 1), 2)


def test_original_mode_falls_back_when_the_source_declares_none():
    assert _export_resolution(_Task(_original_settings(dpi=300), _exif_at(None))) == Resolution.from_dpi(300)


@pytest.mark.parametrize(
    ("settings", "expected"),
    [(_print_settings_at(600), 600), (_pixels_settings(ExportFormat.TIFF), _DERIVED_DPI)],
)
def test_resampling_modes_use_the_size_the_user_asked_for(settings, expected):
    assert _export_resolution(_Task(settings, _exif_at(_SCANNER_DPI))) == Resolution.from_dpi(expected)


@pytest.mark.parametrize("settings", [_original_settings(), _print_settings_at(600), _pixels_settings(ExportFormat.TIFF)])
def test_protect_keeps_the_source_resolution_in_every_mode(settings):
    """Protect wins over an explicit Print DPI: the user asked for the source's
    metadata untouched, and that is the more deliberate of the two choices."""
    assert _export_resolution(_Task(settings, _exif_at(_SCANNER_DPI), protect=True)) == Resolution((_SCANNER_DPI, 1), (_SCANNER_DPI, 1), 2)


@pytest.mark.parametrize(
    ("exif", "expected"),
    [
        (
            {"0th": {piexif.ImageIFD.XResolution: (600, 1), piexif.ImageIFD.YResolution: (300, 1), piexif.ImageIFD.ResolutionUnit: 2}},
            ((600, 1), (300, 1), 2),
        ),
        (
            {"0th": {piexif.ImageIFD.XResolution: (118, 1), piexif.ImageIFD.YResolution: (118, 1), piexif.ImageIFD.ResolutionUnit: 3}},
            ((118, 1), (118, 1), 3),
        ),
        (
            {"0th": {piexif.ImageIFD.XResolution: (601, 2), piexif.ImageIFD.YResolution: (601, 2), piexif.ImageIFD.ResolutionUnit: 2}},
            ((601, 2), (601, 2), 2),
        ),
    ],
)
def test_protect_writes_the_source_tags_verbatim(proc, exif, expected):
    """No rounding to one integer, no forcing inches, no collapsing the two axes."""
    settings = _pixels_settings(ExportFormat.TIFF)
    res = _export_resolution(_Task(settings, exif, protect=True))

    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value, resolution=res)

    assert _tiff_resolution(bits) == expected


def test_protect_without_a_source_resolution_writes_no_claim(proc):
    """Rule: never synthesize under protect. tifffile's unit-less default is what a
    file with no resolution looks like, and that is the honest state here."""
    settings = _pixels_settings(ExportFormat.TIFF)
    res = _export_resolution(_Task(settings, _exif_at(None), protect=True))

    assert res is None
    bits, _ = proc._encode_export(_buf(), settings, ColorSpace.SRGB.value, ColorSpace.SRGB.value, resolution=res)
    assert _tiff_resolution(bits) == ((1, 1), (1, 1), 1)


def test_linear_output_inherits_the_source_resolution(tmp_path):
    """A linear dump resamples nothing, so it keeps what the source declared."""
    src = tmp_path / "scan.tif"
    tifffile.imwrite(src, np.zeros((8, 8, 3), np.uint16), photometric="rgb", resolution=(600, 600), resolutionunit="INCH")
    dest = io.BytesIO()

    _write_tiff(_buf(), dest, "scan.tif", source_path=str(src))

    assert _tiff_resolution(dest.getvalue())[0] == (600, 1)
