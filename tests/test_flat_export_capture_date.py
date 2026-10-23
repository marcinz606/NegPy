"""Flat TIFF export keeps the source capture date where an EXIF reader looks."""

import io

import numpy as np
import piexif
import pytest
import tifffile
from PIL import Image

from negpy.domain.models import ColorSpace, ExportConfig, ExportFormat, flat_export_config
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.writer import attach_exif_ifd, embed_metadata, export_embed_plan
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.infrastructure.loaders.helpers import read_exif_from_file
from negpy.services.rendering.image_processor import ImageProcessor

_CAPTURE = b"2024:05:12 14:30:00"


def test_flat_tiff_export_keeps_datetime_original(tmp_path) -> None:
    source = tmp_path / "camera.jpg"
    exif = {
        "0th": {
            piexif.ImageIFD.Make: b"Canon",
            piexif.ImageIFD.Model: b"EOS R5",
        },
        "Exif": {piexif.ExifIFD.DateTimeOriginal: _CAPTURE},
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(source, format="JPEG", quality=90, exif=piexif.dump(exif))
    source_exif = read_exif_from_file(str(source))
    settings = flat_export_config(ExportConfig(export_color_space=ColorSpace.SRGB.value))

    embed_plan = None
    if settings.export_fmt in (ExportFormat.TIFF, ExportFormat.PNG):
        embed_plan = export_embed_plan(MetadataConfig(), source_exif, str(source))
    bits, _status = ImageProcessor().encode_export(
        np.full((8, 8, 3), 0.4, np.float32),
        settings,
        ColorSpace.SRGB.value,
        WORKING_COLOR_SPACE,
        embed_plan=embed_plan,
    )
    if embed_plan is None:
        bits = embed_metadata(bits, MetadataConfig(), source_exif)

    assert piexif.load(bits)["Exif"][piexif.ExifIFD.DateTimeOriginal] == _CAPTURE


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_multi_strip_tiff_round_trip_keeps_pixels_and_capture_date(byteorder: str) -> None:
    image = np.arange(18 * 8 * 3, dtype=np.uint16).reshape(18, 8, 3)
    plain_buf = io.BytesIO()
    tifffile.imwrite(plain_buf, image, rowsperstrip=1, byteorder=byteorder)
    plain = plain_buf.getvalue()
    if len(plain) % 2 == 0:
        plain += b"\x00"
    exif = piexif.dump({"0th": {}, "Exif": {piexif.ExifIFD.DateTimeOriginal: _CAPTURE}, "GPS": {}, "1st": {}})
    patched = attach_exif_ifd(plain, exif)

    assert np.array_equal(tifffile.imread(io.BytesIO(plain)), image)
    assert np.array_equal(tifffile.imread(io.BytesIO(patched)), image)
    assert piexif.load(patched)["Exif"][piexif.ExifIFD.DateTimeOriginal] == _CAPTURE
    width = "little" if byteorder == "<" else "big"
    assert int.from_bytes(patched[4:8], width) % 2 == 0
    with tifffile.TiffFile(io.BytesIO(patched)) as handle:
        assert len(handle.pages[0].dataoffsets) == 18
        assert handle.byteorder == byteorder
