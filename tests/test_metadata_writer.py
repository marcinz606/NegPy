"""
Guards the JPEG EXIF embed against the 64 KB APP1 overflow: source RAWs often carry an
embedded thumbnail + multi-KB MakerNote that piexif.insert can't pack into one segment.
"""

import io
import shutil
import subprocess

import piexif
import pytest
from PIL import Image

import imagecodecs
import numpy as np

from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.writer import embed_metadata, preserve_source_metadata
from negpy.infrastructure.loaders.jxl_boxes import read_jxl_exif

_RAW_PREVIEW_0TH_TAGS = (330, 273, 279, 256, 257, 513, 514)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 0, 0)).save(buf, "JPEG")
    return buf.getvalue()


def _jxl() -> bytes:
    img = np.zeros((4, 4, 3), dtype=np.uint16)
    return bytes(
        imagecodecs.jpegxl_encode(img, bitspersample=16, photometric="RGB", primaries="SRGB", transfer="SRGB", lossless=True, effort=1)
    )


def _raw_like_source_exif() -> dict:
    """Synthetic EXIF mimicking piexif.load() output from a Nikon RAW preview IFD."""
    return {
        "0th": {
            piexif.ImageIFD.Make: b"NIKON CORPORATION",
            piexif.ImageIFD.Model: b"NIKON D750",
            330: (12894, 13012, 13238),
            273: 210440,
            279: 57600,
            256: 160,
            257: 120,
            513: 999,
            514: 12345,
        },
        "Exif": {
            piexif.ExifIFD.ExposureTime: (1, 640),
            piexif.ExifIFD.FNumber: (56, 10),
            piexif.ExifIFD.ISOSpeedRatings: 100,
            piexif.ExifIFD.FocalLengthIn35mmFilm: 60,
            piexif.ExifIFD.DateTimeOriginal: b"2026:07:03 18:51:59",
        },
        "GPS": {},
        "Interop": {},
        "1st": {},
    }


def test_embed_strips_raw_preview_ifd_tags_from_jpeg() -> None:
    """RAW EXIF carries embedded preview IFD0 tags that break ExifTool on exported JPEGs."""
    source_exif = _raw_like_source_exif()

    out = embed_metadata(_jpeg(), MetadataConfig(), source_exif)

    loaded = piexif.load(out)
    zeroth = loaded["0th"]
    for tag in _RAW_PREVIEW_0TH_TAGS:
        assert tag not in zeroth
    assert zeroth[piexif.ImageIFD.Make] == b"NIKON CORPORATION"
    assert loaded["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] == 60


@pytest.mark.skipif(not shutil.which("exiftool"), reason="exiftool not installed")
def test_embed_jpeg_exiftool_can_write_user_comment(tmp_path) -> None:
    """Regression: exported JPEG EXIF must be writable by ExifTool (issue 0.32.1)."""
    jpeg = embed_metadata(_jpeg(), MetadataConfig(), _raw_like_source_exif())
    path = tmp_path / "export.jpg"
    path.write_bytes(jpeg)

    result = subprocess.run(
        ["exiftool", "-overwrite_original", "-UserComment=foo", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Error" not in (result.stderr or "")


def test_embed_handles_oversized_exif_without_dropping_metadata() -> None:
    # Source EXIF far larger than the 64 KB APP1 limit (fat thumbnail + MakerNote).
    source_exif = {
        "0th": {},
        "Exif": {piexif.ExifIFD.MakerNote: b"\x00" * 70_000},
        "GPS": {},
        "Interop": {},
        "1st": {},
        "thumbnail": b"\xff\xd8" + b"\x00" * 70_000,
    }

    out = embed_metadata(_jpeg(), MetadataConfig(), source_exif)

    # Embed succeeded (didn't fall back to the original): our normalized orientation is present.
    loaded = piexif.load(out)
    assert loaded["0th"][piexif.ImageIFD.Orientation] == 1
    # The oversized blobs were trimmed to fit.
    assert b"thumbnail" not in loaded or not loaded.get("thumbnail")
    assert piexif.ExifIFD.MakerNote not in loaded["Exif"]


def test_embed_trims_oversized_nonstandard_tag_keeps_custom_fields() -> None:
    # Overflow from a tag the targeted trims don't touch (e.g. bloated ImageDescription/XMP).
    source_exif = {
        "0th": {piexif.ImageIFD.ImageDescription: b"x" * 70_000},
        "Exif": {},
        "GPS": {},
        "Interop": {},
        "1st": {},
    }
    config = MetadataConfig(camera_model="MyCam")

    out = embed_metadata(_jpeg(), config, source_exif)

    loaded = piexif.load(out)
    # Embed succeeded (no fallback to original) and the user's field survived.
    assert loaded["0th"][piexif.ImageIFD.Orientation] == 1
    assert loaded["0th"][piexif.ImageIFD.Model] == b"MyCam"


def test_embed_keeps_small_exif_intact() -> None:
    source_exif = {
        "0th": {piexif.ImageIFD.Make: b"TestCam"},
        "Exif": {},
        "GPS": {},
        "Interop": {},
        "1st": {},
    }
    out = embed_metadata(_jpeg(), MetadataConfig(), source_exif)
    loaded = piexif.load(out)
    assert loaded["0th"][piexif.ImageIFD.Make] == b"TestCam"
    assert loaded["0th"][piexif.ImageIFD.Orientation] == 1


def test_embed_strips_stale_colorspace_tag_jpeg() -> None:
    """Nikon's non-standard ColorSpace=2 (Adobe RGB) describes the camera's own JPEG
    rendering, not NegPy's re-render; carried through unchanged it contradicts the
    file's real color tag and reads as untagged to a strict consumer."""
    source_exif = _raw_like_source_exif()
    source_exif["Exif"][piexif.ExifIFD.ColorSpace] = 2

    out = embed_metadata(_jpeg(), MetadataConfig(), source_exif)

    loaded = piexif.load(out)
    assert piexif.ExifIFD.ColorSpace not in loaded["Exif"]


def test_preserve_strips_stale_colorspace_tag_jpeg() -> None:
    source_exif = _raw_like_source_exif()
    source_exif["Exif"][piexif.ExifIFD.ColorSpace] = 2

    out = preserve_source_metadata(_jpeg(), source_path="unused.nef", source_exif=source_exif)

    loaded = piexif.load(out)
    assert piexif.ExifIFD.ColorSpace not in loaded["Exif"]


def test_embed_strips_stale_colorspace_tag_jxl() -> None:
    """Regression: a Nikon ColorSpace=2 carried into a JPEG XL export contradicted
    the file's own sRGB color tag and made Adobe Bridge report the file untagged."""
    source_exif = _raw_like_source_exif()
    source_exif["Exif"][piexif.ExifIFD.ColorSpace] = 2

    out = embed_metadata(_jxl(), MetadataConfig(), source_exif)

    tiff = read_jxl_exif(out)
    assert tiff is not None
    loaded = piexif.load(tiff)
    assert piexif.ExifIFD.ColorSpace not in loaded["Exif"]
    assert loaded["0th"][piexif.ImageIFD.Make] == b"NIKON CORPORATION"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
