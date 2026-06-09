"""
Guards the JPEG EXIF embed against the 64 KB APP1 overflow: source RAWs often carry an
embedded thumbnail + multi-KB MakerNote that piexif.insert can't pack into one segment.
"""

import io

import piexif
import pytest
from PIL import Image

from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.writer import embed_metadata


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 0, 0)).save(buf, "JPEG")
    return buf.getvalue()


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
    config = MetadataConfig(camera_override="MyCam")

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
