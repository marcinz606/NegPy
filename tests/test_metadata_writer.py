"""
Guards the EXIF embed/preserve paths against stale or oversized source metadata: RAW
preview-IFD and maker-note pointers that are absolute to the source's own byte layout,
a ColorSpace tag describing the camera's own rendering rather than NegPy's, and a
64 KB JPEG APP1 overflow from an oversized thumbnail + maker note.
"""

import io
import shutil
import subprocess

import piexif
import pytest
from PIL import Image

import imagecodecs
import numpy as np

from negpy.features.metadata.exif_read import source_xmp_rating
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.writer import _extract_jpeg_xmp, embed_metadata, export_embed_plan, preserve_source_metadata
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
    """A Nikon ColorSpace=2 carried into a JPEG XL export would contradict the
    file's own sRGB color tag."""
    source_exif = _raw_like_source_exif()
    source_exif["Exif"][piexif.ExifIFD.ColorSpace] = 2

    out = embed_metadata(_jxl(), MetadataConfig(), source_exif)

    tiff = read_jxl_exif(out)
    assert tiff is not None
    loaded = piexif.load(tiff)
    assert piexif.ExifIFD.ColorSpace not in loaded["Exif"]
    assert loaded["0th"][piexif.ImageIFD.Make] == b"NIKON CORPORATION"


def test_embed_strips_raw_preview_ifd_tags_from_jxl() -> None:
    """Regression: a RAW's 0th-IFD preview/SubIFD pointers are absolute to the
    source file's own byte layout. Carried into a JPEG XL export unchanged, they
    pointed at the wrong bytes ("Bad SubIFD SubDirectory start" from ExifTool) and
    made Adobe Bridge report the file untagged and refuse to preview it."""
    source_exif = _raw_like_source_exif()

    out = embed_metadata(_jxl(), MetadataConfig(), source_exif)

    tiff = read_jxl_exif(out)
    assert tiff is not None
    loaded = piexif.load(tiff)
    zeroth = loaded["0th"]
    for tag in _RAW_PREVIEW_0TH_TAGS:
        assert tag not in zeroth
    assert zeroth[piexif.ImageIFD.Make] == b"NIKON CORPORATION"


def test_embed_strips_raw_capture_tags_from_jxl() -> None:
    """Regression: CFAPattern/SensingMethod/FileSource/SceneType/TIFFEPStandardID
    describe the original sensor capture, not NegPy's finished render. Carried
    through, they told Adobe Bridge the file was TIFF/EP raw-compliant, and it
    tried to route a rendered JPEG XL through Camera Raw instead of previewing it
    directly -- landing on a slow, low-res-only preview with no way to open it."""
    source_exif = _raw_like_source_exif()
    source_exif["0th"][piexif.ImageIFD.TIFFEPStandardID] = (1, 0, 0, 0)
    source_exif["Exif"][piexif.ExifIFD.CFAPattern] = b"\x00\x02\x00\x02\x00\x01\x01\x02"
    source_exif["Exif"][piexif.ExifIFD.SensingMethod] = 2
    source_exif["Exif"][piexif.ExifIFD.FileSource] = b"\x03"
    source_exif["Exif"][piexif.ExifIFD.SceneType] = b"\x01"

    out = embed_metadata(_jxl(), MetadataConfig(), source_exif)

    tiff = read_jxl_exif(out)
    assert tiff is not None
    loaded = piexif.load(tiff)
    assert piexif.ImageIFD.TIFFEPStandardID not in loaded["0th"]
    for tag in (
        piexif.ExifIFD.CFAPattern,
        piexif.ExifIFD.SensingMethod,
        piexif.ExifIFD.FileSource,
        piexif.ExifIFD.SceneType,
    ):
        assert tag not in loaded["Exif"]
    assert loaded["0th"][piexif.ImageIFD.Make] == b"NIKON CORPORATION"


def test_embed_strips_makernote_tag() -> None:
    """A maker note's internal sub-IFDs use the same source-relative pointer
    scheme as the 0th-IFD preview tags; relocated into an export, they resolve
    to the wrong bytes for anything that decodes into them."""
    source_exif = _raw_like_source_exif()
    source_exif["Exif"][piexif.ExifIFD.MakerNote] = b"\x00" * 100

    out = embed_metadata(_jpeg(), MetadataConfig(), source_exif)

    loaded = piexif.load(out)
    assert piexif.ExifIFD.MakerNote not in loaded["Exif"]


def test_embed_strips_vestigial_gps_block_from_jxl() -> None:
    """Regression: a camera's GPS IFD with no actual coordinates -- just a
    version marker, common on bodies with no GPS receiver -- is vestigial
    boilerplate. Isolated by an 8-file bisection, its bare presence (not
    exposure, date, Artist/Copyright, or resolution tags, all of which were
    also tested and are safe) was the one difference between a Print/Flat
    JPEG XL export Adobe Bridge opened correctly and one it called untagged
    and refused to preview, routing it through Camera Raw instead."""
    source_exif = _raw_like_source_exif()
    source_exif["GPS"] = {piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0)}

    out = embed_metadata(_jxl(), MetadataConfig(), source_exif)

    tiff = read_jxl_exif(out)
    assert tiff is not None
    loaded = piexif.load(tiff)
    assert loaded["GPS"] == {}
    assert loaded["0th"][piexif.ImageIFD.Make] == b"NIKON CORPORATION"


def test_embed_keeps_real_gps_coordinates() -> None:
    """A GPS block with actual coordinates is real, user-relevant data (unlike
    a bare version marker) and is not part of the vestigial-block strip."""
    source_exif = _raw_like_source_exif()
    source_exif["GPS"] = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLatitude: ((59, 1), (54, 1), (0, 1)),
        piexif.GPSIFD.GPSLongitudeRef: b"E",
        piexif.GPSIFD.GPSLongitude: ((10, 1), (44, 1), (0, 1)),
    }

    out = embed_metadata(_jxl(), MetadataConfig(), source_exif)

    tiff = read_jxl_exif(out)
    assert tiff is not None
    loaded = piexif.load(tiff)
    assert loaded["GPS"][piexif.GPSIFD.GPSLatitude] == ((59, 1), (54, 1), (0, 1))


def _camera_xmp(rating: str = "3") -> bytes:
    return (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="' + rating.encode() + b'"/>'
        b'</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    )


def _with_camera_xmp(packet: bytes | None = None) -> dict:
    """Source EXIF as piexif.load() returns it: tag 700 is a tuple of byte values."""
    source_exif = _raw_like_source_exif()
    source_exif["0th"][700] = tuple(packet or _camera_xmp())
    return source_exif


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


def _webp() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 0, 0)).save(buf, "WEBP")
    return buf.getvalue()


def _exported_zeroth(out: bytes) -> dict:
    if out[:2] == b"\xff\xd8":
        tiff = out
    elif out[:4] == b"RIFF" or out[:8] == b"\x89PNG\r\n\x1a\n":
        tiff = Image.open(io.BytesIO(out)).info.get("exif")
    else:
        tiff = read_jxl_exif(out)
    assert tiff
    return piexif.load(tiff)["0th"]


@pytest.mark.parametrize("encode", [_jpeg, _jxl, _png, _webp])
def test_embed_drops_source_xmp_packet_from_exif(encode) -> None:
    """A TIFF-based RAW (Panasonic RW2) keeps its XMP in IFD0 tag 700; the export
    carries NegPy's XMP only, not a second packet inside EXIF (#1367)."""
    zeroth = _exported_zeroth(embed_metadata(encode(), MetadataConfig(), _with_camera_xmp()))
    assert 700 not in zeroth
    assert zeroth[piexif.ImageIFD.Make] == b"NIKON CORPORATION"


def test_embed_carries_source_rating_into_negpy_xmp() -> None:
    out = embed_metadata(_jpeg(), MetadataConfig(), _with_camera_xmp(_camera_xmp("3")))
    xmp = _extract_jpeg_xmp(out)
    assert xmp is not None
    assert b"<xmp:Rating>3</xmp:Rating>" in xmp


def test_embed_plan_carries_source_rating() -> None:
    plan = export_embed_plan(MetadataConfig(), _with_camera_xmp(_camera_xmp("-1")), "unused.rw2")
    assert plan is not None
    _, xmp_bytes, _ = plan
    assert xmp_bytes is not None and b"<xmp:Rating>-1</xmp:Rating>" in xmp_bytes


@pytest.mark.parametrize(
    "packet, expected",
    [
        (_camera_xmp("4"), "4"),
        (_camera_xmp("0"), ""),
        (_camera_xmp("7"), ""),
        (_camera_xmp("high"), ""),
        (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/"><xmp:Rating>2</xmp:Rating></rdf:Description>'
            b"</rdf:RDF></x:xmpmeta>\x00\x00",
            "2",
        ),
        (b"<x:xmpmeta", ""),
    ],
)
def test_source_xmp_rating(packet: bytes, expected: str) -> None:
    assert source_xmp_rating(_with_camera_xmp(packet)) == expected


def test_preserve_moves_source_xmp_packet_out_of_exif() -> None:
    """tifffile cannot open an RW2, so Protect original metadata reads its XMP from
    the EXIF and writes it to the export's own XMP slot."""
    out = preserve_source_metadata(_jpeg(), source_path="unused.rw2", source_exif=_with_camera_xmp())
    assert 700 not in _exported_zeroth(out)
    assert _extract_jpeg_xmp(out) == _camera_xmp()


def test_preserve_embed_plan_keeps_source_xmp_packet() -> None:
    plan = export_embed_plan(MetadataConfig(protect_original_metadata=True), _with_camera_xmp(), "unused.rw2")
    assert plan is not None
    exif_bytes, xmp_bytes, _ = plan
    assert 700 not in piexif.load(exif_bytes)["0th"]
    assert xmp_bytes == _camera_xmp()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize(
    ("rational", "shown"),
    [((10, 300), "1/30s"), ((1, 30), "1/30s"), ((10, 25), "0.4s"), ((13, 10), "1.3s"), ((10, 75), "1/7.5s"), ((5, 0), "")],
)
def test_the_exposure_time_shows_as_the_camera_marks_it_and_parses_back(rational, shown):
    from fractions import Fraction

    from negpy.features.metadata.exif_read import format_exposure
    from negpy.features.metadata.writer import _parse_exposure_str

    text = format_exposure({piexif.ExifIFD.ExposureTime: rational})

    assert text == shown
    if shown:
        assert Fraction(*_parse_exposure_str(text)[piexif.ExifIFD.ExposureTime]) == Fraction(*rational)
