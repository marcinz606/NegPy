"""Tests for metadata embed writer."""

import io

import numpy as np
import piexif
import tifffile

from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.writer import _sanitize_exif, embed_metadata


def _make_tiff_bytes() -> bytes:
    """16-bit RGB TIFF in the shape produced by the real export pipeline."""
    arr = np.random.randint(0, 65535, (16, 16, 3), dtype=np.uint16)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, photometric="rgb", compression="lzw")
    return buf.getvalue()


class TestSanitizeExif:
    def test_drops_rational_bytes(self) -> None:
        raw = {
            "0th": {},
            "Exif": {piexif.ExifIFD.ExposureTime: b"\x00\x01\x02\x03"},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        clean = _sanitize_exif(raw)
        assert piexif.ExifIFD.ExposureTime not in clean["Exif"]


class TestEmbedMetadata:
    def test_preserves_16bit_and_hoists_subifd_tags(self) -> None:
        """End-to-end on a tifffile-produced 16-bit RGB TIFF with stale EXIF
        sub-IFD pointer in source. Catches three regressions at once:
        - PIL round-trip would crush 16-bit to 8-bit
        - libtiff would reject the stale EXIFIFDOffset sub-IFD pointer
        - sub-IFD tags must reach the main IFD where readers can find them
        """
        image_bytes = _make_tiff_bytes()
        source_exif = {
            "0th": {
                piexif.ImageIFD.ExifTag: 0xFFFFFFFFFFFFFFFF,
                piexif.ImageIFD.Make: b"Plustek",
                piexif.ImageIFD.Model: b"OpticFilm",
            },
            "Exif": {piexif.ExifIFD.LensModel: b"Nikkor 50mm"},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        config = MetadataConfig(film="Portra 400", developer="C-41")

        out = embed_metadata(image_bytes, config, source_exif)

        assert out != image_bytes, "embed fell back to input"
        with tifffile.TiffFile(io.BytesIO(out)) as tf:
            page = tf.pages[0]
            tags = page.tags
            assert page.dtype == np.uint16
            desc = tags.get(piexif.ImageIFD.ImageDescription).value
            assert "Portra 400" in desc and "C-41" in desc
            assert tags.get(piexif.ImageIFD.Make).value == "Plustek"
            assert tags.get(piexif.ImageIFD.Model).value == "OpticFilm"
            assert tags.get(piexif.ExifIFD.LensModel).value == "Nikkor 50mm"

    def test_filters_reserved_tags_and_flattens_multi_rational(self) -> None:
        """Real scanner EXIF carries core TIFF tags (256, 257, ...) tifffile
        manages itself, plus multi-element RATIONALs (e.g. PrimaryChromaticities
        = 6 rationals). The former must be silently dropped, the latter must be
        passed as a flat int sequence — list-of-tuples blows up tifffile's
        struct.pack with ``pack expected 18 items for packing (got 9)``."""
        image_bytes = _make_tiff_bytes()
        source_exif = {
            "0th": {
                256: 4096,
                257: 2731,
                258: (16, 16, 16),
                259: 5,
                262: 2,
                273: (8, 12345),
                277: 3,
                278: 16,
                279: (8, 67890),
                282: (300, 1),
                283: (300, 1),
                284: 1,
                296: 2,
                305: b"VueScan",
                319: [(64, 100), (33, 100), (21, 100), (71, 100), (15, 100), (6, 100)],
                piexif.ImageIFD.Make: b"Plustek",
            },
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }

        out = embed_metadata(image_bytes, MetadataConfig(film="Portra 400"), source_exif)

        assert out != image_bytes
        with tifffile.TiffFile(io.BytesIO(out)) as tf:
            page = tf.pages[0]
            assert page.tags.get(256).value == page.shape[1]
            assert page.tags.get(piexif.ImageIFD.Make).value == "Plustek"
            chroma = page.tags.get(319)
            assert chroma is not None and chroma.count == 6

    def test_folds_user_comment_into_image_description(self) -> None:
        """tifffile can't write a real EXIF sub-IFD, so UserComment must be
        mirrored into ImageDescription to stay visible in viewers that only
        surface tag 270 (macOS Preview, Lightroom)."""
        out = embed_metadata(
            _make_tiff_bytes(),
            MetadataConfig(film="Portra 400", format="35mm", developer="HC-110", push_pull=1),
            source_exif=None,
        )
        with tifffile.TiffFile(io.BytesIO(out)) as tf:
            desc = tf.pages[0].tags.get(piexif.ImageIFD.ImageDescription).value
        for fragment in ("Portra 400", "35mm", "HC-110", "Push +1"):
            assert fragment in desc, f"missing {fragment!r} in {desc!r}"
