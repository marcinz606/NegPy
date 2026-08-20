"""Tests for metadata embed writer."""

import io

import imagecodecs
import numpy as np
import piexif
import tifffile
from PIL import Image

from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.writer import (
    _decode_ascii,
    _read_xmp_from_source,
    _sanitize_exif,
    _webp_chunks,
    embed_metadata,
    preserve_source_metadata,
)
from negpy.infrastructure.loaders.jxl_boxes import JXL_SIGNATURE, jxl_boxes


def _make_tiff_bytes() -> bytes:
    """16-bit RGB TIFF in the shape produced by the real export pipeline."""
    arr = np.random.randint(0, 65535, (16, 16, 3), dtype=np.uint16)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, photometric="rgb", compression="zlib", predictor=True)
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

    def test_normalizes_orientation_tag_jpeg(self) -> None:
        """NegPy bakes orientation into pixels, so the exported file must declare
        Orientation=1 — otherwise viewers re-rotate the already-upright image (#218)."""
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        jpeg = io.BytesIO()
        from PIL import Image

        Image.fromarray(arr).save(jpeg, format="JPEG")
        source_exif = {
            "0th": {piexif.ImageIFD.Orientation: 6, piexif.ImageIFD.Make: b"Nikon"},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {piexif.ImageIFD.Orientation: 6},
        }

        out = embed_metadata(jpeg.getvalue(), MetadataConfig(), source_exif)

        exif = piexif.load(out)
        assert exif["0th"].get(piexif.ImageIFD.Orientation) == 1
        assert piexif.ImageIFD.Orientation not in exif["1st"]

    def test_embeds_into_png_and_preserves_icc(self) -> None:
        """PNG export must not be routed through the TIFF path (it raised
        'not a TIFF file: header=\\x89PNG'). EXIF goes into an eXIf chunk and the
        embedded ICC profile survives the re-save."""
        from PIL import Image

        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        png = io.BytesIO()
        icc = b"fake-icc-profile-bytes"
        Image.fromarray(arr).save(png, format="PNG", icc_profile=icc)
        source_exif = {
            "0th": {piexif.ImageIFD.Make: b"Plustek", piexif.ImageIFD.Orientation: 6},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }

        out = embed_metadata(png.getvalue(), MetadataConfig(film="Portra 400"), source_exif)

        assert out != png.getvalue(), "embed fell back to input"
        assert out[:8] == b"\x89PNG\r\n\x1a\n"
        with Image.open(io.BytesIO(out)) as im:
            assert im.info.get("icc_profile") == icc
            exif = im.getexif()
            assert exif.get(piexif.ImageIFD.Make) == "Plustek"
            assert exif.get(piexif.ImageIFD.Orientation) == 1

    def test_normalizes_orientation_tag_tiff(self) -> None:
        source_exif = {
            "0th": {piexif.ImageIFD.Orientation: 8, piexif.ImageIFD.Make: b"Plustek"},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        out = embed_metadata(_make_tiff_bytes(), MetadataConfig(), source_exif)
        with tifffile.TiffFile(io.BytesIO(out)) as tf:
            ori = tf.pages[0].tags.get(piexif.ImageIFD.Orientation)
            # tifffile defaults Orientation to 1 when not emitted; explicit 1 also fine.
            assert ori is None or ori.value == 1

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


class TestPreserveSourceMetadata:
    def test_copies_source_exif_without_negpy_software(self) -> None:
        image_bytes = _make_tiff_bytes()
        source_exif = {
            "0th": {
                piexif.ImageIFD.Make: b"Nikon",
                piexif.ImageIFD.Model: b"F6",
                piexif.ImageIFD.Software: b"MV-1 Recorder",
            },
            "Exif": {piexif.ExifIFD.LensModel: b"Nikkor 50mm f/1.8"},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        config = MetadataConfig(film="Portra 400", developer="C-41")

        embedded = embed_metadata(image_bytes, config, source_exif)
        preserved = preserve_source_metadata(image_bytes, "/unused/source.dng", source_exif)

        with tifffile.TiffFile(io.BytesIO(embedded)) as tf:
            tags = tf.pages[0].tags
            desc = tags.get(piexif.ImageIFD.ImageDescription).value
            assert "Portra 400" in desc and "C-41" in desc

        with tifffile.TiffFile(io.BytesIO(preserved)) as tf:
            tags = tf.pages[0].tags
            assert tags.get(piexif.ImageIFD.Make).value == "Nikon"
            assert tags.get(piexif.ImageIFD.Model).value == "F6"
            assert tags.get(piexif.ImageIFD.Software).value == "MV-1 Recorder"
            assert tags.get(piexif.ExifIFD.LensModel).value == "Nikkor 50mm f/1.8"

    def test_does_not_normalize_orientation(self) -> None:
        from PIL import Image

        jpeg = io.BytesIO()
        Image.new("RGB", (16, 16), (128, 0, 0)).save(jpeg, "JPEG")
        source_exif = {
            "0th": {piexif.ImageIFD.Orientation: 6, piexif.ImageIFD.Make: b"Nikon"},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }

        out = preserve_source_metadata(jpeg.getvalue(), "/unused/source.dng", source_exif)
        loaded = piexif.load(out)
        assert loaded["0th"].get(piexif.ImageIFD.Orientation) == 6
        assert loaded["0th"].get(piexif.ImageIFD.Make) == b"Nikon"


class TestDecodeAscii:
    """_decode_ascii must always return pure-ASCII str (#452)."""

    def test_bytes_with_non_ascii(self) -> None:
        assert _decode_ascii(b"4\xd75 negative") == "4?5 negative"

    def test_str_with_non_ascii(self) -> None:
        assert _decode_ascii("4\u00d75 negative") == "4?5 negative"

    def test_pure_ascii_bytes_unchanged(self) -> None:
        assert _decode_ascii(b"Portra 400") == "Portra 400"

    def test_pure_ascii_str_unchanged(self) -> None:
        assert _decode_ascii("Portra 400") == "Portra 400"

    def test_null_terminated_bytes_stripped(self) -> None:
        assert _decode_ascii(b"Hello\x00World\x00") == "Hello\x00World"

    def test_none_returns_none(self) -> None:
        assert _decode_ascii(None) is None
        assert _decode_ascii(42) is None

    def test_non_ascii_exif_does_not_crash_tiff_metadata_embed(self) -> None:
        """Regression: non-ASCII EXIF description must not crash tifffile (#452)."""
        image_bytes = _make_tiff_bytes()
        source_exif = {
            "0th": {
                piexif.ImageIFD.Make: b"Nikon",
                piexif.ImageIFD.ImageDescription: "4\u00d75 - Portra 400",
            },
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        out = embed_metadata(image_bytes, MetadataConfig(), source_exif)
        assert out != image_bytes
        with tifffile.TiffFile(io.BytesIO(out)) as tf:
            desc = tf.pages[0].tags.get("ImageDescription")
            # ASCII-safe -- tifffile would reject non-ASCII
            desc.value.encode("ascii")

    def test_film_format_with_non_ascii_does_not_crash(self) -> None:
        """Regression: FilmFormat uses x (U+00D7) in '4x5'/'8x10', must not crash export (#487)."""
        image_bytes = _make_tiff_bytes()
        config = MetadataConfig(format="4×5")
        out = embed_metadata(image_bytes, config, None)
        assert out != image_bytes
        with tifffile.TiffFile(io.BytesIO(out)) as tf:
            desc = tf.pages[0].tags.get("ImageDescription")
            desc.value.encode("ascii")

    def test_user_comment_fold_with_non_ascii_bytes(self) -> None:
        """Regression: UserComment with non-ASCII bytes must not produce \ufffd in description (#487)."""
        image_bytes = _make_tiff_bytes()
        # Simulate source EXIF with a UserComment containing non-ASCII bytes
        uc_body = b"Film: 4\xd75 - Portra 400"
        uc_raw = b"ASCII\x00\x00\x00" + uc_body
        source_exif = {
            "0th": {},
            "Exif": {piexif.ExifIFD.UserComment: uc_raw},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        out = embed_metadata(image_bytes, MetadataConfig(), source_exif)
        assert out != image_bytes
        with tifffile.TiffFile(io.BytesIO(out)) as tf:
            desc = tf.pages[0].tags.get("ImageDescription")
            desc.value.encode("ascii")


def _jpeg_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, "JPEG")
    return buf.getvalue()


def _source_exif(exif: dict | None = None) -> dict:
    return {"0th": {}, "Exif": dict(exif or {}), "GPS": {}, "Interop": {}, "1st": {}}


class TestCaptureDateAndPlace:
    def test_capture_date_replaces_the_scan_timestamp(self) -> None:
        source = _source_exif({piexif.ExifIFD.DateTimeOriginal: b"2026:07:03 18:51:59"})
        out = embed_metadata(_jpeg_bytes(), MetadataConfig(capture_date="1998-07-14 16:30+02:00"), source)
        exif = piexif.load(out)["Exif"]
        assert exif[piexif.ExifIFD.DateTimeOriginal] == b"1998:07:14 16:30:00"
        assert exif[piexif.ExifIFD.OffsetTimeOriginal] == b"+02:00"
        # The scan is what was digitized, so its timestamp moves to that tag.
        assert exif[piexif.ExifIFD.DateTimeDigitized] == b"2026:07:03 18:51:59"

    def test_partial_date_is_padded_for_exif_and_kept_in_xmp(self) -> None:
        out = embed_metadata(_jpeg_bytes(), MetadataConfig(capture_date="1998"), _source_exif())
        assert piexif.load(out)["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"1998:01:01 00:00:00"
        assert b"<photoshop:DateCreated>1998</photoshop:DateCreated>" in out
        assert b"<negpy:CaptureDatePrecision>year</negpy:CaptureDatePrecision>" in out

    def test_unset_capture_date_leaves_the_source_timestamp_alone(self) -> None:
        source = _source_exif({piexif.ExifIFD.DateTimeOriginal: b"2026:07:03 18:51:59"})
        out = embed_metadata(_jpeg_bytes(), MetadataConfig(), source)
        exif = piexif.load(out)["Exif"]
        assert exif[piexif.ExifIFD.DateTimeOriginal] == b"2026:07:03 18:51:59"
        assert piexif.ExifIFD.DateTimeDigitized not in exif

    def test_gps_ifd_and_place_names(self) -> None:
        config = MetadataConfig(
            gps_latitude=-33.8688,
            gps_longitude=151.2093,
            location_city="Sydney",
            location_country="Australia",
        )
        out = embed_metadata(_jpeg_bytes(), config, _source_exif())
        gps = piexif.load(out)["GPS"]
        assert gps[piexif.GPSIFD.GPSLatitudeRef] == b"S"
        assert gps[piexif.GPSIFD.GPSLongitudeRef] == b"E"
        assert gps[piexif.GPSIFD.GPSMapDatum] == b"WGS-84"
        assert b"<photoshop:City>Sydney</photoshop:City>" in out
        assert b"<exif:GPSLatitude>" in out

    def test_tiff_carries_location_in_xmp_and_not_as_top_level_tags(self) -> None:
        """GPS tag numbers 1-4 are not TIFF tags; a TIFF gets its location from XMP."""
        config = MetadataConfig(gps_latitude=35.6762, gps_longitude=139.6503)
        out = embed_metadata(_make_tiff_bytes(), config, _source_exif())
        with tifffile.TiffFile(io.BytesIO(out)) as tf:
            codes = {tag.code for tag in tf.pages[0].tags}
        assert not codes & {1, 2, 3, 4}
        assert b"<exif:GPSLongitude>" in out


class TestSourceGps:
    _SOURCE_GPS = {
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLatitude: ((51, 1), (30, 1), (0, 100)),
        piexif.GPSIFD.GPSLongitudeRef: b"W",
        piexif.GPSIFD.GPSLongitude: ((0, 1), (7, 1), (3900, 100)),
        piexif.GPSIFD.GPSAltitude: (35, 1),
    }

    def _with_gps(self) -> dict:
        source = _source_exif()
        source["GPS"] = dict(self._SOURCE_GPS)
        return source

    def test_source_position_survives_when_no_place_is_set(self) -> None:
        out = embed_metadata(_jpeg_bytes(), MetadataConfig(), self._with_gps())
        assert piexif.load(out)["GPS"] == self._SOURCE_GPS

    def test_picked_place_replaces_the_whole_source_block(self) -> None:
        """Keeping the scan's altitude or heading beside our coordinates would mix two places."""
        config = MetadataConfig(gps_latitude=35.6762, gps_longitude=139.6503)
        out = embed_metadata(_jpeg_bytes(), config, self._with_gps())
        gps = piexif.load(out)["GPS"]
        assert piexif.GPSIFD.GPSAltitude not in gps
        assert gps[piexif.GPSIFD.GPSLongitudeRef] == b"E"


def _jxl_bytes() -> bytes:
    """16-bit RGB JPEG XL in the shape produced by the real export pipeline."""
    arr = np.random.default_rng(0).integers(0, 65535, (16, 16, 3), dtype=np.uint16)
    return bytes(
        imagecodecs.jpegxl_encode(
            arr,
            bitspersample=16,
            photometric="RGB",
            primaries="SRGB",
            transfer="SRGB",
            lossless=True,
            effort=1,
        )
    )


def _jxl_box_types(data: bytes) -> list[bytes]:
    return [btype for btype, _start, _end in jxl_boxes(data)]


class TestJxlMetadata:
    def test_exif_and_xmp_boxes_reach_the_container(self) -> None:
        source_exif = {
            "0th": {piexif.ImageIFD.Make: b"Plustek"},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        config = MetadataConfig(film="Portra 400", camera_make="Nikon", camera_model="FM2")

        out = embed_metadata(_jxl_bytes(), config, source_exif)

        types = _jxl_box_types(out)
        assert b"Exif" in types and b"xml " in types
        exif_at, codestream_at = types.index(b"Exif"), types.index(b"jxlc")
        assert exif_at < codestream_at, "metadata must precede the codestream"

    def test_exif_box_payload_is_offset_plus_tiff_header(self) -> None:
        """The JPEG XL Exif box holds a 4-byte offset to the TIFF header, not the
        JPEG 'Exif\\x00\\x00' APP1 prefix."""
        out = embed_metadata(_jxl_bytes(), MetadataConfig(film="Portra 400"), None)

        payload = next(out[start:end] for btype, start, end in jxl_boxes(out) if btype == b"Exif")
        assert payload[:4] == b"\x00\x00\x00\x00"
        assert payload[4:6] in (b"MM", b"II")
        assert piexif.load(payload[4:])["0th"][piexif.ImageIFD.Software] == b"NegPy"

    def test_pixels_survive_the_rewrite(self) -> None:
        image_bytes = _jxl_bytes()
        out = embed_metadata(image_bytes, MetadataConfig(film="Portra 400"), None)
        np.testing.assert_array_equal(imagecodecs.jpegxl_decode(out), imagecodecs.jpegxl_decode(image_bytes))

    def test_re_embedding_replaces_rather_than_stacks_boxes(self) -> None:
        config = MetadataConfig(film="Portra 400")
        once = embed_metadata(_jxl_bytes(), config, None)
        twice = embed_metadata(once, config, None)
        assert twice == once

    def test_preserve_copies_source_exif_without_negpy_software(self) -> None:
        source_exif = {
            "0th": {piexif.ImageIFD.Make: b"Nikon", piexif.ImageIFD.Software: b"MV-1 Recorder"},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        out = preserve_source_metadata(_jxl_bytes(), "/unused/source.dng", source_exif)

        payload = next(out[start:end] for btype, start, end in jxl_boxes(out) if btype == b"Exif")
        zeroth = piexif.load(payload[4:])["0th"]
        assert zeroth[piexif.ImageIFD.Make] == b"Nikon"
        assert zeroth[piexif.ImageIFD.Software] == b"MV-1 Recorder"

    def test_bare_codestream_is_wrapped_in_a_container(self) -> None:
        """imagecodecs can emit a container-less codestream, which has nowhere to
        hold metadata boxes."""
        bare = bytes(imagecodecs.jpegxl_encode(np.zeros((8, 8, 3), dtype=np.uint8), lossless=True, effort=1, usecontainer=False))
        assert bare[:2] == b"\xff\x0a"

        out = embed_metadata(bare, MetadataConfig(film="Portra 400"), None)

        assert out[:12] == JXL_SIGNATURE
        assert _jxl_box_types(out) == [b"JXL ", b"ftyp", b"Exif", b"xml ", b"jxlc"]
        np.testing.assert_array_equal(imagecodecs.jpegxl_decode(out), imagecodecs.jpegxl_decode(bare))

    def test_malformed_container_falls_back_to_the_input(self) -> None:
        broken = JXL_SIGNATURE + b"\x00\x00\xff\xffjxlc"
        assert embed_metadata(broken, MetadataConfig(film="Portra 400"), None) == broken


def _webp_bytes(**save_kwargs) -> bytes:
    """Lossy WebP in the shape produced by the real export pipeline."""
    arr = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="WEBP", quality=80, **save_kwargs)
    return buf.getvalue()


def _webp_chunk_names(data: bytes) -> list[bytes]:
    return [fourcc for fourcc, _start, _end in _webp_chunks(data)]


class TestWebpMetadata:
    def test_exif_and_xmp_chunks_reach_the_container(self) -> None:
        source_exif = {
            "0th": {piexif.ImageIFD.Make: b"Plustek"},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        out = embed_metadata(_webp_bytes(), MetadataConfig(film="Portra 400"), source_exif)

        names = _webp_chunk_names(out)
        assert names[0] == b"VP8X", "metadata needs the extended container"
        assert b"EXIF" in names and b"XMP " in names
        payload = next(out[s:e] for f, s, e in _webp_chunks(out) if f == b"EXIF")
        assert piexif.load(payload)["0th"][piexif.ImageIFD.Make] == b"Plustek"

    def test_vp8x_flags_announce_what_the_file_carries(self) -> None:
        """A reader that trusts the flags over the chunk list must still find both."""
        out = embed_metadata(_webp_bytes(), MetadataConfig(film="Portra 400"), None)

        flags = next(out[s:e] for f, s, e in _webp_chunks(out) if f == b"VP8X")[0]
        assert flags & 0x08, "EXIF flag"
        assert flags & 0x04, "XMP flag"

    def test_icc_profile_survives_and_keeps_its_place(self) -> None:
        """ICCP must follow VP8X, and PIL embeds one on every color-managed export."""
        icc = b"\x00" * 128
        out = embed_metadata(_webp_bytes(icc_profile=icc), MetadataConfig(film="Portra 400"), None)

        names = _webp_chunk_names(out)
        assert names[:2] == [b"VP8X", b"ICCP"]
        assert next(out[s:e] for f, s, e in _webp_chunks(out) if f == b"ICCP") == icc
        with Image.open(io.BytesIO(out)) as im:
            assert im.info.get("icc_profile") == icc

    def test_pixels_are_not_re_encoded(self) -> None:
        """The lossy codestream is copied, never decoded and re-compressed."""
        image_bytes = _webp_bytes()
        out = embed_metadata(image_bytes, MetadataConfig(film="Portra 400"), None)

        original = next(image_bytes[s:e] for f, s, e in _webp_chunks(image_bytes) if f in (b"VP8 ", b"VP8L"))
        assert next(out[s:e] for f, s, e in _webp_chunks(out) if f in (b"VP8 ", b"VP8L")) == original

    def test_re_embedding_replaces_rather_than_stacks_chunks(self) -> None:
        config = MetadataConfig(film="Portra 400")
        once = embed_metadata(_webp_bytes(), config, None)
        twice = embed_metadata(once, config, None)
        assert twice == once

    def test_odd_sized_payload_stays_word_aligned(self) -> None:
        """A RIFF chunk pads to an even size, and the pad is outside the stated size."""
        out = embed_metadata(_webp_bytes(), MetadataConfig(film="Portra 400"), None)
        for _fourcc, start, end in _webp_chunks(out):
            assert start % 2 == 0, "chunk payload must start on an even offset"
        assert len(out) == 8 + int.from_bytes(out[4:8], "little")

    def test_malformed_container_falls_back_to_the_input(self) -> None:
        broken = b"RIFF" + (20).to_bytes(4, "little") + b"WEBPVP8 " + (999).to_bytes(4, "little")
        assert embed_metadata(broken, MetadataConfig(film="Portra 400"), None) == broken


class TestExifAsciiTransliteration:
    def test_description_separator_survives_as_ascii(self) -> None:
        """ImageDescription joins its parts with a bullet, which is not ASCII."""
        config = MetadataConfig(camera_make="Nikon", camera_model="FM2", film="Portra 400")
        out = embed_metadata(_jpeg_bytes(), config, None)

        description = piexif.load(out)["0th"][piexif.ImageIFD.ImageDescription].decode("ascii")
        assert "?" not in description
        assert description == "Nikon FM2 - Portra 400"

    def test_sheet_film_size_keeps_its_dimensions(self) -> None:
        """4×5 must not reach EXIF as 4?5."""
        out = embed_metadata(_jpeg_bytes(), MetadataConfig(format="4×5", film="HP5"), None)

        comment = piexif.load(out)["Exif"][piexif.ExifIFD.UserComment].decode("ascii")
        assert "4x5" in comment and "?" not in comment


class TestTiffSoftware:
    def test_negpy_is_named_as_the_writer(self) -> None:
        """tifffile stamps its own module name unless the caller passes one, so the
        Software tag has to be handed over with the rest of the EXIF."""
        out = embed_metadata(_make_tiff_bytes(), MetadataConfig(film="Portra 400"), None)

        with tifffile.TiffFile(io.BytesIO(out)) as tf:
            assert tf.pages[0].tags["Software"].value == "NegPy"

    def test_passthrough_keeps_the_scanner_software(self) -> None:
        source_exif = {
            "0th": {piexif.ImageIFD.Software: b"SilverFast"},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        out = preserve_source_metadata(_make_tiff_bytes(), "/unused/source.dng", source_exif)

        with tifffile.TiffFile(io.BytesIO(out)) as tf:
            assert tf.pages[0].tags["Software"].value == "SilverFast"


class TestJxlSourceMetadata:
    def test_xmp_is_read_back_out_of_a_jxl_scan(self, tmp_path) -> None:
        """A JXL source is as valid a scan as a TIFF, and protect-original has to
        find its XMP to copy it."""
        scan = embed_metadata(_jxl_bytes(), MetadataConfig(film="Portra 400"), None)
        path = tmp_path / "scan.jxl"
        path.write_bytes(scan)

        xmp = _read_xmp_from_source(str(path))

        assert xmp is not None and b"Portra 400" in xmp
