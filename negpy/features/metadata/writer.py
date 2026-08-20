"""Pure functions to embed custom metadata into exported image bytes via piexif + XMP."""

import copy
import io
import logging
import re
import struct
from fractions import Fraction
from typing import Any, Optional

import piexif
import tifffile
from PIL import Image, PngImagePlugin

from negpy.features.metadata.capture import exif_gps_rationals
from negpy.features.metadata.exif_read import strip_scan_exif_for_capture, strip_scan_gps
from negpy.features.metadata.gear_models import GearLibrary
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.payload import NEGPY_SOFTWARE, MetadataPayload, build_metadata_payload
from negpy.features.metadata.xmp import build_xmp_bytes
from negpy.infrastructure.loaders.jxl_boxes import (
    JXL_CODESTREAM,
    JXL_EXIF_BOX,
    JXL_SIGNATURE,
    JXL_XMP_BOX,
    is_jxl,
    jxl_boxes,
    read_jxl_xmp,
)
from negpy.services.assets.gear import GearProfiles

_log = logging.getLogger(__name__)

_XMP_APP1_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
_TIFF_XMP_TAG = 700  # XMLPacket

_JXL_FTYP = b"jxl \x00\x00\x00\x00jxl "
_JXL_CODESTREAM_BOXES = frozenset({b"jxlc", b"jxlp"})
_JXL_METADATA_BOXES = frozenset({JXL_EXIF_BOX, JXL_XMP_BOX})

_WEBP_METADATA_CHUNKS = frozenset({b"EXIF", b"XMP "})
# VP8X feature flags, MSB first: reserved(2), ICC, alpha, EXIF, XMP, animation, reserved.
_VP8X_ICC = 0x20
_VP8X_ALPHA = 0x10
_VP8X_EXIF = 0x08
_VP8X_XMP = 0x04


def _parse_exposure_str(text: str) -> dict:
    """
    Parse a free-form exposure string like '1/125s f/2.8 ISO 400' into
    piexif-format rational tuples for ExposureTime, FNumber, and ISOSpeedRatings.
    Returns an empty dict if parsing fails.
    """
    result: dict = {}

    m_shutter = re.search(r"(\d+(?:/\d+)?(?:\.\d+)?)\s*s", text)
    if m_shutter:
        val = m_shutter.group(1)
        if "/" in val:
            num_str, den_str = val.split("/")
            result[piexif.ExifIFD.ExposureTime] = (int(num_str), int(den_str))
        elif "." in val:
            f = Fraction(val)
            result[piexif.ExifIFD.ExposureTime] = (f.numerator, f.denominator)
        else:
            result[piexif.ExifIFD.ExposureTime] = (int(val), 1)

    m_aperture = re.search(r"f/\s*(\d+(?:\.\d+)?)", text)
    if m_aperture:
        val = m_aperture.group(1)
        if "." in val:
            int_part, frac_part = val.split(".")
            den = 10 ** len(frac_part)
            num = int(int_part) * den + int(frac_part)
            result[piexif.ExifIFD.FNumber] = (num, den)
        else:
            result[piexif.ExifIFD.FNumber] = (int(val), 1)

    m_iso = re.search(r"ISO\s*(\d+)", text)
    if m_iso:
        iso_val = int(m_iso.group(1))
        result[piexif.ExifIFD.ISOSpeedRatings] = iso_val

    return result


def _rational_tuple(value: float) -> tuple[int, int]:
    f = Fraction(value).limit_denominator(1000)
    return f.numerator, f.denominator


def _apex_from_f_number(f_number: float) -> float:
    import math

    return 2.0 * math.log(f_number, 2.0)


_ASCII_SUBSTITUTIONS = str.maketrans(
    {
        "\u2022": "-",  # the bullet joining ImageDescription parts
        "\u00d7": "x",  # sheet-film sizes: 4x5, 8x10
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def _exif_ascii(text: str) -> bytes:
    """EXIF ASCII holds 7-bit only, so transliterate the punctuation NegPy and its
    users write before the encoder turns it into '?'."""
    return text.translate(_ASCII_SUBSTITUTIONS).encode("ascii", errors="replace")


def _build_custom_exif(payload: MetadataPayload) -> dict:
    """Build a piexif-format EXIF dict from a resolved metadata payload."""

    zeroth: dict = {}
    exif: dict = {}
    flags = payload.exif_flags

    if payload.image_description:
        zeroth[piexif.ImageIFD.ImageDescription] = _exif_ascii(payload.image_description)
    elif payload.film_stock:
        zeroth[piexif.ImageIFD.ImageDescription] = _exif_ascii(payload.film_stock)

    zeroth[piexif.ImageIFD.Software] = _exif_ascii(NEGPY_SOFTWARE)

    if flags.camera:
        if payload.camera_make:
            zeroth[piexif.ImageIFD.Make] = _exif_ascii(payload.camera_make)
        if payload.camera_model:
            zeroth[piexif.ImageIFD.Model] = _exif_ascii(payload.camera_model)

    if flags.lens:
        if payload.lens_make:
            exif[piexif.ExifIFD.LensMake] = _exif_ascii(payload.lens_make)
        if payload.lens_model:
            exif[piexif.ExifIFD.LensModel] = _exif_ascii(payload.lens_model)
        if payload.focal_length_mm is not None:
            exif[piexif.ExifIFD.FocalLength] = _rational_tuple(payload.focal_length_mm)
        if payload.max_aperture is not None:
            exif[piexif.ExifIFD.FNumber] = _rational_tuple(payload.max_aperture)
            exif[piexif.ExifIFD.MaxApertureValue] = _rational_tuple(_apex_from_f_number(payload.max_aperture))

    if flags.film_iso and payload.iso is not None:
        exif[piexif.ExifIFD.ISOSpeedRatings] = payload.iso

    user_comment_parts: dict[str, str] = {}
    if payload.film_stock:
        user_comment_parts["film"] = payload.film_stock
    if payload.film_format:
        user_comment_parts["format"] = payload.film_format
    if payload.developer:
        user_comment_parts["developer"] = payload.developer
    if payload.push_pull and payload.push_pull != "Normal":
        user_comment_parts["push_pull"] = payload.push_pull

    if user_comment_parts:
        lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in user_comment_parts.items()]
        uc_bytes = b"ASCII\x00\x00\x00" + _exif_ascii("\n".join(lines))
        exif[piexif.ExifIFD.UserComment] = uc_bytes

    if flags.exposure and payload.capture_exposure:
        exif.update(_parse_exposure_str(payload.capture_exposure))

    if payload.capture_date is not None:
        exif[piexif.ExifIFD.DateTimeOriginal] = _exif_ascii(payload.capture_date.exif_text())
        if payload.capture_date.tz_offset:
            exif[piexif.ExifIFD.OffsetTimeOriginal] = _exif_ascii(payload.capture_date.tz_offset)

    gps: dict = {}
    if payload.gps_latitude is not None and payload.gps_longitude is not None:
        gps = exif_gps_rationals(payload.gps_latitude, payload.gps_longitude)

    return {"0th": zeroth, "Exif": exif, "GPS": gps, "Interop": {}, "1st": {}}


def _demote_scan_datetime(merged: dict) -> None:
    """The source timestamp records when the frame was digitized, not when it was shot."""
    exif = merged.setdefault("Exif", {})
    source = exif.get(piexif.ExifIFD.DateTimeOriginal)
    if source and piexif.ExifIFD.DateTimeDigitized not in exif:
        exif[piexif.ExifIFD.DateTimeDigitized] = source


def _sanitize_exif(exif_dict: dict) -> dict:
    """Drop entries piexif can't serialize."""
    _RATIONAL_TYPES = {5, 10}

    def _short_overflows(value) -> bool:
        vals = value if isinstance(value, (tuple, list)) else (value,)
        return any(isinstance(v, int) and not (0 <= v <= 65535) for v in vals)

    result = {}
    for ifd_name, ifd_data in exif_dict.items():
        if not isinstance(ifd_data, dict):
            result[ifd_name] = ifd_data
            continue
        tags_info = piexif.TAGS.get(ifd_name, {})
        clean = {}
        for tag, value in ifd_data.items():
            tag_type = tags_info.get(tag, {}).get("type")
            if isinstance(value, bytes) and tag_type in _RATIONAL_TYPES:
                continue
            if tag_type == 3 and _short_overflows(value):
                continue
            clean[tag] = value
        result[ifd_name] = clean
    return result


_JPEG_STRIP_0TH = frozenset(
    {
        254,
        256,
        257,
        258,
        259,
        262,
        273,
        277,
        278,
        279,
        284,
        330,
        513,
        514,
    }
)


def _prepare_jpeg_exif(exif_dict: dict) -> dict:
    prepared = _sanitize_exif(exif_dict)
    prepared.pop("thumbnail", None)
    prepared["1st"] = {}
    zeroth = prepared.get("0th")
    if isinstance(zeroth, dict):
        for tag in _JPEG_STRIP_0TH:
            zeroth.pop(tag, None)
    return prepared


_APP1_EXIF_LIMIT = 65533


def _resolve_payload(
    config: MetadataConfig,
    gear: Optional[GearLibrary],
    source_exif: Optional[dict],
) -> MetadataPayload:
    if gear is None:
        gear = GearProfiles.load_library()
    return build_metadata_payload(config, gear, source_exif)


def _read_xmp_from_source(source_path: str) -> Optional[bytes]:
    """Read embedded XMP from a source JPEG or TIFF/DNG file."""
    try:
        with open(source_path, "rb") as fh:
            head = fh.read(12)
        if head[:2] == b"\xff\xd8":
            with open(source_path, "rb") as fh:
                data = fh.read()
            return _extract_jpeg_xmp(data)
        if is_jxl(head):
            with open(source_path, "rb") as fh:
                return read_jxl_xmp(fh.read())
        with tifffile.TiffFile(source_path) as tf:
            tag = tf.pages[0].tags.get(_TIFF_XMP_TAG)
            if tag is None:
                return None
            value = tag.value
            if isinstance(value, bytes):
                return value
            if isinstance(value, str):
                return value.encode("utf-8")
    except Exception:
        pass
    return None


def _extract_jpeg_xmp(data: bytes) -> Optional[bytes]:
    i = 2
    n = len(data)
    while i < n:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xD9:
            break
        if marker in range(0xD0, 0xD8):
            i += 2
            continue
        if i + 4 > n:
            break
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        seg_end = i + 2 + seg_len
        if marker == 0xE1 and seg_end <= n:
            payload_start = i + 4
            if data[payload_start : payload_start + len(_XMP_APP1_HEADER)] == _XMP_APP1_HEADER:
                return data[payload_start + len(_XMP_APP1_HEADER) : seg_end]
        i = seg_end
    return None


def _load_source_exif(source_path: str, source_exif: Optional[dict]) -> Optional[dict]:
    if source_exif is not None:
        return copy.deepcopy(source_exif)
    from negpy.infrastructure.loaders.helpers import read_exif_from_file

    return read_exif_from_file(source_path)


def _dump_exif_preserve(exif_dict: dict) -> Optional[bytes]:
    """Serialize source EXIF for embed without NegPy field injection."""
    candidate = _prepare_jpeg_exif(exif_dict)

    def _fits(strip_maker_note: bool = False) -> Optional[bytes]:
        work = copy.deepcopy(candidate)
        if strip_maker_note and isinstance(work.get("Exif"), dict):
            work["Exif"].pop(piexif.ExifIFD.MakerNote, None)
        try:
            b = piexif.dump(work)
        except Exception:
            return None
        return b if len(b) <= _APP1_EXIF_LIMIT else None

    exif_bytes = _fits()
    if exif_bytes is not None:
        return exif_bytes
    exif_bytes = _fits(strip_maker_note=True)
    if exif_bytes is not None:
        _log.warning("source EXIF too large for JPEG APP1; dropped MakerNote for passthrough")
        return exif_bytes
    _log.warning("source EXIF too large for JPEG APP1; metadata passthrough skipped")
    return None


def preserve_source_metadata(
    image_bytes: bytes,
    source_path: str,
    source_exif: Optional[dict] = None,
) -> bytes:
    """
    Copy EXIF/XMP from the source file onto exported image bytes without
    adding or altering NegPy metadata fields.
    """
    exif_dict = _load_source_exif(source_path, source_exif)
    if not exif_dict:
        return image_bytes

    xmp_bytes = _read_xmp_from_source(source_path)

    try:
        output = io.BytesIO()
        if image_bytes[:2] == b"\xff\xd8":
            exif_bytes = _dump_exif_preserve(exif_dict)
            if exif_bytes is None:
                return image_bytes
            jpeg_buf = io.BytesIO()
            piexif.insert(exif_bytes, image_bytes, jpeg_buf)
            result = _inject_jpeg_xmp(jpeg_buf.getvalue(), xmp_bytes) if xmp_bytes else jpeg_buf.getvalue()
            output.write(result)
        elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            exif_bytes = piexif.dump(_sanitize_exif(exif_dict))
            _rewrite_png_with_metadata(image_bytes, exif_bytes, output, xmp_bytes)
        elif image_bytes[:4] == b"RIFF":
            exif_bytes = piexif.dump(_sanitize_exif(exif_dict))
            _rewrite_webp_with_metadata(image_bytes, exif_bytes, output, xmp_bytes)
        elif is_jxl(image_bytes):
            exif_bytes = piexif.dump(_sanitize_exif(exif_dict))
            _rewrite_jxl_with_metadata(image_bytes, exif_bytes, output, xmp_bytes)
        else:
            exif_bytes = piexif.dump(_sanitize_exif(exif_dict))
            _rewrite_tiff_preserve(image_bytes, exif_bytes, output, xmp_bytes, fold_user_comment=False)
        return output.getvalue()
    except Exception:
        _log.warning("metadata passthrough failed", exc_info=True)
        return image_bytes


def embed_jxl_boxes(
    image_bytes: bytes,
    exif_bytes: Optional[bytes] = None,
    xmp_bytes: Optional[bytes] = None,
) -> bytes:
    """Attach EXIF/XMP to JPEG XL bytes assembled outside the MetadataConfig path,
    such as a Linear Output dump. Returns the input unchanged if the container
    cannot be rebuilt."""
    if not exif_bytes and not xmp_bytes:
        return image_bytes
    try:
        output = io.BytesIO()
        _rewrite_jxl_with_metadata(image_bytes, exif_bytes or b"", output, xmp_bytes)
        return output.getvalue()
    except Exception:
        _log.warning("JPEG XL metadata embed failed", exc_info=True)
        return image_bytes


def _merged_exif_and_payload(
    config: MetadataConfig,
    source_exif: Optional[dict],
    gear: Optional[GearLibrary] = None,
) -> tuple[dict, Any]:
    """Source EXIF merged with the config's custom fields, plus the resolved payload."""
    payload = _resolve_payload(config, gear, source_exif)

    if source_exif is not None:
        merged = copy.deepcopy(source_exif)
    else:
        merged = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}

    if payload.exif_flags.strip_scan_residuals:
        strip_scan_exif_for_capture(merged)

    if payload.capture_date is not None:
        _demote_scan_datetime(merged)

    if payload.gps_latitude is not None and payload.gps_longitude is not None:
        strip_scan_gps(merged)

    custom = _build_custom_exif(payload)
    for ifd_name in ("0th", "Exif", "GPS", "Interop", "1st"):
        if ifd_name in custom and custom[ifd_name]:
            if ifd_name not in merged:
                merged[ifd_name] = {}
            merged[ifd_name].update(custom[ifd_name])

    merged.setdefault("0th", {})[piexif.ImageIFD.Orientation] = 1
    if isinstance(merged.get("1st"), dict):
        merged["1st"].pop(piexif.ImageIFD.Orientation, None)
    return merged, payload


def embed_metadata(
    image_bytes: bytes,
    config: MetadataConfig,
    source_exif: Optional[dict],
    gear: Optional[GearLibrary] = None,
) -> bytes:
    """
    Insert custom metadata + preserved source EXIF + XMP into exported image bytes.
    """
    merged, payload = _merged_exif_and_payload(config, source_exif, gear)
    xmp_bytes = build_xmp_bytes(payload) if payload.has_any_data() else None

    try:
        output = io.BytesIO()
        if image_bytes[:2] == b"\xff\xd8":
            exif_bytes = _dump_exif_within_app1_limit(merged, payload)
            jpeg_buf = io.BytesIO()
            piexif.insert(exif_bytes, image_bytes, jpeg_buf)
            jpeg_with_exif = jpeg_buf.getvalue()
            output = io.BytesIO()
            result = _inject_jpeg_xmp(jpeg_with_exif, xmp_bytes) if xmp_bytes else jpeg_with_exif
            output.write(result)
        elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            exif_bytes = piexif.dump(_sanitize_exif(merged))
            _rewrite_png_with_metadata(image_bytes, exif_bytes, output, xmp_bytes)
        elif image_bytes[:4] == b"RIFF":
            exif_bytes = piexif.dump(_sanitize_exif(merged))
            _rewrite_webp_with_metadata(image_bytes, exif_bytes, output, xmp_bytes)
        elif is_jxl(image_bytes):
            exif_bytes = piexif.dump(_sanitize_exif(merged))
            _rewrite_jxl_with_metadata(image_bytes, exif_bytes, output, xmp_bytes)
        else:
            exif_bytes = piexif.dump(_sanitize_exif(merged))
            _rewrite_tiff_with_metadata(image_bytes, exif_bytes, output, xmp_bytes)
        return output.getvalue()
    except Exception:
        _log.warning("metadata embed failed", exc_info=True)
        return image_bytes


def _strip_jpeg_xmp_segments(data: bytes) -> bytes:
    out = bytearray(data[:2])
    i = 2
    n = len(data)
    while i < n:
        if data[i] != 0xFF:
            out.extend(data[i:])
            break
        marker = data[i + 1]
        if marker == 0xD9:
            out.extend(data[i:])
            break
        if marker in range(0xD0, 0xD8):
            out.extend(data[i : i + 2])
            i += 2
            continue
        if i + 4 > n:
            out.extend(data[i:])
            break
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        seg_end = i + 2 + seg_len
        if marker == 0xE1 and seg_end <= n:
            payload_start = i + 4
            if data[payload_start : payload_start + len(_XMP_APP1_HEADER)] == _XMP_APP1_HEADER:
                i = seg_end
                continue
        out.extend(data[i:seg_end])
        i = seg_end
    return bytes(out)


def _inject_jpeg_xmp(jpeg_bytes: bytes, xmp_bytes: bytes) -> bytes:
    """Insert or replace an XMP APP1 segment in a JPEG."""
    if not xmp_bytes:
        return jpeg_bytes
    cleaned = _strip_jpeg_xmp_segments(jpeg_bytes)
    payload = _XMP_APP1_HEADER + xmp_bytes
    seg_len = len(payload) + 2
    if seg_len > 65535:
        _log.warning("XMP packet too large for JPEG APP1; skipping XMP embed")
        return jpeg_bytes
    xmp_segment = b"\xff\xe1" + struct.pack(">H", seg_len) + payload
    insert_at = 2
    i = 2
    n = len(cleaned)
    while i < n:
        if cleaned[i] != 0xFF:
            break
        marker = cleaned[i + 1]
        if marker in range(0xD0, 0xD8):
            i += 2
            continue
        if i + 4 > n:
            break
        seg_len = struct.unpack(">H", cleaned[i + 2 : i + 4])[0]
        seg_end = i + 2 + seg_len
        if marker in (0xE0, 0xE1, 0xED, 0xFE):
            insert_at = seg_end
            i = seg_end
            continue
        break
    return cleaned[:insert_at] + xmp_segment + cleaned[insert_at:]


def _dump_exif_within_app1_limit(merged: dict, payload: MetadataPayload) -> bytes:
    candidate = _prepare_jpeg_exif(merged)

    def _fits() -> Optional[bytes]:
        try:
            b = piexif.dump(candidate)
        except Exception:
            return None
        return b if len(b) <= _APP1_EXIF_LIMIT else None

    exif_bytes = _fits()
    if exif_bytes is not None:
        return exif_bytes

    if isinstance(candidate.get("Exif"), dict):
        candidate["Exif"].pop(piexif.ExifIFD.MakerNote, None)
    exif_bytes = _fits()
    if exif_bytes is not None:
        return exif_bytes

    _log.warning("source EXIF too large for JPEG APP1; keeping only NegPy metadata")
    candidate = _prepare_jpeg_exif(_build_custom_exif(payload))
    candidate.setdefault("0th", {})[piexif.ImageIFD.Orientation] = 1
    exif_bytes = _fits()
    if exif_bytes is not None:
        return exif_bytes

    candidate = {"0th": {piexif.ImageIFD.Orientation: 1}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}
    return piexif.dump(candidate)


_TIFF_TYPE_SCALAR = {3, 4, 8, 9}
_TIFF_TYPE_RATIONAL = {5, 10}
_TIFFFILE_RESERVED_TAGS: set[int] = set(tifffile.TIFF.TAG_FILTERED) | {270, 282, 283, 296, 305, 34675, _TIFF_XMP_TAG}


def _decode_ascii(value: object) -> str | None:
    if isinstance(value, bytes):
        value = value.rstrip(b"\x00").decode("ascii", "replace")
    if isinstance(value, str):
        return value.encode("ascii", "replace").decode("ascii")
    return None


def _exif_bytes_to_extratags(exif_bytes: bytes) -> tuple[str | None, list[tuple]]:
    exif_dict = piexif.load(exif_bytes)
    description = _decode_ascii(exif_dict.get("0th", {}).get(piexif.ImageIFD.ImageDescription))

    extratags: list[tuple] = []
    for ifd_name in ("0th", "Exif"):
        ifd_data = exif_dict.get(ifd_name) or {}
        type_table = piexif.TAGS.get(ifd_name, {})
        for tag, value in ifd_data.items():
            if tag in _TIFFFILE_RESERVED_TAGS:
                continue
            tag_info = type_table.get(tag)
            if not tag_info:
                continue
            entry = _build_extratag(tag, tag_info["type"], value)
            if entry is not None:
                extratags.append(entry)

    return description, extratags


def _build_extratag(tag: int, ttype: int, value: object) -> tuple | None:
    if ttype == 2:
        text = _decode_ascii(value)
        if text is None:
            return None
        return (tag, ttype, 0, text, True)

    if ttype in (1, 7):
        if not isinstance(value, (bytes, bytearray)):
            return None
        return (tag, ttype, len(value), bytes(value), True)

    if ttype in _TIFF_TYPE_SCALAR:
        if isinstance(value, int):
            return (tag, ttype, 1, value, True)
        if isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value):
            return (tag, ttype, len(value), value, True)
        return None

    if ttype in _TIFF_TYPE_RATIONAL:
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
            return (tag, ttype, 1, value, True)
        if isinstance(value, (list, tuple)) and all(isinstance(v, tuple) and len(v) == 2 for v in value):
            flat = [n for pair in value for n in pair]
            return (tag, ttype, len(value), flat, True)
        return None

    return None


def _tiff_metadata_from_exif_bytes(exif_bytes: bytes) -> tuple[dict | None, str | None]:
    """Map reserved TIFF tags (managed by tifffile, not extratags) from source EXIF."""
    exif_dict = piexif.load(exif_bytes)
    zeroth = exif_dict.get("0th", {}) or {}
    metadata: dict[str, str] = {}
    software: str | None = None
    for tag, key in (
        (piexif.ImageIFD.Artist, "Artist"),
        (piexif.ImageIFD.Copyright, "Copyright"),
    ):
        text = _decode_ascii(zeroth.get(tag))
        if text:
            metadata[key] = text
    software = _decode_ascii(zeroth.get(piexif.ImageIFD.Software))
    return (metadata or None), software


def tiff_metadata_kwargs(exif_bytes: bytes, xmp_bytes: Optional[bytes] = None, *, fold_user_comment: bool = True) -> dict:
    """tifffile.imwrite kwargs carrying the EXIF/XMP payload, for embedding at the
    first encode."""
    description, extratags = _exif_bytes_to_extratags(exif_bytes)
    if fold_user_comment:
        description = _fold_user_comment_into_description(description, extratags)
    if xmp_bytes:
        extratags.append((_TIFF_XMP_TAG, 7, len(xmp_bytes), xmp_bytes, True))
    metadata, software = _tiff_metadata_from_exif_bytes(exif_bytes)
    if fold_user_comment:
        # Artist and Copyright reach the file through extratags on the embed path.
        metadata = None
    return {"description": description or "", "metadata": metadata, "software": software, "extratags": extratags}


def export_embed_plan(
    config: Optional[MetadataConfig],
    source_exif: Optional[dict],
    source_path: str,
    gear: Optional[GearLibrary] = None,
) -> Optional[tuple[bytes, Optional[bytes], bool]]:
    """(exif_bytes, xmp_bytes, fold_user_comment) for embedding at first encode.
    None when there is nothing to embed or the payload cannot be built; the
    caller then keeps the post-hoc rewrite path."""
    if config is None:
        return None
    try:
        if config.protect_original_metadata:
            exif_dict = _load_source_exif(source_path, source_exif)
            if not exif_dict:
                return None
            return piexif.dump(_sanitize_exif(exif_dict)), _read_xmp_from_source(source_path), False
        merged, payload = _merged_exif_and_payload(config, source_exif, gear)
        xmp_bytes = build_xmp_bytes(payload) if payload.has_any_data() else None
        return piexif.dump(_sanitize_exif(merged)), xmp_bytes, True
    except Exception:
        _log.warning("metadata embed plan failed", exc_info=True)
        return None


def _rewrite_tiff_preserve(
    image_bytes: bytes,
    exif_bytes: bytes,
    output: io.BytesIO,
    xmp_bytes: Optional[bytes] = None,
    *,
    fold_user_comment: bool = True,
) -> None:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        _rewrite_png_with_metadata(image_bytes, exif_bytes, output, xmp_bytes)
        return

    with tifffile.TiffFile(io.BytesIO(image_bytes)) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        photometric = page.photometric.name.lower()
        compression = page.compression.name.lower() if int(page.compression) != 1 else None
        icc = page.iccprofile

    tifffile.imwrite(
        output,
        arr,
        photometric=photometric,
        compression=compression,
        iccprofile=icc,
        **tiff_metadata_kwargs(exif_bytes, xmp_bytes, fold_user_comment=fold_user_comment),
    )


def _rewrite_png_with_metadata(
    image_bytes: bytes,
    exif_bytes: bytes,
    output: io.BytesIO,
    xmp_bytes: Optional[bytes] = None,
) -> None:
    with Image.open(io.BytesIO(image_bytes)) as im:
        im.load()
        icc = im.info.get("icc_profile")
        pnginfo = PngImagePlugin.PngInfo()
        if xmp_bytes:
            pnginfo.add_itxt("XML:com.adobe.xmp", xmp_bytes.decode("utf-8"), zip=False)
        save_kwargs: dict = {"format": "PNG", "compress_level": 6, "exif": exif_bytes, "pnginfo": pnginfo}
        if icc:
            save_kwargs["icc_profile"] = icc
        im.save(output, **save_kwargs)


def _jxl_box(btype: bytes, payload: bytes) -> bytes:
    size = len(payload) + 8
    if size > 0xFFFFFFFF:
        return struct.pack(">I", 1) + btype + struct.pack(">Q", size + 8) + payload
    return struct.pack(">I", size) + btype + payload


def _tiff_header_exif(exif_bytes: bytes) -> bytes:
    """Strip the JPEG APP1 'Exif\x00\x00' prefix. Every other container stores the
    TIFF header on its own."""
    return exif_bytes[6:] if exif_bytes[:6] == b"Exif\x00\x00" else exif_bytes


def _is_jxl_metadata_box(btype: bytes, data: bytes, start: int) -> bool:
    """Exif/XMP boxes, including the brotli-compressed 'brob' form that names its
    inner type in the first four payload bytes."""
    if btype in _JXL_METADATA_BOXES:
        return True
    return btype == b"brob" and data[start : start + 4] in _JXL_METADATA_BOXES


def _rewrite_jxl_with_metadata(
    image_bytes: bytes,
    exif_bytes: bytes,
    output: io.BytesIO,
    xmp_bytes: Optional[bytes] = None,
) -> None:
    """Rebuild a JPEG XL container with fresh Exif/XMP boxes ahead of the codestream."""
    if image_bytes[:2] == JXL_CODESTREAM:
        container = JXL_SIGNATURE + _jxl_box(b"ftyp", _JXL_FTYP) + _jxl_box(b"jxlc", image_bytes)
    else:
        container = image_bytes

    metadata: list[bytes] = []
    if exif_bytes:
        # The Exif box payload is a 4-byte offset to the TIFF header, then the header itself.
        metadata.append(_jxl_box(JXL_EXIF_BOX, b"\x00\x00\x00\x00" + _tiff_header_exif(exif_bytes)))
    if xmp_bytes:
        metadata.append(_jxl_box(JXL_XMP_BOX, xmp_bytes))

    parts: list[bytes] = []
    inserted = False
    for btype, start, end in jxl_boxes(container):
        if _is_jxl_metadata_box(btype, container, start):
            continue
        if not inserted and btype in _JXL_CODESTREAM_BOXES:
            parts.extend(metadata)
            inserted = True
        parts.append(_jxl_box(btype, container[start:end]))
    if not inserted:
        raise ValueError("no JPEG XL codestream box")

    output.write(b"".join(parts))


def _webp_chunks(image_bytes: bytes) -> list[tuple[bytes, int, int]]:
    """Split a WebP file into (fourcc, payload start, payload end) chunks. Odd-sized
    payloads carry a pad byte that belongs to the chunk but not to its size."""
    if image_bytes[:4] != b"RIFF" or image_bytes[8:12] != b"WEBP":
        raise ValueError("not a WebP file")
    riff_end = min(len(image_bytes), 8 + int.from_bytes(image_bytes[4:8], "little"))
    chunks: list[tuple[bytes, int, int]] = []
    pos = 12
    while pos + 8 <= riff_end:
        size = int.from_bytes(image_bytes[pos + 4 : pos + 8], "little")
        start = pos + 8
        end = start + size
        if end > riff_end:
            raise ValueError("malformed WebP chunk size")
        chunks.append((image_bytes[pos : pos + 4], start, end))
        pos = end + (size & 1)
    if not chunks:
        raise ValueError("no WebP chunks")
    return chunks


def _webp_chunk(fourcc: bytes, payload: bytes) -> bytes:
    return fourcc + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) & 1 else b"")


def _webp_vp8x_header(image_bytes: bytes, chunks: list[tuple[bytes, int, int]]) -> bytearray:
    """The VP8X header a simple WebP has to gain before it can hold metadata:
    flags, then canvas width and height less one, little-endian in 3 bytes each."""
    for fourcc, start, end in chunks:
        if fourcc == b"VP8X":
            return bytearray(image_bytes[start:end])

    with Image.open(io.BytesIO(image_bytes)) as im:
        width, height = im.size
        has_alpha = im.mode in ("RGBA", "LA") or "transparency" in im.info
    header = bytearray(10)
    header[0] = _VP8X_ALPHA if has_alpha else 0
    header[4:7] = (width - 1).to_bytes(3, "little")
    header[7:10] = (height - 1).to_bytes(3, "little")
    return header


def _rewrite_webp_with_metadata(
    image_bytes: bytes,
    exif_bytes: bytes,
    output: io.BytesIO,
    xmp_bytes: Optional[bytes] = None,
) -> None:
    """Rebuild the RIFF container with fresh EXIF/XMP chunks. Only the extended
    (VP8X) form can hold them, and the spec fixes the chunk order."""
    chunks = _webp_chunks(image_bytes)
    header = _webp_vp8x_header(image_bytes, chunks)
    kept = [c for c in chunks if c[0] not in _WEBP_METADATA_CHUNKS and c[0] != b"VP8X"]

    exif_payload = _tiff_header_exif(exif_bytes) if exif_bytes else b""
    flags = header[0] & ~(_VP8X_EXIF | _VP8X_XMP)
    if any(fourcc == b"ICCP" for fourcc, _s, _e in kept):
        flags |= _VP8X_ICC
    if exif_payload:
        flags |= _VP8X_EXIF
    if xmp_bytes:
        flags |= _VP8X_XMP
    header[0] = flags

    ordered = [c for c in kept if c[0] == b"ICCP"] + [c for c in kept if c[0] != b"ICCP"]
    parts = [_webp_chunk(b"VP8X", bytes(header))]
    parts += [_webp_chunk(fourcc, image_bytes[start:end]) for fourcc, start, end in ordered]
    if exif_payload:
        parts.append(_webp_chunk(b"EXIF", exif_payload))
    if xmp_bytes:
        parts.append(_webp_chunk(b"XMP ", xmp_bytes))

    body = b"".join(parts)
    output.write(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body)


def _rewrite_tiff_with_metadata(
    image_bytes: bytes,
    exif_bytes: bytes,
    output: io.BytesIO,
    xmp_bytes: Optional[bytes] = None,
) -> None:
    _rewrite_tiff_preserve(image_bytes, exif_bytes, output, xmp_bytes, fold_user_comment=True)


def _fold_user_comment_into_description(description: str | None, extratags: list[tuple]) -> str | None:
    uc_text: str | None = None
    for entry in extratags:
        tag, _ttype, _count, value, _ = entry
        if tag != piexif.ExifIFD.UserComment or not isinstance(value, (bytes, bytearray)):
            continue
        raw = bytes(value)
        if raw[:8] == b"ASCII\x00\x00\x00":
            uc_text = _decode_ascii(raw[8:])
            if uc_text is not None:
                uc_text = uc_text.strip()
        break

    if not uc_text:
        return description
    if not description or description in uc_text:
        return uc_text
    if uc_text in description:
        return description
    return f"{description}\n{uc_text}"
