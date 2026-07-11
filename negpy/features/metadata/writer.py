"""Pure functions to embed custom metadata into exported image bytes via piexif + XMP."""

import copy
import io
import logging
import re
import struct
from fractions import Fraction
from typing import Optional

import piexif
import tifffile
from PIL import Image, PngImagePlugin

from negpy.features.metadata.exif_read import strip_scan_exif_for_capture
from negpy.features.metadata.gear_models import GearLibrary
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.payload import NEGPY_SOFTWARE, MetadataPayload, build_metadata_payload
from negpy.features.metadata.xmp import build_xmp_bytes
from negpy.services.assets.gear import GearProfiles

_log = logging.getLogger(__name__)

_XMP_APP1_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
_TIFF_XMP_TAG = 700  # XMLPacket


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


def _exif_ascii(text: str) -> bytes:
    return text.encode("ascii", errors="replace")


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
        uc_bytes = b"ASCII\x00\x00\x00" + "\n".join(lines).encode("ascii")
        exif[piexif.ExifIFD.UserComment] = uc_bytes

    if flags.exposure and payload.capture_exposure:
        exif.update(_parse_exposure_str(payload.capture_exposure))

    return {"0th": zeroth, "Exif": exif, "GPS": {}, "Interop": {}, "1st": {}}


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
        else:
            exif_bytes = piexif.dump(_sanitize_exif(exif_dict))
            _rewrite_tiff_preserve(image_bytes, exif_bytes, output, xmp_bytes, fold_user_comment=False)
        return output.getvalue()
    except Exception:
        _log.warning("metadata passthrough failed", exc_info=True)
        return image_bytes


def embed_metadata(
    image_bytes: bytes,
    config: MetadataConfig,
    source_exif: Optional[dict],
    gear: Optional[GearLibrary] = None,
) -> bytes:
    """
    Insert custom metadata + preserved source EXIF + XMP into exported image bytes.
    """
    payload = _resolve_payload(config, gear, source_exif)

    if source_exif is not None:
        merged = copy.deepcopy(source_exif)
    else:
        merged = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}

    if payload.exif_flags.strip_scan_residuals:
        strip_scan_exif_for_capture(merged)

    custom = _build_custom_exif(payload)
    for ifd_name in ("0th", "Exif", "GPS", "Interop", "1st"):
        if ifd_name in custom and custom[ifd_name]:
            if ifd_name not in merged:
                merged[ifd_name] = {}
            merged[ifd_name].update(custom[ifd_name])

    merged.setdefault("0th", {})[piexif.ImageIFD.Orientation] = 1
    if isinstance(merged.get("1st"), dict):
        merged["1st"].pop(piexif.ImageIFD.Orientation, None)

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
    for ifd_name in ("0th", "Exif", "GPS"):
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

    description, extratags = _exif_bytes_to_extratags(exif_bytes)
    if fold_user_comment:
        description = _fold_user_comment_into_description(description, extratags)

    if xmp_bytes:
        extratags.append((_TIFF_XMP_TAG, 7, len(xmp_bytes), xmp_bytes, True))

    metadata = None
    software = None
    if not fold_user_comment:
        metadata, software = _tiff_metadata_from_exif_bytes(exif_bytes)

    tifffile.imwrite(
        output,
        arr,
        photometric=photometric,
        compression=compression,
        iccprofile=icc,
        description=description or "",
        metadata=metadata,
        software=software,
        extratags=extratags,
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
            uc_text = raw[8:].decode("ascii", "replace").rstrip("\x00").strip()
        break

    if not uc_text:
        return description
    if not description or description in uc_text:
        return uc_text
    if uc_text in description:
        return description
    return f"{description}\n{uc_text}"
