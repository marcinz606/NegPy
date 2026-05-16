"""Pure functions to embed custom metadata into exported image bytes via piexif."""

import io
import logging
import re
from fractions import Fraction
from typing import Optional

import piexif
import tifffile

from negpy.features.metadata.models import MetadataConfig

_log = logging.getLogger(__name__)


# Push/pull label mapping
PUSH_PULL_LABELS = {
    -3: "Pull -3",
    -2: "Pull -2",
    -1: "Pull -1",
    0: "Normal",
    1: "Push +1",
    2: "Push +2",
    3: "Push +3",
}


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


def _build_custom_exif(config: MetadataConfig) -> dict:
    """Build a piexif-format EXIF dict containing only the custom metadata fields."""

    zeroth: dict = {}
    exif: dict = {}

    if config.film:
        zeroth[piexif.ImageIFD.ImageDescription] = config.film

    if config.scanning:
        zeroth[piexif.ImageIFD.Software] = config.scanning

    # Pack film/format/developer/push_pull into UserComment
    user_comment_parts = {}
    if config.film:
        user_comment_parts["film"] = config.film
    fmt_value = config.format_other if config.format == "Other" else config.format
    if fmt_value:
        user_comment_parts["format"] = fmt_value
    if config.developer:
        user_comment_parts["developer"] = config.developer
    if config.push_pull != 0:
        user_comment_parts["push_pull"] = PUSH_PULL_LABELS.get(config.push_pull, str(config.push_pull))

    if user_comment_parts:
        # EXIF UserComment: 8-byte character code prefix + encoded content.
        # ASCII prefix is universally supported; UNICODE/UTF-16-LE causes garbled
        # output in most EXIF readers (ExifTool, macOS Preview, Lightroom).
        lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in user_comment_parts.items()]
        uc_bytes = b"ASCII\x00\x00\x00" + "\n".join(lines).encode("ascii")
        exif[piexif.ExifIFD.UserComment] = uc_bytes

    # ── EXIF field overrides ─────────────────────────────────────────────
    if config.camera_override:
        zeroth[piexif.ImageIFD.Model] = config.camera_override

    if config.lens_override:
        exif[piexif.ExifIFD.LensModel] = config.lens_override

    if config.exposure_override:
        parsed = _parse_exposure_str(config.exposure_override)
        exif.update(parsed)

    return {"0th": zeroth, "Exif": exif, "GPS": {}, "Interop": {}, "1st": {}}


def _sanitize_exif(exif_dict: dict) -> dict:
    """Drop RATIONAL/SRATIONAL entries stored as raw bytes (piexif cannot serialize them).
    ASCII tags (type 2) legitimately use bytes and are left untouched."""
    _RATIONAL_TYPES = {5, 10}  # RATIONAL, SRATIONAL
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
            clean[tag] = value
        result[ifd_name] = clean
    return result


def embed_metadata(
    image_bytes: bytes,
    config: MetadataConfig,
    source_exif: Optional[dict],
) -> bytes:
    """
    Insert custom metadata + preserved source EXIF into exported image bytes.

    Args:
        image_bytes: JPEG or TIFF image bytes from the rendering pipeline.
        config: MetadataConfig with user-entered custom fields.
        source_exif: piexif-format EXIF dict from the source file (or None).

    Returns:
        Image bytes with embedded metadata.
    """
    # Start with source EXIF if available, otherwise empty shell
    if source_exif is not None:
        merged = source_exif
    else:
        merged = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}

    # Overlay custom metadata
    custom = _build_custom_exif(config)
    for ifd_name in ("0th", "Exif", "GPS", "Interop", "1st"):
        if ifd_name in custom and custom[ifd_name]:
            if ifd_name not in merged:
                merged[ifd_name] = {}
            merged[ifd_name].update(custom[ifd_name])

    try:
        exif_bytes = piexif.dump(_sanitize_exif(merged))
        output = io.BytesIO()
        if image_bytes[:2] == b"\xff\xd8":
            piexif.insert(exif_bytes, image_bytes, output)
        else:
            _rewrite_tiff_with_metadata(image_bytes, exif_bytes, output)
        return output.getvalue()
    except Exception:
        _log.warning("metadata embed failed", exc_info=True)
        return image_bytes


# TIFF type codes we know how to map onto tifffile extratags.
_TIFF_TYPE_SCALAR = {3, 4, 8, 9}  # SHORT, LONG, SSHORT, SLONG
_TIFF_TYPE_RATIONAL = {5, 10}  # RATIONAL, SRATIONAL

# Tags tifffile owns. TAG_FILTERED covers the core image IFD (and the EXIF/GPS
# sub-IFD pointers, conveniently); the rest correspond to tifffile.imwrite
# kwargs we already pass (description, resolution, software, iccprofile).
_TIFFFILE_RESERVED_TAGS: set[int] = set(tifffile.TIFF.TAG_FILTERED) | {270, 282, 283, 296, 305, 34675}


def _decode_ascii(value: object) -> str | None:
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("ascii", "replace")
    if isinstance(value, str):
        return value
    return None


def _exif_bytes_to_extratags(exif_bytes: bytes) -> tuple[str | None, list[tuple]]:
    """Flatten a piexif EXIF block into ``(description, extratags)`` for tifffile.

    EXIF/GPS sub-IFD entries are hoisted into the main IFD because tifffile
    has no API for writing sub-IFDs, and PIL's own ``exif=`` path is broken
    for TIFF (see ``_rewrite_tiff_with_metadata``). The description is split
    out so it can be passed via ``description=`` instead of ``extratags``.
    """
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
    """Coerce a piexif value into a tifffile extratag tuple, or None if untranslatable."""
    if ttype == 2:  # ASCII
        text = _decode_ascii(value)
        if text is None:
            return None
        return (tag, ttype, 0, text, True)  # count=0: tifffile sizes ASCII itself

    if ttype in (1, 7):  # BYTE, UNDEFINED
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
            # tifffile internally doubles count for RATIONAL and unpacks `*value`,
            # so multi-element values must be a flat sequence of ints.
            flat = [n for pair in value for n in pair]
            return (tag, ttype, len(value), flat, True)
        return None

    return None


def _rewrite_tiff_with_metadata(image_bytes: bytes, exif_bytes: bytes, output: io.BytesIO) -> None:
    """Re-encode a TIFF with EXIF metadata via tifffile.

    PIL's ``img.save(format="TIFF", exif=...)`` path is doubly unusable here:
    it writes the EXIF sub-IFD pointer as a dict-coerced LONG8 which libtiff
    rejects with ``_TIFFVSetField: Bad LONG8 ... EXIFIFDOffset``, and it has
    no 16-bit RGB mode so it would silently downconvert the image to 8-bit.
    Round-tripping through tifffile preserves the pixel data; EXIF tags are
    folded into the main IFD via ``extratags``.
    """
    with tifffile.TiffFile(io.BytesIO(image_bytes)) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        photometric = page.photometric.name.lower()
        compression = page.compression.name.lower() if int(page.compression) != 1 else None
        icc = page.iccprofile

    description, extratags = _exif_bytes_to_extratags(exif_bytes)
    description = _fold_user_comment_into_description(description, extratags)

    tifffile.imwrite(
        output,
        arr,
        photometric=photometric,
        compression=compression,
        iccprofile=icc,
        description=description or "",
        metadata=None,
        extratags=extratags,
    )


def _fold_user_comment_into_description(description: str | None, extratags: list[tuple]) -> str | None:
    """Mirror UserComment into ImageDescription for TIFF output.

    UserComment lives in the EXIF sub-IFD on JPEG, but tifffile can only emit
    it as a main-IFD tag — and most TIFF readers (macOS Preview, Lightroom)
    don't expose UNDEFINED main-IFD tags. Folding the text into description
    keeps every custom field (film, format, developer, push/pull) visible.
    """
    uc_text: str | None = None
    for entry in extratags:
        tag, _ttype, _count, value, _ = entry
        if tag != piexif.ExifIFD.UserComment or not isinstance(value, (bytes, bytearray)):
            continue
        raw = bytes(value)
        # EXIF spec: 8-byte character-code prefix + payload. We only decode
        # ASCII; UNICODE (UTF-16-LE) and JIS would garble under ASCII decode.
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
