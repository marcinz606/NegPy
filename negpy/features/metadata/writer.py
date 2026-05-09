"""Pure functions to embed custom metadata into exported image bytes via piexif."""

import io
import logging
import re
from fractions import Fraction
from typing import Optional

import piexif
from PIL import Image

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
            img = Image.open(io.BytesIO(image_bytes))
            img.save(output, format="TIFF", exif=exif_bytes)
        return output.getvalue()
    except Exception:
        _log.warning("metadata embed failed", exc_info=True)
        return image_bytes
