"""Pure functions to embed custom metadata into exported image bytes via piexif."""

import io
import json
import logging
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
        # EXIF UserComment: 8-byte character code prefix + ASCII content.
        # ASCII prefix is universally supported; UNICODE/UTF-16-LE causes garbled
        # output in most EXIF readers (ExifTool, macOS Preview, Lightroom).
        json_str = json.dumps(user_comment_parts, ensure_ascii=True)
        uc_bytes = b"ASCII\x00\x00\x00" + json_str.encode("ascii")
        exif[piexif.ExifIFD.UserComment] = uc_bytes

    return {"0th": zeroth, "Exif": exif, "GPS": {}, "Interop": {}, "1st": {}}


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
        exif_bytes = piexif.dump(merged)
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
