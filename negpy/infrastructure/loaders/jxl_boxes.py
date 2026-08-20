"""ISOBMFF box access for JPEG XL files: the container that holds Exif and XMP.

Kept apart from jxl_loader so both the loaders and the metadata writer can use it
without an import cycle.
"""

from typing import Optional

JXL_SIGNATURE = b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a"
JXL_CODESTREAM = b"\xff\x0a"
JXL_EXIF_BOX = b"Exif"
JXL_XMP_BOX = b"xml "


def is_jxl(data: bytes) -> bool:
    return data[:12] == JXL_SIGNATURE or data[:2] == JXL_CODESTREAM


def jxl_boxes(data: bytes) -> list[tuple[bytes, int, int]]:
    """Split a JPEG XL container into (type, start, end) boxes. Raises unless the
    boxes tile the file exactly, so a malformed input cannot be silently truncated."""
    boxes: list[tuple[bytes, int, int]] = []
    pos = 0
    n = len(data)
    while pos < n:
        if pos + 8 > n:
            raise ValueError("truncated JPEG XL box header")
        size = int.from_bytes(data[pos : pos + 4], "big")
        btype = data[pos + 4 : pos + 8]
        header = 8
        if size == 1:
            if pos + 16 > n:
                raise ValueError("truncated JPEG XL box header")
            size = int.from_bytes(data[pos + 8 : pos + 16], "big")
            header = 16
        elif size == 0:
            size = n - pos
        if size < header or pos + size > n:
            raise ValueError("malformed JPEG XL box size")
        boxes.append((btype, pos + header, pos + size))
        pos += size
    return boxes


def _box_payload(data: bytes, wanted: bytes) -> Optional[bytes]:
    """Payload of the first box of this type, decompressing the brotli 'brob' form
    that names its inner type in the first four payload bytes."""
    try:
        boxes = jxl_boxes(data)
    except ValueError:
        return None
    for btype, start, end in boxes:
        if btype == wanted:
            return data[start:end]
        if btype == b"brob" and data[start : start + 4] == wanted:
            import imagecodecs

            try:
                return bytes(imagecodecs.brotli_decode(data[start + 4 : end]))
            except Exception:
                return None
    return None


def read_jxl_exif(data: bytes) -> Optional[bytes]:
    """EXIF payload of a JPEG XL file, from the TIFF header on. The box prefixes it
    with a 4-byte offset to that header."""
    payload = _box_payload(data, JXL_EXIF_BOX)
    if payload is None or len(payload) < 8:
        return None
    offset = int.from_bytes(payload[:4], "big")
    tiff = payload[4 + offset :]
    return tiff if tiff[:2] in (b"MM", b"II") else None


def read_jxl_xmp(data: bytes) -> Optional[bytes]:
    return _box_payload(data, JXL_XMP_BOX)
