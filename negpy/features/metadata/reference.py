"""Reference photos: the capture time, offset and place of a digital photo (HEIC or JPEG)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

import piexif

from negpy.features.metadata.exif_read import gps_decimal, rational_to_float, safe_str

HEIF_BRANDS = frozenset({b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1", b"mif2", b"avif"})
MAX_META_BOX = 8 << 20
MAX_EXIF_ITEM = 4 << 20
TIFF_MAGIC = (b"MM\x00*", b"II*\x00")

_TIFF_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_SUB_IFD_TAGS = (0x8769, 0x8825, 0xA005)  # Exif, GPS, Interoperability
_MAX_OFFSET = timedelta(hours=14)


class ReferenceReadError(Exception):
    """The file is not a readable HEIC or JPEG."""


@dataclass(frozen=True)
class ReferencePhoto:
    """What a reference photo tells about its moment and place. `error` is empty on success."""

    path: str
    local: Optional[datetime] = None
    offset: Optional[timedelta] = None
    offset_source: str = ""  # "exif", "gps" or ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    make: str = ""
    model: str = ""
    error: str = ""

    @property
    def instant(self) -> Optional[datetime]:
        """The absolute moment, when the local time and its offset are both known."""
        if self.local is None or self.offset is None:
            return None
        return self.local.replace(tzinfo=timezone(self.offset))


def read_reference(path: str | Path) -> ReferencePhoto:
    """Read a reference photo. A damaged or unknown file gives `error`, never an exception."""
    try:
        kind = _sniff(path)
        if kind == "jpeg":
            tiff = _jpeg_exif_tiff(path)
        elif kind == "heif":
            tiff = _heif_exif_tiff(path)
        else:
            raise ReferenceReadError("not a JPEG or HEIC file")
        if tiff and not tiff_is_sane(tiff):
            raise ReferenceReadError("damaged EXIF block")
        photo = _photo_from_exif(str(path), piexif.load(tiff) if tiff else {})
    except Exception as exc:  # noqa: BLE001 - piexif raises assorted types on damaged EXIF; one file must not stop a batch
        return ReferencePhoto(path=str(path), error=f"{type(exc).__name__}: {exc}")
    if photo.local is None:
        return ReferencePhoto(path=str(path), error="no capture time")
    return photo


def tiff_is_sane(tiff: bytes) -> bool:
    """Every IFD entry fits inside the block. piexif builds a struct format as long as an entry's
    count, so a damaged count must never reach it."""
    if len(tiff) < 8 or tiff[:4] not in TIFF_MAGIC:
        return False
    e = ">" if tiff[:2] == b"MM" else "<"
    n = len(tiff)
    first = struct.unpack_from(e + "L", tiff, 4)[0]
    todo, seen = [first], set()
    while todo:
        off = todo.pop()
        if off == 0 or off in seen:
            continue
        seen.add(off)
        if off + 2 > n:
            return False
        count = struct.unpack_from(e + "H", tiff, off)[0]
        if off + 2 + 12 * count > n:
            return False
        for i in range(count):
            p = off + 2 + 12 * i
            tag, typ, cnt = struct.unpack_from(e + "HHL", tiff, p)
            if cnt * _TIFF_TYPE_SIZE.get(typ, 1) > n:
                return False
            if tag in _SUB_IFD_TAGS:
                todo.append(struct.unpack_from(e + "L", tiff, p + 8)[0])
        nxt = off + 2 + 12 * count
        if off == first and nxt + 4 <= n:
            todo.append(struct.unpack_from(e + "L", tiff, nxt)[0])
    return True


def _sniff(path: str | Path) -> str:
    """'jpeg', 'heif' or 'other', from the first bytes of the file, never from its extension."""
    with open(path, "rb") as f:
        head = f.read(12)
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[4:8] == b"ftyp":
        return "heif"
    return "other"


def _jpeg_exif_tiff(path: str | Path) -> Optional[bytes]:
    """The TIFF block of the JPEG's Exif APP1 segment. Reading stops at the image data."""
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ReferenceReadError("not a JPEG file")
        while True:
            marker = f.read(2)
            while marker[:1] == b"\xff" and marker[1:2] == b"\xff":
                marker = marker[1:] + f.read(1)
            if len(marker) < 2 or marker[0] != 0xFF or marker[1] in (0xD9, 0xDA):
                return None
            raw = f.read(2)
            if len(raw) < 2:
                return None
            length = struct.unpack(">H", raw)[0]
            if length < 2:
                return None
            if marker[1] == 0xE1:
                segment = f.read(length - 2)
                if segment[:6] == b"Exif\x00\x00":
                    return segment[6:]
            else:
                f.seek(length - 2, 1)


def _box_header(f: BinaryIO) -> Optional[tuple[bytes, int]]:
    """Type and payload size of the box at the file position. None at the end of the file."""
    head = f.read(8)
    if len(head) < 8:
        return None
    size, btype = struct.unpack(">I4s", head)
    header = 8
    if size == 1:
        size = struct.unpack(">Q", f.read(8))[0]
        header = 16
    elif size == 0:
        here = f.tell()
        size = f.seek(0, 2) - here + header
        f.seek(here)
    if size < header:
        raise ReferenceReadError(f"bad box size for {btype!r}")
    return btype, size - header


def _children(buf: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """(type, payload start, box end) of each box in buf[start:end]."""
    pos = start
    while pos + 8 <= end:
        size, btype = struct.unpack_from(">I4s", buf, pos)
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", buf, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise ReferenceReadError(f"bad child box {btype!r}")
        yield btype, pos + header, pos + size
        pos += size


def _uint(buf: bytes, pos: int, n: int) -> tuple[int, int]:
    """Big-endian unsigned integer of n bytes. Returns (value, next position)."""
    if n == 0:
        return 0, pos
    if pos + n > len(buf):
        raise ReferenceReadError("truncated box")
    return int.from_bytes(buf[pos : pos + n], "big"), pos + n


def _parse_iinf(buf: bytes, start: int, end: int) -> dict[int, bytes]:
    """item_ID to item_type, from the ItemInfoBox."""
    version = buf[start]
    pos = start + 4 + (2 if version == 0 else 4)
    items: dict[int, bytes] = {}
    for btype, s, _e in _children(buf, pos, end):
        if btype != b"infe" or buf[s] < 2:
            continue
        item_id, p = _uint(buf, s + 4, 2 if buf[s] == 2 else 4)
        items[item_id] = buf[p + 2 : p + 6]
    return items


def _parse_iloc(buf: bytes, start: int, end: int) -> dict[int, tuple[int, int, list[tuple[int, int]]]]:
    """item_ID to (construction method, base offset, [(extent offset, extent length)])."""
    version = buf[start]
    p = start + 4
    offset_size, length_size = buf[p] >> 4, buf[p] & 0x0F
    base_offset_size = buf[p + 1] >> 4
    index_size = buf[p + 1] & 0x0F if version in (1, 2) else 0
    p += 2
    count, p = _uint(buf, p, 2 if version < 2 else 4)
    locs: dict[int, tuple[int, int, list[tuple[int, int]]]] = {}
    for _ in range(count):
        item_id, p = _uint(buf, p, 2 if version < 2 else 4)
        construction = 0
        if version in (1, 2):
            raw, p = _uint(buf, p, 2)
            construction = raw & 0x0F
        p += 2
        base, p = _uint(buf, p, base_offset_size)
        extent_count, p = _uint(buf, p, 2)
        extents = []
        for _ in range(extent_count):
            _index, p = _uint(buf, p, index_size)
            off, p = _uint(buf, p, offset_size)
            length, p = _uint(buf, p, length_size)
            extents.append((off, length))
        if p > end:
            raise ReferenceReadError("truncated iloc box")
        locs[item_id] = (construction, base, extents)
    return locs


def _parse_cdsc(buf: bytes, start: int, end: int) -> dict[int, list[int]]:
    """from_item_ID to [to_item_IDs] of the 'cdsc' (content describes) references."""
    id_size = 2 if buf[start] == 0 else 4
    refs: dict[int, list[int]] = {}
    for btype, s, _e in _children(buf, start + 4, end):
        if btype != b"cdsc":
            continue
        from_id, p = _uint(buf, s, id_size)
        n, p = _uint(buf, p, 2)
        to_ids = []
        for _ in range(n):
            to_id, p = _uint(buf, p, id_size)
            to_ids.append(to_id)
        refs[from_id] = to_ids
    return refs


def _heif_exif_tiff(path: str | Path) -> Optional[bytes]:
    """The TIFF block of the EXIF item that describes the primary image. None without one."""
    with open(path, "rb") as f:
        first = _box_header(f)
        if first is None or first[0] != b"ftyp":
            raise ReferenceReadError("not an ISOBMFF file")
        ftyp = f.read(first[1])
        brands = {ftyp[0:4]} | {ftyp[i : i + 4] for i in range(8, len(ftyp) - 3, 4)}
        if not brands & HEIF_BRANDS:
            raise ReferenceReadError(f"not a HEIF brand: {ftyp[0:4]!r}")

        meta = None
        while (hdr := _box_header(f)) is not None:
            btype, size = hdr
            if btype == b"meta":
                if size > MAX_META_BOX:
                    raise ReferenceReadError("meta box too large")
                meta = f.read(size)
                break
            f.seek(size, 1)
        if meta is None:
            return None

        items: dict[int, bytes] = {}
        locs: dict[int, tuple[int, int, list[tuple[int, int]]]] = {}
        cdsc: dict[int, list[int]] = {}
        primary = None
        idat_start = None
        for btype, s, e in _children(meta, 4, len(meta)):
            if btype == b"iinf":
                items = _parse_iinf(meta, s, e)
            elif btype == b"iloc":
                locs = _parse_iloc(meta, s, e)
            elif btype == b"iref":
                cdsc = _parse_cdsc(meta, s, e)
            elif btype == b"pitm":
                primary, _ = _uint(meta, s + 4, 2 if meta[s] == 0 else 4)
            elif btype == b"idat":
                idat_start = s

        exif_ids = sorted((i for i, t in items.items() if t == b"Exif"), key=lambda i: primary not in cdsc.get(i, []))
        if not exif_ids:
            return None
        if exif_ids[0] not in locs:
            raise ReferenceReadError("EXIF item has no location")
        construction, base, extents = locs[exif_ids[0]]

        data = bytearray()
        for off, length in extents:
            if construction == 0:
                f.seek(base + off)
                chunk = f.read(length if length else MAX_EXIF_ITEM)
            elif construction == 1 and idat_start is not None:
                s = idat_start + base + off
                chunk = meta[s : s + length]
            else:
                raise ReferenceReadError(f"unsupported item construction method {construction}")
            data += chunk
            if len(data) > MAX_EXIF_ITEM:
                raise ReferenceReadError("EXIF item too large")

    # ISO/IEC 23008-12 Annex A: a 4-byte offset to the TIFF header comes before the EXIF block.
    if len(data) < 12:
        return None
    skip = struct.unpack_from(">I", data, 0)[0]
    tiff = bytes(data[4 + skip :])
    if tiff[:4] not in TIFF_MAGIC:
        hits = [i for i in (data.find(m, 4, 64) for m in TIFF_MAGIC) if i >= 0]
        if not hits:
            raise ReferenceReadError("EXIF item has no TIFF header")
        tiff = bytes(data[min(hits) :])
    return tiff


def _parse_offset(text: str) -> Optional[timedelta]:
    """'+03:00' to timedelta(hours=3). None for empty or malformed text."""
    if len(text) != 6 or text[0] not in "+-" or text[3] != ":":
        return None
    try:
        hours, minutes = int(text[1:3]), int(text[4:6])
    except ValueError:
        return None
    if hours > 14 or minutes > 59:
        return None
    return (-1 if text[0] == "-" else 1) * timedelta(hours=hours, minutes=minutes)


def _parse_exif_datetime(text: str, subsec: str) -> Optional[datetime]:
    try:
        dt = datetime.strptime(text[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    digits = "".join(c for c in subsec if c.isdigit())[:6]
    return dt.replace(microsecond=int(digits.ljust(6, "0"))) if digits else dt


def _gps_utc(gps: dict) -> Optional[datetime]:
    date = safe_str(gps.get(piexif.GPSIFD.GPSDateStamp)).strip()
    stamp = gps.get(piexif.GPSIFD.GPSTimeStamp)
    if not date or not isinstance(stamp, tuple) or len(stamp) != 3:
        return None
    parts = [rational_to_float(p) for p in stamp]
    if any(p is None for p in parts):
        return None
    try:
        day = datetime.strptime(date, "%Y:%m:%d")
    except ValueError:
        return None
    return day + timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])


def _photo_from_exif(path: str, exif: dict) -> ReferencePhoto:
    zeroth = exif.get("0th") or {}
    ifd = exif.get("Exif") or {}
    gps = exif.get("GPS") or {}

    taken = safe_str(ifd.get(piexif.ExifIFD.DateTimeOriginal)).strip()
    subsec = safe_str(ifd.get(piexif.ExifIFD.SubSecTimeOriginal)).strip()
    offset_text = safe_str(ifd.get(piexif.ExifIFD.OffsetTimeOriginal)).strip()
    if not taken:
        taken = safe_str(ifd.get(piexif.ExifIFD.DateTimeDigitized)).strip()
        subsec = safe_str(ifd.get(piexif.ExifIFD.SubSecTimeDigitized)).strip()
        offset_text = safe_str(ifd.get(piexif.ExifIFD.OffsetTimeDigitized)).strip()
    local = _parse_exif_datetime(taken, subsec) if taken else None

    offset = _parse_offset(offset_text)
    offset_source = "exif" if offset is not None else ""
    utc = _gps_utc(gps)
    if offset is None and local is not None and utc is not None:
        # The GPS time is the time of the fix, not of the shutter, so round to whole quarter hours.
        quarters = round((local - utc).total_seconds() / 900)
        if abs(quarters) * timedelta(minutes=15) <= _MAX_OFFSET:
            offset = timedelta(minutes=15 * quarters)
            offset_source = "gps"

    return ReferencePhoto(
        path=path,
        local=local,
        offset=offset,
        offset_source=offset_source,
        latitude=gps_decimal(gps.get(piexif.GPSIFD.GPSLatitude), gps.get(piexif.GPSIFD.GPSLatitudeRef)),
        longitude=gps_decimal(gps.get(piexif.GPSIFD.GPSLongitude), gps.get(piexif.GPSIFD.GPSLongitudeRef)),
        make=safe_str(zeroth.get(piexif.ImageIFD.Make)).strip(),
        model=safe_str(zeroth.get(piexif.ImageIFD.Model)).strip(),
    )
