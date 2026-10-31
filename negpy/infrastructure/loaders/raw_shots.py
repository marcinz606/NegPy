"""Camera RAW files that hold several shots in one file, and the choice of what to decode.

A Pentax Pixel Shift RAW holds four shots one photosite apart, an in-camera HDR RAW holds three
exposures. LibRaw decodes one shot per open, the first unless told otherwise; rawpy does not
report how many shots a file holds, so the TIFF structure of the file tells. Any TIFF-based RAW
with several same-size CFA images offers its shots. Only a Pentax Pixel Shift combines: the shot
order and step of another make need their own mapping. The per-edit choice `raw_shot` is 0 for
Auto (combine a Pixel Shift, else the first shot) or N for shot N.
"""

import functools
import os
import struct
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional

import tifffile

from negpy.infrastructure.loaders.helpers import open_shot
from negpy.infrastructure.loaders.pixel_shift import SHOT_SHIFT, PixelShiftStats, PixelShiftWrapper, combined_image

_TIFF_MAGIC = (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+")
_CFA = 32803
# Pentax makernote tag 0x0008 (Quality): 7 is "RAW with Pixel Shift" (LibRaw metadata/pentax.cpp).
_QUALITY_TAG = 0x0008
_PIXEL_SHIFT_QUALITY = 7


class ShotKind(StrEnum):
    PIXEL_SHIFT = "pixel_shift"  # the shots combine into one image
    SHOTS = "shots"  # one shot at a time, such as the exposures of an in-camera HDR


@dataclass(frozen=True)
class ShotLayout:
    count: int
    kind: ShotKind


@dataclass(frozen=True)
class ShotsInfo:
    """What a multi-shot RAW decoded to, for the panel that offers the choice."""

    count: int
    kind: ShotKind
    shot: int  # the decoded shot, 1 is the first; 0 when the shots are combined
    stats: Optional[PixelShiftStats] = None
    failed: bool = False  # the chosen shots could not be decoded, so the first shot shows


def composite_shots(parts: Sequence[Optional[ShotsInfo]]) -> Optional[ShotsInfo]:
    """The shots of a composite's files read as one, since the parts share one raw_shot: a part
    that failed, else the part whose shots disagree most, else the first part."""
    known = [p for p in parts if p is not None]
    failed = [p for p in known if p.failed]
    if failed:
        return failed[0]
    disagree = [(s.disagreement, p) for p in known if (s := p.stats) is not None and s.shots_disagree]
    if disagree:
        return max(disagree, key=lambda d: d[0])[1]
    return known[0] if known else None


def probe_shots(file_path: str) -> Optional[ShotLayout]:
    """The shots of a TIFF-based camera RAW (PEF, DNG, NEF, ARW, CR2, 3FR ...), read from its
    structure without a decode. None for a file with one shot, a format that is not TIFF, or a
    file that cannot be read."""
    try:
        st = os.stat(file_path)
        with open(file_path, "rb") as f:
            if f.read(4) not in _TIFF_MAGIC:
                return None
    except OSError:
        return None
    return _probe(os.path.abspath(file_path), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=256)
def _probe(file_path: str, _size: int, _mtime_ns: int) -> Optional[ShotLayout]:
    try:
        with tifffile.TiffFile(file_path) as tif:
            shapes = [(int(page.imagelength), int(page.imagewidth)) for page in _all_pages(tif.pages) if _is_cfa(page)]
            if len(shapes) < 2 or len(set(shapes)) != 1:
                return None
            page0 = tif.pages[0]
            make = str(_tag_value(page0, "Make") or "").lower()
            quality = _pentax_quality(page0, tif.byteorder)
    except Exception:
        return None
    pixel_shift = len(shapes) == len(SHOT_SHIFT) and quality == _PIXEL_SHIFT_QUALITY and ("pentax" in make or "ricoh" in make)
    return ShotLayout(len(shapes), ShotKind.PIXEL_SHIFT if pixel_shift else ShotKind.SHOTS)


def _all_pages(pages: Any) -> list:
    found = []
    for page in pages:
        found.append(page)
        sub = getattr(page, "pages", None)
        if sub:
            found.extend(_all_pages(sub))
    return found


def _is_cfa(page: Any) -> bool:
    try:
        return int(page.photometric) == _CFA
    except (AttributeError, TypeError, ValueError):
        return False


def _tag_value(page: Any, name: str) -> Any:
    tag = page.tags.get(name)
    return None if tag is None else tag.value


def _pentax_quality(page0: Any, byteorder: str) -> Optional[int]:
    """Makernote Quality of the first IFD: EXIF MakerNote in a PEF, DNGPrivateData in a DNG."""
    exif = _tag_value(page0, "ExifTag")
    blobs = [exif.get("MakerNote")] if isinstance(exif, dict) else []
    blobs.append(_tag_value(page0, "DNGPrivateData"))
    for blob in blobs:
        if isinstance(blob, (bytes, bytearray)):
            quality = makernote_quality(bytes(blob), byteorder)
            if quality is not None:
                return quality
    return None


def makernote_quality(blob: bytes, byteorder: str = "<") -> Optional[int]:
    """Tag 0x0008 of a Pentax makernote, read with the header rules of LibRaw parse_makernote:
    "PENTAX \\0" and the byte order put the IFD at byte 10, "AOC\\0" at byte 6. A DNG converter
    wraps the makernote in "Adobe\\0MakN", a size, the byte order and the original offset."""
    if blob.startswith(b"Adobe\0MakN"):
        blob = blob[20:]
    if blob.startswith(b"PENTAX \0"):
        order, ifd = blob[8:10], 10
    elif blob.startswith(b"AOC\0"):
        order, ifd = blob[4:6], 6
    else:
        return None
    end = ">" if order == b"MM" else "<" if order == b"II" else byteorder
    try:
        (entries,) = struct.unpack_from(end + "H", blob, ifd)
        for i in range(entries):
            tag, kind, count = struct.unpack_from(end + "HHI", blob, ifd + 2 + 12 * i)
            if tag == _QUALITY_TAG and kind == 3 and count == 1:
                return int(struct.unpack_from(end + "H", blob, ifd + 2 + 12 * i + 8)[0])
    except struct.error:
        return None
    return None


class ShotSource(ABC):
    """One way to decode a multi-shot RAW: a single shot, or the shots combined."""

    def __init__(self, file_path: str, layout: ShotLayout) -> None:
        self.file_path = file_path
        self.layout = layout

    @abstractmethod
    def open(self, should_cancel: Optional[Callable[[], bool]] = None, export: bool = False) -> tuple[Any, ShotsInfo]:
        """The decoder NegPy reads like a rawpy.RawPy, and what it holds. `export` asks for the
        linear export's white scale where the decoder has one of its own."""


class SingleShot(ShotSource):
    """One shot, decoded by LibRaw like any camera RAW."""

    def __init__(self, file_path: str, layout: ShotLayout, index: int) -> None:
        super().__init__(file_path, layout)
        self.index = index

    def open(self, should_cancel: Optional[Callable[[], bool]] = None, export: bool = False) -> tuple[Any, ShotsInfo]:
        raw = open_shot(self.file_path, self.index)
        try:
            raw.unpack()  # now, so that a shot LibRaw does not hold fails here and not mid-render
        except Exception:
            raw.close()
            raise
        return raw, ShotsInfo(self.layout.count, self.layout.kind, self.index + 1)


class CombinedShots(ShotSource):
    """The four shots of a Pixel Shift RAW combined (see pixel_shift)."""

    def open(self, should_cancel: Optional[Callable[[], bool]] = None, export: bool = False) -> tuple[Any, ShotsInfo]:
        image = combined_image(self.file_path, should_cancel)
        wrapper = PixelShiftWrapper(image, image.export_white if export else image.render_white)
        return wrapper, ShotsInfo(self.layout.count, self.layout.kind, 0, image.stats)


def shot_source(file_path: str, raw_shot: int) -> Optional[ShotSource]:
    """How to decode `file_path` for the per-edit choice `raw_shot`, or None for a file with one
    shot, which LibRaw decodes like any camera RAW. A shot number beyond the file's shots reads as Auto."""
    layout = probe_shots(file_path)
    if layout is None:
        return None
    if 1 <= raw_shot <= layout.count:
        return SingleShot(file_path, layout, raw_shot - 1)
    if layout.kind is ShotKind.PIXEL_SHIFT:
        return CombinedShots(file_path, layout)
    return SingleShot(file_path, layout, 0)


def thumbnail_shot(file_path: str, raw_shot: int) -> int:
    """The shot choice for a thumbnail decode. A thumbnail never combines: a Pixel Shift on Auto
    shows its first shot. Every other choice passes through unchanged, so the thumbnail and the
    canvas of an ordinary file share one cached decode."""
    if raw_shot == 0 and isinstance(shot_source(file_path, 0), CombinedShots):
        return 1
    return raw_shot
