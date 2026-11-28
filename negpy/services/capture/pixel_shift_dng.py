"""Write a merged Pixel Shift set as a LinearRaw DNG that carries the camera's color data,
so the frame decodes through the RAW path like the camera's own file."""

import os
import struct
import tempfile
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from typing import Optional

import piexif
import rawpy
import tifffile

from negpy.infrastructure.capture.raw_demosaic import _linearity_limit
from negpy.infrastructure.loaders.helpers import camera_wb_multipliers, camera_xyz_matrix, read_exif_from_file
from negpy.kernel.system.i18n import tr
from negpy.kernel.system.paths import apply_umask_mode, replace_file, safe_unlink
from negpy.services.export.linear_output import _read_source_meta_tiff


_STRIP_ROWS = 16
#: The Software tag of a merged file, followed by its shot count. It tells a merged file from a body's own DNG.
_SOFTWARE = "NegPy Pixel Shift "
_VERIFY_ROWS = 512


class DngVerifyError(RuntimeError):
    """The written file did not read back as written."""


@dataclass(frozen=True)
class RawColor:
    """What a reader needs to treat the pixels as this camera's RAW."""

    black: tuple[float, float, float]
    white: int
    color_matrix: tuple[float, ...]  # XYZ (D65) to camera, 9 values, row-major
    as_shot_neutral: tuple[float, float, float]
    make: str
    model: str
    orientation: int = 1
    datetime: str = ""  # "YYYY:MM:DD HH:MM:SS", as the camera wrote it
    exposure_time: tuple[int, int] = (0, 1)
    f_number: tuple[int, int] = (0, 1)
    iso: int = 0
    focal_length: tuple[int, int] = (0, 1)
    lens_model: str = ""


def raw_color(source_raw: str) -> RawColor:
    """`RawColor` of a camera RAW file."""
    with rawpy.imread(source_raw) as raw:
        red, green, blue, green2 = (float(v) for v in raw.black_level_per_channel)
        white = _linearity_limit(raw)  # the ceiling NegPy decodes the camera's own file against
        matrix = camera_xyz_matrix(raw)
        gains = camera_wb_multipliers(raw)
    if matrix is None or gains is None:
        matrix, gains = _dng_color(source_raw)  # a body that shoots DNG: the decoder reports no matrix for it
    if matrix is None or gains is None:
        raise ValueError(tr("{value} has no camera color data", value=os.path.basename(source_raw)))
    meta = _read_source_meta_tiff(source_raw)
    exif = read_exif_from_file(source_raw) or {}
    zeroth, shot = exif.get("0th", {}), exif.get("Exif", {})
    stamp = shot.get(piexif.ExifIFD.DateTimeOriginal) or zeroth.get(piexif.ImageIFD.DateTime) or b""
    iso = shot.get(piexif.ExifIFD.ISOSpeedRatings, 0)
    return RawColor(
        black=(red, (green + green2) / 2, blue),
        white=white,
        color_matrix=tuple(v for row in matrix for v in row),
        as_shot_neutral=(gains[1] / gains[0], 1.0, gains[1] / gains[2]),
        make=meta.make or "",
        model=meta.model or "",
        orientation=int(zeroth.get(piexif.ImageIFD.Orientation, 1)),
        datetime=_text(stamp),
        exposure_time=tuple(shot.get(piexif.ExifIFD.ExposureTime, (0, 1))),
        f_number=tuple(shot.get(piexif.ExifIFD.FNumber, (0, 1))),
        iso=int(iso[0] if isinstance(iso, tuple) else iso),
        focal_length=tuple(shot.get(piexif.ExifIFD.FocalLength, (0, 1))),
        lens_model=_text(shot.get(piexif.ExifIFD.LensModel, b"")),
    )


def _dng_color(path: str) -> tuple[Optional[list], Optional[list]]:
    """ColorMatrix1 as three rows and the white balance gains of a DNG's own tags, or (None, None)."""
    try:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            matrix, neutral = tags[50721].value, tags[50728].value
        ratios = lambda flat: [n / d for n, d in zip(flat[0::2], flat[1::2])]  # noqa: E731
        matrix, neutral = ratios(matrix), ratios(neutral)
        return [matrix[0:3], matrix[3:6], matrix[6:9]], [1.0 / v for v in neutral]
    except (KeyError, IndexError, ValueError, ZeroDivisionError, OSError, tifffile.TiffFileError):
        return None, None


#: tifffile refuses to write ExifTag (34665) itself. This unused code sorts in its place, and the
# entry is rewritten as ExifTag after the Exif IFD is appended.
_EXIF_SLOT = 34664


def _text(value) -> str:
    return value.decode("ascii", "ignore").strip("\x00 ") if isinstance(value, bytes) else str(value)


def _exif_ifd(color: RawColor, offset: int, byteorder: str) -> bytes:
    """A classic-TIFF Exif IFD for a file position of `offset`: what the capture was shot with.
    An unknown value (a manual lens reports no aperture) has no entry."""
    rational, short, ascii_ = 5, 3, 2
    fields = []  # (tag, type, count, packed value)
    for tag, pair in ((33434, color.exposure_time), (33437, color.f_number), (37386, color.focal_length)):
        if pair[0] and pair[1]:
            fields.append((tag, rational, 1, struct.pack(byteorder + "2I", *pair)))
    if color.iso:
        fields.append((34855, short, 1, struct.pack(byteorder + "H", min(color.iso, 65535))))
    for tag, text in ((36867, color.datetime), (42036, color.lens_model)):
        if text:
            data = text.encode("ascii", "ignore") + b"\x00"
            fields.append((tag, ascii_, len(data), data))
    fields.sort()
    head = struct.pack(byteorder + "H", len(fields))
    data_at = offset + len(head) + 12 * len(fields) + 4
    entries, extra = b"", b""
    for tag, kind, count, value in fields:
        if len(value) <= 4:
            slot = value.ljust(4, b"\x00")
        else:
            slot = struct.pack(byteorder + "I", data_at + len(extra))
            extra += value + b"\x00" * (len(value) % 2)  # values start on a word boundary
        entries += struct.pack(byteorder + "HHI", tag, kind, count) + slot
    return head + entries + b"\x00" * 4 + extra


def _rationals(values: tuple[float, ...]) -> tuple[int, ...]:
    """Numerator and denominator pairs, as a TIFF RATIONAL tag takes them."""
    fractions = [Fraction(v).limit_denominator(1_000_000) for v in values]
    return tuple(n for f in fractions for n in (f.numerator, f.denominator))


def _tags(color: RawColor) -> list[tuple]:
    camera = f"{color.make} {color.model}".strip()
    # The black level goes into LinearizationTable: libraw does not take a 3-sample BlackLevel off
    # the white level, so it decodes such a file darker than the camera's own.
    # shortcut: libraw is exact only for equal channel blacks, add a per-channel path if a camera differs.
    pedestal = int(min(color.black))
    table = np.clip(np.arange(65536) - pedestal, 0, None).astype(np.uint16)
    return [
        (271, "s", 0, color.make, True),
        (272, "s", 0, color.model, True),
        (274, "H", 1, color.orientation, True),
        # The exposure tags sit in the IFD itself, as TIFF/EP allows. NegPy takes a LinearRaw DNG with
        # no exposure tag for a film scanner's file and drops its white balance and matrix
        # (`is_scanned_dng`), so FNumber is always written: 0 is "aperture unknown".
        (33434, "2I", 1, color.exposure_time, True),
        (33437, "2I", 1, color.f_number, True),
        (34855, "H", 1, min(color.iso, 65535), True),
        (_EXIF_SLOT, "I", 1, 0, True),  # becomes ExifTag once the file's length is known
        (50706, "B", 4, (1, 4, 0, 0), True),  # DNGVersion
        (50708, "s", 0, camera, True),  # UniqueCameraModel
        (50712, "H", table.size, tuple(table.tolist()), True),  # LinearizationTable
        (50713, "H", 2, (1, 1), True),  # BlackLevelRepeatDim
        (50714, "2I", 3, _rationals(tuple(b - pedestal for b in color.black)), True),  # BlackLevel
        (50717, "I", 3, (color.white - pedestal,) * 3, True),  # WhiteLevel
        (50721, "2i", 9, _rationals(color.color_matrix), True),  # ColorMatrix1
        (50728, "2I", 3, _rationals(color.as_shot_neutral), True),  # AsShotNeutral
        (50778, "H", 1, 21, True),  # CalibrationIlluminant1: D65
    ]


def merged_shots(path: str) -> int:
    """The shot count of the Pixel Shift set that `path` was merged from; 0 for any other file."""
    if not path.lower().endswith(".dng"):
        return 0
    try:
        with tifffile.TiffFile(path) as tif:
            software = str(tif.pages[0].tags[305].value)
        return int(software.removeprefix(_SOFTWARE)) if software.startswith(_SOFTWARE) else 0
    except (KeyError, IndexError, ValueError, OSError, tifffile.TiffFileError):
        return 0


def write_linear_dng(rgb: np.ndarray, color: RawColor, out_path: str, shots: int) -> None:
    """Write uint16 HxWx3 `rgb`, merged from `shots` frames, as an uncompressed 3-sample LinearRaw DNG,
    under a temporary name, read it back, compare, then rename to `out_path`. Raises `DngVerifyError`
    and leaves no file."""
    if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("a LinearRaw DNG takes a uint16 HxWx3 image")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(out_path) or ".", delete=False, suffix=".part") as tmp:
            tmp_path = tmp.name
        tifffile.imwrite(
            tmp_path,
            rgb,
            photometric="rgb",
            planarconfig="contig",
            metadata=None,
            extratags=_tags(color),
            rowsperstrip=_STRIP_ROWS,  # NegPy reads a thumbnail only from a DNG in small strips
            datetime=color.datetime or None,
            software=f"{_SOFTWARE}{shots}",
        )
        # Written as RGB then retagged: for LinearRaw, tifffile declares two of the samples as extra.
        with tifffile.TiffFile(tmp_path, mode="r+") as tif:
            tif.pages[0].tags["PhotometricInterpretation"].overwrite(34892)
            # The Exif IFD goes at the end of the file; a BigTIFF has another IFD layout and gets none.
            if not tif.is_bigtiff:
                handle, slot = tif.filehandle, tif.pages[0].tags[_EXIF_SLOT]
                offset = handle.seek(0, os.SEEK_END)
                offset += offset % 2
                handle.seek(offset)
                handle.write(_exif_ifd(color, offset, tif.byteorder))
                handle.seek(slot.offset)
                handle.write(struct.pack(tif.byteorder + "HHII", 34665, 4, 1, offset))
        written = tifffile.memmap(tmp_path, mode="r")
        try:
            same = written.dtype == np.uint16 and written.shape == rgb.shape
            # Compared in blocks: a whole-image compare holds a second copy of a 16-shot frame.
            for y in range(0, rgb.shape[0] if same else 0, _VERIFY_ROWS):
                if not np.array_equal(written[y : y + _VERIFY_ROWS], rgb[y : y + _VERIFY_ROWS]):
                    same = False
                    break
        finally:
            del written
        if not same:
            raise DngVerifyError(tr("{value} did not read back as written", value=os.path.basename(out_path)))
        apply_umask_mode(tmp_path)
        replace_file(tmp_path, out_path)
        tmp_path = None
    finally:
        safe_unlink(tmp_path)
