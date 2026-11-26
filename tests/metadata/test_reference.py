import io
import struct
from datetime import datetime, timedelta

import piexif
from PIL import Image

from negpy.features.metadata.capture import exif_gps_rationals
from negpy.features.metadata.reference import read_reference, tiff_is_sane


def _exif_tiff(
    *,
    taken: str | None = "2026:10:25 11:09:21",
    digitized: str | None = None,
    subsec: str = "198",
    offset: str | None = "+03:00",
    place: tuple[float, float] | None = (27.1414, 33.8834),
    gps_time: tuple[str, tuple[int, int, int]] | None = None,
) -> bytes:
    exif: dict = {"0th": {piexif.ImageIFD.Make: "Apple", piexif.ImageIFD.Model: "iPhone 16 Pro"}, "Exif": {}, "GPS": {}}
    if taken:
        exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = taken
        exif["Exif"][piexif.ExifIFD.SubSecTimeOriginal] = subsec
    if digitized:
        exif["Exif"][piexif.ExifIFD.DateTimeDigitized] = digitized
    if offset:
        exif["Exif"][piexif.ExifIFD.OffsetTimeOriginal] = offset
    if place:
        exif["GPS"].update(exif_gps_rationals(*place))
    if gps_time:
        day, (h, m, s) = gps_time
        exif["GPS"][piexif.GPSIFD.GPSDateStamp] = day
        exif["GPS"][piexif.GPSIFD.GPSTimeStamp] = ((h, 1), (m, 1), (s, 1))
    data = piexif.dump(exif)
    return data[6:] if data.startswith(b"Exif\x00\x00") else data


def _box(btype: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), btype) + payload


def _full_box(btype: bytes, version: int, payload: bytes) -> bytes:
    return _box(btype, bytes([version, 0, 0, 0]) + payload)


def _heif(tiff: bytes | None, *, in_idat: bool = False) -> bytes:
    """A minimal HEIF container: a stand-in image item and, optionally, an EXIF item for it."""
    image = b"\x00" * 32
    exif_item = struct.pack(">I", 6) + b"Exif\x00\x00" + tiff if tiff is not None else None
    ftyp = _box(b"ftyp", b"heic" + b"\x00" * 4 + b"mif1heic")
    hdlr = _full_box(b"hdlr", 0, b"\x00" * 4 + b"pict" + b"\x00" * 13)
    pitm = _full_box(b"pitm", 0, struct.pack(">H", 1))
    entries = [_full_box(b"infe", 2, struct.pack(">HH", 1, 0) + b"hvc1\x00")]
    iref = b""
    if exif_item is not None:
        entries.append(_full_box(b"infe", 2, struct.pack(">HH", 2, 0) + b"Exif\x00"))
        iref = _full_box(b"iref", 0, _box(b"cdsc", struct.pack(">HHH", 2, 1, 1)))
    iinf = _full_box(b"iinf", 0, struct.pack(">H", len(entries)) + b"".join(entries))

    def meta(image_at: int, exif_at: int) -> bytes:
        body = bytes([0x44, 0x00]) + struct.pack(">H", 2 if exif_item is not None else 1)
        body += struct.pack(">HHHH", 1, 0, 0, 1) + struct.pack(">II", image_at, len(image))
        idat = b""
        if exif_item is not None:
            body += struct.pack(">HHHH", 2, 1 if in_idat else 0, 0, 1) + struct.pack(">II", exif_at, len(exif_item))
            idat = _box(b"idat", exif_item) if in_idat else b""
        return _full_box(b"meta", 0, hdlr + pitm + iinf + iref + _full_box(b"iloc", 1, body) + idat)

    data_start = len(ftyp) + len(meta(0, 0)) + 8
    exif_at = 0 if in_idat else data_start + len(image)
    mdat = _box(b"mdat", image + (exif_item if exif_item is not None and not in_idat else b""))
    return ftyp + meta(data_start, exif_at) + mdat


def _jpeg(tiff: bytes) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, "JPEG", exif=b"Exif\x00\x00" + tiff)
    return buf.getvalue()


def _write(tmp_path, name: str, data: bytes):
    path = tmp_path / name
    path.write_bytes(data)
    return path


class TestReadReference:
    def test_heic_gives_time_offset_place_and_camera(self, tmp_path) -> None:
        photo = read_reference(_write(tmp_path, "a.heic", _heif(_exif_tiff())))
        assert photo.error == ""
        assert photo.local == datetime(2026, 10, 25, 11, 9, 21, 198000)
        assert (photo.offset, photo.offset_source) == (timedelta(hours=3), "exif")
        assert photo.instant.isoformat() == "2026-10-25T11:09:21.198000+03:00"
        assert round(photo.latitude, 4) == 27.1414 and round(photo.longitude, 4) == 33.8834
        assert (photo.make, photo.model) == ("Apple", "iPhone 16 Pro")

    def test_heic_with_the_exif_item_inside_the_meta_box(self, tmp_path) -> None:
        photo = read_reference(_write(tmp_path, "a.heic", _heif(_exif_tiff(), in_idat=True)))
        assert photo.error == "" and photo.offset == timedelta(hours=3)

    def test_jpeg_gives_the_same_fields(self, tmp_path) -> None:
        photo = read_reference(_write(tmp_path, "a.jpg", _jpeg(_exif_tiff())))
        assert photo.error == ""
        assert photo.instant.isoformat() == "2026-10-25T11:09:21.198000+03:00"

    def test_the_file_type_comes_from_the_content(self, tmp_path) -> None:
        photo = read_reference(_write(tmp_path, "named-like-a.jpg", _heif(_exif_tiff())))
        assert photo.error == ""

    def test_hemisphere_refs_give_the_signs(self, tmp_path) -> None:
        photo = read_reference(_write(tmp_path, "a.heic", _heif(_exif_tiff(place=(-33.8568, -9.1393)))))
        assert round(photo.latitude, 4) == -33.8568 and round(photo.longitude, 4) == -9.1393

    def test_the_gps_time_gives_the_offset_when_the_tag_is_missing(self, tmp_path) -> None:
        tiff = _exif_tiff(offset=None, gps_time=("2026:10:25", (8, 8, 34)))
        photo = read_reference(_write(tmp_path, "a.heic", _heif(tiff)))
        assert (photo.offset, photo.offset_source) == (timedelta(hours=3), "gps")

    def test_without_offset_or_gps_time_the_instant_is_unknown(self, tmp_path) -> None:
        photo = read_reference(_write(tmp_path, "a.heic", _heif(_exif_tiff(offset=None))))
        assert photo.error == "" and photo.offset is None and photo.instant is None

    def test_the_digitized_time_is_the_fallback(self, tmp_path) -> None:
        tiff = _exif_tiff(taken=None, digitized="2026:10:25 11:10:01")
        photo = read_reference(_write(tmp_path, "a.jpg", _jpeg(tiff)))
        assert photo.local == datetime(2026, 10, 25, 11, 10, 1)


class TestUnreadableFiles:
    def test_no_exif_item_means_no_capture_time(self, tmp_path) -> None:
        photo = read_reference(_write(tmp_path, "a.heic", _heif(None)))
        assert photo.error == "no capture time" and photo.local is None

    def test_other_files_give_an_error(self, tmp_path) -> None:
        movie = _box(b"ftyp", b"qt  " + b"\x00" * 4 + b"qt  ") + _box(b"mdat", b"\x00" * 16)
        assert "HEIF brand" in read_reference(_write(tmp_path, "a.mov", movie)).error
        assert "not a JPEG or HEIC" in read_reference(_write(tmp_path, "a.txt", b"plain text")).error

    def test_truncated_files_give_an_error_and_never_raise(self, tmp_path) -> None:
        data = _heif(_exif_tiff())
        for cut in (6, 20, 60, len(data) // 2, len(data) - 40):
            photo = read_reference(_write(tmp_path, f"cut{cut}.heic", data[:cut]))
            assert photo.local is None and photo.error

    def test_a_damaged_entry_count_never_reaches_piexif(self, tmp_path) -> None:
        tiff = bytearray(_exif_tiff())
        e = ">" if tiff[:2] == b"MM" else "<"
        first_ifd = struct.unpack_from(e + "L", tiff, 4)[0]
        struct.pack_into(e + "L", tiff, first_ifd + 2 + 4, 0x7FFFFFFF)
        assert not tiff_is_sane(bytes(tiff))
        photo = read_reference(_write(tmp_path, "a.heic", _heif(bytes(tiff))))
        assert "damaged EXIF block" in photo.error
