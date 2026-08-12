"""Explain a raw NegPy cannot decode, when the reason is knowable.

Nikon's High Efficiency raw (Z 8 / Z 9) is intoPIX TicoRAW, but its TIFF Compression tag
reads 34713 — the same value a lossless NEF uses. So the tags say "ordinary NEF", libraw
parses the file and reports full sensor dimensions, and the failure only arrives when the
payload is unpacked, as "Unsupported file format or not RAW file". That reads as "your file
is corrupt" to someone whose other NEFs all work.
"""

import struct

import pytest

from negpy.infrastructure.loaders.helpers import unsupported_raw_reason


def _nef(tmp_path, name, strip_payload: bytes, compression: int = 34713):
    """A minimal little-endian TIFF with one SubIFD carrying `strip_payload`.

    Only the parts the detector reads: IFD0 with a SubIFDs tag, and a SubIFD with
    Compression and StripOffsets.
    """
    path = tmp_path / name
    strip_offset = 512
    sub_offset = 200
    ifd0_offset = 8

    def ifd(entries, next_ifd=0):
        out = struct.pack("<H", len(entries))
        for tag, typ, count, value in entries:
            out += struct.pack("<HHI", tag, typ, count) + struct.pack("<I", value)
        return out + struct.pack("<I", next_ifd)

    buf = bytearray(b"\x00" * (strip_offset + len(strip_payload)))
    buf[0:8] = struct.pack("<2sHI", b"II", 42, ifd0_offset)
    buf[ifd0_offset : ifd0_offset + 100] = ifd([(330, 4, 1, sub_offset)]).ljust(100, b"\x00")  # SubIFDs
    sub = ifd(
        [
            (256, 3, 1, 8280),  # ImageWidth
            (257, 3, 1, 5520),  # ImageLength
            (258, 3, 1, 14),  # BitsPerSample
            (259, 3, 1, compression),  # Compression
            (273, 4, 1, strip_offset),  # StripOffsets
            (279, 4, 1, len(strip_payload)),  # StripByteCounts
        ]
    )
    buf[sub_offset : sub_offset + len(sub)] = sub
    buf[strip_offset : strip_offset + len(strip_payload)] = strip_payload
    path.write_bytes(bytes(buf))
    return str(path)


TICORAW = b"\xff\x10\xff\x50\x00\x22CONTACT_INTOPIX_\xef\xc0" + b"\x00" * 64
LOSSLESS = b"\x00\x4a\x00\x01\x00\x00" + bytes(range(64))


def test_high_efficiency_is_named(tmp_path):
    reason = unsupported_raw_reason(_nef(tmp_path, "he.nef", TICORAW))
    assert reason is not None
    assert "High Efficiency" in reason
    assert "Lossless" in reason and "DNG" in reason, "say what to do about it, not just what is wrong"


def test_an_ordinary_nef_has_no_complaint(tmp_path):
    """Same Compression tag, different payload — only the bytes can tell them apart."""
    assert unsupported_raw_reason(_nef(tmp_path, "lossless.nef", LOSSLESS)) is None


@pytest.mark.parametrize("name", ["shot.cr2", "shot.arw", "shot.dng", "scan.tif"])
def test_only_nef_is_inspected(tmp_path, name):
    """The marker is Nikon-specific; other formats fail for their own reasons and keep
    libraw's own message."""
    assert unsupported_raw_reason(_nef(tmp_path, name, TICORAW)) is None


def test_a_corrupt_file_yields_no_false_explanation(tmp_path):
    """Nothing recognised means libraw's own error is the honest answer."""
    p = tmp_path / "broken.nef"
    p.write_bytes(b"not a tiff at all")
    assert unsupported_raw_reason(str(p)) is None


def test_a_missing_file_does_not_raise(tmp_path):
    assert unsupported_raw_reason(str(tmp_path / "absent.nef")) is None
