"""Orientation of camera RAWs whose EXIF piexif cannot parse comes from LibRaw."""

from unittest.mock import MagicMock, patch

import pytest

from negpy.infrastructure.loaders import helpers


@pytest.mark.parametrize("flip,expected", [(0, 1), (3, 3), (5, 8), (6, 6)])
def test_a_cr3_without_readable_exif_takes_libraws_flip(tmp_path, flip, expected):
    path = tmp_path / "a.cr3"
    path.write_bytes(b"\x00\x00\x00\x18ftypcrx ")
    raw = MagicMock()
    raw.sizes.flip = flip
    opened = MagicMock()
    opened.__enter__.return_value = raw
    with patch("rawpy.imread", return_value=opened), patch.object(helpers, "read_exif_from_file", return_value=None):
        assert helpers.read_orientation(str(path)) == expected


def test_a_jpeg_without_exif_stays_upright(tmp_path):
    from PIL import Image

    path = tmp_path / "a.jpg"
    Image.new("RGB", (4, 4)).save(path)
    assert helpers.read_orientation(str(path)) == 1
