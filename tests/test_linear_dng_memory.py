import numpy as np
import pytest
import tifffile

from negpy.infrastructure.loaders.rawpy_loader import _peek_linear_dng_rgb


@pytest.mark.parametrize("use_table", [False, True])
@pytest.mark.parametrize("crop", [None, (3, 5, 251, 1493), (0, 0, 999, 2000), (-1, 0, 10, 10)])
def test_block_decode_matches_full_array(tmp_path, use_table, crop):
    rng = np.random.default_rng(42)
    codes = rng.integers(0, 4096, (1500, 257, 3), dtype=np.uint16)
    table = np.linspace(0, 65535, 1024).astype(np.uint16)
    tags = [
        (50714, 5, 3, (17, 2, 256, 1, 3, 2), False),
        (50717, 4, 3, (65535, 50000, 60000), False),
        (50728, 5, 3, (1, 2, 1, 1, 1, 3), False),
    ]
    if use_table:
        tags.append((50712, 3, len(table), tuple(table), False))
    if crop is not None:
        tags.extend([(50719, 10, 2, (crop[0], 1, crop[1], 1), False), (50720, 4, 2, crop[2:], False)])
    path = tmp_path / "linear.dng"
    tifffile.imwrite(path, codes, photometric=34892, planarconfig="contig", extratags=tags)
    expected = codes.astype(np.float64)
    if use_table:
        expected = table[np.clip(expected, 0, len(table) - 1).astype(np.int64)].astype(np.float64)
    black = np.array([8.5, 256.0, 1.5])
    expected = np.clip((expected - black) / (np.array([65535.0, 50000.0, 60000.0]) - black), 0, 1)
    if crop is not None:
        x, y, w, h = crop
        if 0 <= y < 1500 and 0 <= x < 257 and w > 0 and h > 0 and (w, h) != (257, 1500):
            expected = expected[y : y + h, x : x + w]
    decoded, gains = _peek_linear_dng_rgb(str(path))
    np.testing.assert_array_equal(decoded, expected.astype(np.float32))
    assert decoded.flags.c_contiguous
    assert gains == (2.0, 1.0, 3.0)
    np.testing.assert_array_equal(tifffile.imread(path), codes)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_block_decode_default_levels(tmp_path, dtype):
    source = np.arange(60).reshape(4, 5, 3).astype(dtype)
    path = tmp_path / "defaults.dng"
    tifffile.imwrite(path, source, photometric=34892, planarconfig="contig")
    scale = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    decoded, gains = _peek_linear_dng_rgb(str(path))
    np.testing.assert_array_equal(decoded, np.clip(source.astype(np.float64) / scale, 0, 1).astype(np.float32))
    assert gains is None
