import os
import tempfile
import unittest

import numpy as np
import rawpy
import tifffile

from negpy.infrastructure.loaders.factory import LoaderFactory
from negpy.infrastructure.loaders.tiff_loader import NonStandardFileWrapper
from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm, is_xtrans


class _FakeRaw:
    def __init__(self, raw_type: rawpy.RawType, block_size: int) -> None:
        self.raw_type = raw_type
        self.raw_pattern = np.zeros((block_size, block_size), dtype=np.uint8)


class TestRawHandlers(unittest.TestCase):
    def test_pakon_detection(self):
        pass

    def test_non_standard_wrapper(self):
        data = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        wrapper = NonStandardFileWrapper(data)

        with wrapper as raw:
            processed = raw.postprocess(gamma=(1, 1), output_bps=16)
            self.assertEqual(processed.dtype, np.uint16)
            self.assertAlmostEqual(np.mean(processed), 32767, delta=100)

    def test_xtrans_full_res_uses_dht_not_vng(self):
        # VNG produces dot/maze artifacts on X-Trans's 6x6 CFA in high-contrast
        # regions (see issue #272). DHT is the LGPL-clean algorithm built for X-Trans.
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=6)
        self.assertEqual(get_best_demosaic_algorithm(raw, for_preview=False), rawpy.DemosaicAlgorithm.DHT)

    def test_xtrans_preview_uses_linear(self):
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=6)
        self.assertEqual(get_best_demosaic_algorithm(raw, for_preview=True), rawpy.DemosaicAlgorithm.LINEAR)

    def test_bayer_full_res_uses_ahd(self):
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=2)
        self.assertEqual(get_best_demosaic_algorithm(raw, for_preview=False), rawpy.DemosaicAlgorithm.AHD)

    def test_is_xtrans(self):
        self.assertTrue(is_xtrans(_FakeRaw(rawpy.RawType.Flat, block_size=6)))
        self.assertFalse(is_xtrans(_FakeRaw(rawpy.RawType.Flat, block_size=2)))
        self.assertFalse(is_xtrans(object()))  # missing raw_pattern


# --- 3-channel LinearRaw DNG libraw can't unpack (DxO PhotoLab/PureRAW, Lightroom
# Enhance: DNG 1.7 JPEG-XL compression) --------------------------------------------


def _write_linear_dng_libraw_cant_read(path: str, h: int, w: int, raw_codes: np.ndarray, table: np.ndarray) -> None:
    """A DNG 1.7 linear RGB SubIFD compressed with JPEG-XL (compression=52546), the same
    shape as DxO PhotoLab's/Lightroom Enhance's compressed output: libraw's DNG-SDK-gated
    decoder isn't linked into rawpy's wheels (see rawpy#207), so `raw.unpack()` fails on
    this regardless of the hand-built DNG being otherwise minimal. `raw_codes` are the
    *stored* per-pixel indices (pre-LinearizationTable), one LinearizationTable shared by
    all channels, BlackLevel (0, 256, 0), WhiteLevel 65535, and a synthetic AsShotNeutral."""
    thumb = np.zeros((max(1, h // 8), max(1, w // 8), 3), dtype=np.uint8)
    dng_tags = [
        (50706, 1, 4, (1, 7, 0, 0), True),  # DNGVersion
        (50707, 1, 4, (1, 4, 0, 0), True),  # DNGBackwardVersion
        (50712, 3, len(table), tuple(int(x) for x in table), True),  # LinearizationTable
        (50714, 5, 3, (0, 1, 256, 1, 0, 1), True),  # BlackLevel: R=0, G=256, B=0
        (50717, 3, 3, (65535, 65535, 65535), True),  # WhiteLevel
        (50728, 5, 3, (477612, 1000000, 1000000, 1000000, 474074, 1000000), True),  # AsShotNeutral
    ]
    with tifffile.TiffWriter(path) as tw:
        tw.write(thumb, photometric="rgb", subfiletype=1, subifds=1, extratags=dng_tags)
        tw.write(raw_codes, photometric=34892, subfiletype=0, planarconfig="contig", compression=52546)


def _write_minimal_linearraw_dng(path: str, h: int, w: int) -> None:
    """A single-IFD 3-sample LinearRaw DNG with no compression libraw can't handle —
    stands in for "libraw can decode this fine" without needing the full SilverFast
    thumbnail+SubIFD+IR-page layout, which is unrelated to what's under test here."""
    rgb = np.random.randint(0, 65535, (h, w, 3)).astype(np.uint16)
    tifffile.imwrite(path, rgb, photometric=34892)


def test_peek_linear_dng_rgb_applies_linearization_and_black_white_level():
    """The decoded codes are table indices, not values: verify the lookup, per-channel
    BlackLevel subtraction, and WhiteLevel normalization all land where hand-computed."""
    from negpy.infrastructure.loaders.rawpy_loader import _peek_linear_dng_rgb

    h, w = 12, 10
    table = np.linspace(0, 65535, 1024).astype(np.uint16)
    codes = np.full((h, w, 3), 511, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dxo.dng")
        _write_linear_dng_libraw_cant_read(path, h, w, codes, table)
        result = _peek_linear_dng_rgb(path)

    assert result is not None
    rgb, wb_gains = result
    assert rgb.shape == (h, w, 3)
    assert rgb.dtype == np.float32

    linear = float(table[511])
    black = np.array([0.0, 256.0, 0.0])
    white = np.array([65535.0, 65535.0, 65535.0])
    expected = np.clip((linear - black) / (white - black), 0.0, 1.0)
    np.testing.assert_allclose(rgb[0, 0], expected, atol=1e-5)

    # AsShotNeutral (0.477612, 1.0, 0.474074) -> gains normalized to green=1.
    assert wb_gains is not None
    np.testing.assert_allclose(wb_gains, (1.0 / 0.477612, 1.0, 1.0 / 0.474074), rtol=1e-4)


def test_rawpy_loader_falls_back_to_tifffile_when_libraw_cant_unpack():
    """The end-to-end loader path: libraw genuinely fails on this file (not mocked), so
    RawpyLoader must return the tifffile-decoded NonStandardFileWrapper instead of the
    (unusable) rawpy object."""
    h, w = 12, 10
    table = np.linspace(0, 65535, 1024).astype(np.uint16)
    codes = np.random.default_rng(0).integers(0, 1024, (h, w, 3)).astype(np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dxo.dng")
        _write_linear_dng_libraw_cant_read(path, h, w, codes, table)
        ctx_mgr, metadata = LoaderFactory().get_loader(path)

    assert isinstance(ctx_mgr, NonStandardFileWrapper)
    assert ctx_mgr.data.shape == (h, w, 3)
    assert ctx_mgr.wb_gains is not None
    assert metadata["ir"] is None
    assert metadata["color_space"] is None


def test_nonstandard_wrapper_applies_wb_gains_only_when_camera_wb_requested():
    """Mirrors the two call shapes _decode_sensor_rgb actually uses: linear_raw=True ->
    use_camera_wb=False (untouched data); linear_raw=False -> use_camera_wb=True (gained)."""
    data = np.full((4, 4, 3), 0.2, dtype=np.float32)
    wrapper = NonStandardFileWrapper(data, wb_gains=(2.0, 1.0, 3.0))

    neutral = wrapper.postprocess(gamma=(1, 1), output_bps=16, use_camera_wb=False)
    np.testing.assert_allclose(neutral[..., 0], neutral[..., 1])

    gained = wrapper.postprocess(gamma=(1, 1), output_bps=16, use_camera_wb=True)
    assert gained[0, 0, 0] > gained[0, 0, 1]
    assert gained[0, 0, 2] > gained[0, 0, 1]
    np.testing.assert_allclose(gained[..., 0].astype(np.float64) / gained[..., 1].astype(np.float64), 2.0, atol=0.01)


def test_rawpy_loader_keeps_libraw_decode_when_unpack_succeeds():
    """A DNG libraw *can* unpack must not fall back to the tifffile path — the fallback
    is failure-driven, not a blanket bypass for every 3-sample LinearRaw DNG."""
    from unittest.mock import MagicMock, patch

    h, w = 40, 24
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "camera.dng")
        _write_minimal_linearraw_dng(path, h, w)
        sentinel = MagicMock(name="rawpy_object")
        with patch("negpy.infrastructure.loaders.rawpy_loader.rawpy.imread", return_value=sentinel):
            ctx_mgr, _metadata = LoaderFactory().get_loader(path)

    sentinel.unpack.assert_called_once()
    assert ctx_mgr is sentinel


def test_rawpy_loader_reraises_when_unpack_fails_without_a_linearraw_frame():
    """A broken/unsupported DNG with no LinearRaw page to fall back to must fail exactly as
    before — the new fallback must not swallow unrelated decode failures."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "camera.dng")
        tifffile.imwrite(path, np.full((20, 16, 3), 1000, dtype=np.uint16), photometric="rgb")
        try:
            LoaderFactory().get_loader(path)
        except rawpy.LibRawError:
            pass
        else:
            raise AssertionError("expected rawpy.LibRawError to propagate")


def test_rawpy_loader_reraises_when_the_linearraw_page_itself_is_unreadable():
    """LinearRaw-shaped (so _find_linearraw_page matches) but the pixel data is corrupt,
    so _peek_linear_dng_rgb's own tifffile decode also fails. The original LibRawError
    must still propagate — not a tifffile-side error, and not a silent bad decode."""
    from unittest.mock import MagicMock, patch

    h, w = 12, 10
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "corrupt.dng")
        rgb = np.random.randint(0, 65535, (h, w, 3)).astype(np.uint16)
        with tifffile.TiffWriter(path) as tw:
            tw.write(rgb, photometric=34892, planarconfig="contig", compression="zlib")

        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            offsets, counts = page.dataoffsets, page.databytecounts

        # Corrupt the compressed strip bytes in place — the directory (and so the
        # LinearRaw match) stays valid, only the pixel payload becomes undecodable.
        with open(path, "r+b") as f:
            for off, cnt in zip(offsets, counts):
                f.seek(off)
                f.write(bytes([0xFF]) * cnt)

        sentinel = MagicMock(name="rawpy_object")
        sentinel.unpack.side_effect = rawpy.LibRawFileUnsupportedError(b"boom")
        with patch("negpy.infrastructure.loaders.rawpy_loader.rawpy.imread", return_value=sentinel):
            try:
                LoaderFactory().get_loader(path)
            except rawpy.LibRawError:
                pass
            else:
                raise AssertionError("expected rawpy.LibRawError to propagate")


if __name__ == "__main__":
    unittest.main()
