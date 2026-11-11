import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import rawpy
import tifffile

from negpy.infrastructure.loaders.factory import LoaderFactory
from negpy.infrastructure.loaders.rawpy_loader import RawpyLoader, _streaming_linearraw_memory_estimate
from negpy.services.rendering.prefetch_policy import decide_prefetch
from negpy.services.rendering.preview_cache import PreviewCacheUsage
from negpy.infrastructure.loaders.tiff_loader import NonStandardFileWrapper
from negpy.features.process.models import DemosaicMode
from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm, is_xtrans, resolve_demosaic, supported_demosaic_modes


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
            self.assertEqual(processed.dtype, np.float32)
            np.testing.assert_array_equal(processed, data)

    def test_xtrans_stays_above_ppg_for_3_pass_markesteijn(self):
        # LibRaw reads this only as a pass count for a 6x6 CFA: PPG or below drops
        # Markesteijn to 1 pass, which brings back dot/maze artifacts (see issue #272).
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=6)
        self.assertGreater(get_best_demosaic_algorithm(raw).value, rawpy.DemosaicAlgorithm.PPG.value)

    def test_bayer_uses_ahd(self):
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=2)
        self.assertEqual(get_best_demosaic_algorithm(raw), rawpy.DemosaicAlgorithm.AHD)

    def test_explicit_mode_wins_on_a_cfa(self):
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=2)
        self.assertEqual(get_best_demosaic_algorithm(raw, DemosaicMode.VNG), rawpy.DemosaicAlgorithm.VNG)
        self.assertEqual(get_best_demosaic_algorithm(raw, DemosaicMode.DHT), rawpy.DemosaicAlgorithm.DHT)

    def test_explicit_mode_ignored_without_a_cfa(self):
        # Nothing to interpolate: a de-mosaiced source and the numpy wrapper both stay LINEAR.
        self.assertEqual(
            get_best_demosaic_algorithm(_FakeRaw(rawpy.RawType.Stack, block_size=2), DemosaicMode.DHT),
            rawpy.DemosaicAlgorithm.LINEAR,
        )
        wrapper = NonStandardFileWrapper(np.zeros((4, 4, 3), dtype=np.float32))
        self.assertEqual(get_best_demosaic_algorithm(wrapper, DemosaicMode.DHT), rawpy.DemosaicAlgorithm.LINEAR)

    def test_unknown_mode_falls_back_to_auto(self):
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=2)
        # AMAZE and friends are absent from a permissive libraw build; a stored one must not raise.
        self.assertEqual(get_best_demosaic_algorithm(raw, "AMAZE"), rawpy.DemosaicAlgorithm.AHD)

    def test_supported_modes_are_offerable(self):
        modes = supported_demosaic_modes()
        self.assertIn(DemosaicMode.AUTO, modes)
        self.assertIn(DemosaicMode.AHD, modes)
        for mode in modes:
            if mode is DemosaicMode.AUTO:
                continue
            self.assertTrue(getattr(rawpy.DemosaicAlgorithm, mode.name).isSupported, mode)

    def test_is_xtrans(self):
        self.assertTrue(is_xtrans(_FakeRaw(rawpy.RawType.Flat, block_size=6)))
        self.assertFalse(is_xtrans(_FakeRaw(rawpy.RawType.Flat, block_size=2)))
        self.assertFalse(is_xtrans(object()))  # missing raw_pattern

    def test_resolve_demosaic_label_matches_the_algorithm_on_bayer(self):
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=2)
        algo, label = resolve_demosaic(raw, DemosaicMode.DCB)
        self.assertEqual(algo, rawpy.DemosaicAlgorithm.DCB)
        self.assertEqual(label, "DCB")

    def test_resolve_demosaic_label_is_markesteijn_on_xtrans(self):
        # The nominal algorithm never runs on a 6x6 CFA (see test_xtrans_stays_above_ppg_for_3_pass_markesteijn);
        # the label must say what LibRaw actually did, not the requested mode.
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=6)
        _, above_ppg = resolve_demosaic(raw, DemosaicMode.DHT)
        self.assertEqual(above_ppg, "Markesteijn 3-pass")
        _, at_or_below_ppg = resolve_demosaic(raw, DemosaicMode.LINEAR)
        self.assertEqual(at_or_below_ppg, "Markesteijn 1-pass")

    def test_resolve_demosaic_label_is_none_without_a_cfa(self):
        _, label = resolve_demosaic(_FakeRaw(rawpy.RawType.Stack, block_size=2), DemosaicMode.DHT)
        self.assertIsNone(label)
        wrapper = NonStandardFileWrapper(np.zeros((4, 4, 3), dtype=np.float32))
        _, wrapper_label = resolve_demosaic(wrapper, DemosaicMode.DHT)
        self.assertIsNone(wrapper_label)

    def test_cancelled_dng_stops_after_native_unpack(self):
        raw = unittest.mock.MagicMock()
        cancelled = unittest.mock.MagicMock(side_effect=[False, True])

        with (
            patch("negpy.infrastructure.loaders.rawpy_loader._is_dng", return_value=True),
            patch("negpy.infrastructure.loaders.rawpy_loader.rawpy.imread", return_value=raw),
            pytest.raises(InterruptedError),
        ):
            RawpyLoader().load("large.dng", preview_max_edge=1600, should_cancel=cancelled)

        raw.unpack.assert_called_once_with()
        raw.close.assert_called_once_with()


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
    rgb, color = result
    wb_gains = color.wb_gains
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


def test_jxl_linear_dng_preview_streams_without_a_full_array_decode():
    h, w = 12, 10
    table = np.linspace(0, 65535, 1024).astype(np.uint16)
    codes = np.full((h, w, 3), 511, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dxo.dng")
        _write_linear_dng_libraw_cant_read(path, h, w, codes, table)
        with (
            patch.object(tifffile.TiffPage, "asarray", side_effect=AssertionError("full array decoded")),
            patch("negpy.infrastructure.loaders.rawpy_loader.rawpy.imread") as imread,
        ):
            ctx_mgr, metadata = LoaderFactory().get_loader(path, preview_max_edge=6)

    imread.assert_not_called()
    assert isinstance(ctx_mgr, NonStandardFileWrapper)
    assert ctx_mgr.data.shape == (6, 5, 3)
    assert (ctx_mgr.sizes.raw_height, ctx_mgr.sizes.raw_width) == (h, w)
    assert metadata["ir"] is None
    linear = float(table[511])
    expected = np.clip((linear - np.array([0.0, 256.0, 0.0])) / (65535.0 - np.array([0.0, 256.0, 0.0])), 0.0, 1.0)
    np.testing.assert_allclose(ctx_mgr.data[0, 0], expected, atol=1e-5)


@pytest.mark.parametrize("orientation", [1, 6, 8])
def test_streamed_preview_reads_orientation_without_full_file_exif(tmp_path, orientation):
    path = tmp_path / "rgbi.dng"
    tifffile.imwrite(
        path,
        np.full((40, 60, 4), 32000, dtype=np.uint16),
        photometric=34892,
        planarconfig="contig",
        rowsperstrip=5,
        extratags=[(274, 3, 1, orientation, False)],
    )
    with patch("negpy.infrastructure.loaders.helpers.read_exif_from_file", side_effect=AssertionError("unbounded EXIF read")):
        context, metadata = LoaderFactory().get_loader(str(path), preview_max_edge=20)
    assert metadata["orientation"] == orientation
    assert max(context.data.shape[:2]) == 20


# Canon EOS R's ColorMatrix2 (D65), the matrix libraw's own table holds for that model.
_CANON_D65 = ((0.8293, -0.1789, -0.1094), (-0.5025, 1.2925, 0.2327), (-0.1199, 0.2769, 0.6108))
_CANON_A = ((0.9605, -0.3678, -0.0150), (-0.4268, 1.2125, 0.2412), (-0.0364, 0.1578, 0.6818))
_RAW_NEUTRAL = (0.5, 1.0, 0.625)


def _write_camera_linear_dng(path: str, compression, neutral=_RAW_NEUTRAL, h: int = 48, w: int = 64) -> None:
    """A LinearRaw camera DNG: a gray in `_RAW_NEUTRAL` ratios on the left, a warm patch on the right. `neutral` is the tag. With
    `compression=None` libraw reads it, with 52546 (JPEG XL) only the tifffile fallback does."""
    img = np.empty((h, w, 3), np.float64)
    img[:, : w // 2] = np.array(_RAW_NEUTRAL) * 0.4
    img[:, w // 2 :] = (0.3, 0.25, 0.1)

    def srational(m):
        return tuple(v for row in m for x in row for v in (int(round(x * 10000)), 10000))

    tags = [
        (50706, 1, 4, (1, 7, 0, 0), True),
        (50707, 1, 4, (1, 4, 0, 0), True),
        (271, 2, 0, "Canon", True),
        (272, 2, 0, "Canon EOS R", True),
        (50708, 2, 0, "Canon EOS R", True),
        (50721, "2i", 9, srational(_CANON_A), True),  # ColorMatrix1
        (50722, "2i", 9, srational(_CANON_D65), True),  # ColorMatrix2
        (50778, 3, 1, 17, True),  # CalibrationIlluminant1: Standard light A
        (50779, 3, 1, 21, True),  # CalibrationIlluminant2: D65
    ]
    if neutral is not None:
        tags.append((50728, 5, 3, tuple(v for n in neutral for v in (int(round(n * 1000000)), 1000000)), True))
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.zeros((h // 8, w // 8, 3), np.uint8), photometric="rgb", subfiletype=1, subifds=1, extratags=tags)
        codes = np.round(img * 65535).astype(np.uint16)
        tw.write(
            codes,
            photometric=34892,
            subfiletype=0,
            planarconfig="contig",
            compression=compression,
            extratags=[(50717, 3, 3, (65535,) * 3, True)],
        )


@pytest.mark.parametrize("preview_max_edge", [None, 32])
def test_jxl_camera_dng_carries_the_color_data_a_libraw_decode_carries(tmp_path, preview_max_edge):
    from negpy.infrastructure.loaders.helpers import camera_wb_multipliers, camera_xyz_matrix

    path = str(tmp_path / "enhanced.dng")
    _write_camera_linear_dng(path, 52546)
    ctx_mgr, _ = LoaderFactory().get_loader(path, preview_max_edge=preview_max_edge)

    assert isinstance(ctx_mgr, NonStandardFileWrapper)
    np.testing.assert_allclose(camera_wb_multipliers(ctx_mgr), (2.0, 1.0, 1.6), rtol=1e-6)
    np.testing.assert_allclose(camera_xyz_matrix(ctx_mgr), _CANON_D65, atol=1e-6)


def test_jxl_dng_without_an_as_shot_neutral_has_no_camera_color(tmp_path):
    from negpy.infrastructure.loaders.helpers import camera_wb_multipliers, camera_xyz_matrix

    path = str(tmp_path / "scan.dng")
    _write_camera_linear_dng(path, 52546, neutral=None)
    ctx_mgr, _ = LoaderFactory().get_loader(path)

    assert camera_wb_multipliers(ctx_mgr) is None
    assert camera_xyz_matrix(ctx_mgr) is None


def test_wrapper_postprocess_takes_user_wb_over_its_own_gains():
    data = np.full((2, 2, 3), 0.25, np.float32)
    wrapper = NonStandardFileWrapper(data, wb_gains=(2.0, 1.0, 1.6))
    linear = {"gamma": (1, 1), "output_bps": 16}

    np.testing.assert_allclose(wrapper.postprocess(use_camera_wb=True, **linear)[0, 0], (0.5, 0.25, 0.4))
    np.testing.assert_allclose(wrapper.postprocess(use_camera_wb=True, user_wb=[1, 1, 1, 1], **linear)[0, 0], (0.25,) * 3)
    np.testing.assert_allclose(wrapper.postprocess(user_wb=[3.0, 2.0, 4.0, 2.0], **linear)[0, 0], (0.375, 0.25, 0.5))


def test_jxl_dng_bracket_keeps_the_reference_white_balance(tmp_path):
    """The HDR solve decodes every frame after the first on the reference's as-shot gains."""
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    reference, sibling = str(tmp_path / "ref.dng"), str(tmp_path / "sib.dng")
    _write_camera_linear_dng(reference, 52546)
    _write_camera_linear_dng(sibling, 52546, neutral=(0.8, 1.0, 0.9))
    params = WorkspaceConfig()
    proc = ImageProcessor(use_gpu=False)
    ref_f32, _, _ = proc._decode_oriented_f32(reference, params)
    bracket_wb = proc.camera_wb_for(reference)
    sib_f32, _, _ = proc._decode_oriented_f32(sibling, params, wb_override=bracket_wb)

    np.testing.assert_allclose(bracket_wb, (2.0, 1.0, 1.6), rtol=1e-6)
    np.testing.assert_allclose(sib_f32, ref_f32, atol=1e-6)


def test_jxl_camera_dng_renders_the_transfer_like_a_libraw_one(tmp_path):
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.models import ProcessMode
    from negpy.services.rendering.image_processor import ImageProcessor

    base = WorkspaceConfig()
    slide = replace(base, process=replace(base.process, process_mode=ProcessMode.E6))
    proc = ImageProcessor(use_gpu=False)
    renders = {}
    for name, compression in (("libraw", None), ("jxl", 52546)):
        path = str(tmp_path / f"{name}.dng")
        _write_camera_linear_dng(path, compression)
        buf, _ = proc.render_export(path, slide, slide.export, name, prefer_gpu=False)
        renders[name] = buf

    h, w = renders["jxl"].shape[:2]
    gray = renders["jxl"][4 : h - 4, 4 : w // 2 - 4].reshape(-1, 3).mean(0)
    assert np.ptp(gray) < 0.01, f"the as-shot gray renders with a cast: {gray}"
    np.testing.assert_allclose(renders["jxl"], renders["libraw"], atol=0.01)


def test_jxl_linear_dng_allows_cancellable_neighbor_prefetch():
    h, w = 12, 10
    table = np.linspace(0, 65535, 1024).astype(np.uint16)
    codes = np.full((h, w, 3), 511, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dxo.dng")
        _write_linear_dng_libraw_cant_read(path, h, w, codes, table)

        factory = LoaderFactory()
        assert factory.estimate_linear_preview_prefetch_memory(path, 1600) is not None


def test_libraw_dng_prefetch_uses_the_whole_decode_estimate():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "camera.dng")
        _write_minimal_linearraw_dng(path, 12, 10)

        factory = LoaderFactory()
        assert factory.estimate_linear_preview_prefetch_memory(path, 1600) == factory.estimate_preview_memory(path, 1600)


def test_large_segmented_linearraw_uses_bounded_memory_estimate():
    width, height = 19_200, 12_752
    page = SimpleNamespace(
        shape=(height, width, 3),
        dtype=np.dtype(np.uint16),
        is_tiled=True,
        tilelength=512,
        tilewidth=512,
        rowsperstrip=None,
        tags={},
    )
    estimate = _streaming_linearraw_memory_estimate(page, page, 1600, normalize_tags=True)

    assert estimate is not None
    assert estimate.source_dimensions == (width, height)
    decision = decide_prefetch(
        estimate,
        PreviewCacheUsage(0, 0, 1, 512 * 1024 * 1024, 0, 0),
        2 * 1024 * 1024 * 1024,
        integrated_gpu=False,
    )
    assert decision.allowed


def test_large_non_segmented_linearraw_has_no_bounded_memory_estimate():
    width, height = 19_200, 12_752
    page = SimpleNamespace(
        shape=(height, width, 3),
        dtype=np.dtype(np.uint16),
        is_tiled=False,
        tilelength=None,
        tilewidth=None,
        rowsperstrip=height,
        tags={},
    )

    assert _streaming_linearraw_memory_estimate(page, page, 1600, normalize_tags=True) is None


def test_streamed_rgbi_preview_preserves_ir_hair_across_strips(tmp_path):
    from negpy.features.retouch.logic import downsample_ir

    h, w = 40, 80
    rgbi = np.full((h, w, 4), 60000, dtype=np.uint16)
    rgbi[:, :, :3] = 30000
    rgbi[7:9, 8:72, 3] = 6000
    path = tmp_path / "rgbi.dng"
    tifffile.imwrite(path, rgbi, photometric=34892, planarconfig="contig", rowsperstrip=8)

    context, metadata = LoaderFactory().get_loader(str(path), preview_max_edge=20)

    assert context.data.shape == (10, 20, 3)
    assert metadata["ir"].shape == (10, 20)
    ir = rgbi[:, :, 3].astype(np.float32) / 65535.0
    expected = downsample_ir(ir, 20)
    np.testing.assert_allclose(metadata["ir"], expected, atol=1e-6)
    area_only = cv2.resize(ir, (20, 10), interpolation=cv2.INTER_AREA)
    assert float(metadata["ir"].min()) < float(area_only.min())


def test_jxl_linear_dng_preview_does_not_fall_back_when_a_segment_is_too_large():
    h, w = 12, 10
    table = np.linspace(0, 65535, 1024).astype(np.uint16)
    codes = np.full((h, w, 3), 511, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dxo.dng")
        _write_linear_dng_libraw_cant_read(path, h, w, codes, table)
        with (
            patch("negpy.infrastructure.loaders.rawpy_loader._PREVIEW_SEGMENT_MAX_BYTES", 1),
            patch.object(tifffile.TiffPage, "asarray", side_effect=AssertionError("full array decoded")),
            patch("negpy.infrastructure.loaders.rawpy_loader.rawpy.imread") as imread,
            pytest.raises(RuntimeError, match="preview memory limit"),
        ):
            LoaderFactory().get_loader(path, preview_max_edge=6)

    imread.assert_not_called()


def test_jxl_linear_dng_preview_honors_cancellation_before_decode():
    h, w = 12, 10
    table = np.linspace(0, 65535, 1024).astype(np.uint16)
    codes = np.full((h, w, 3), 511, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dxo.dng")
        _write_linear_dng_libraw_cant_read(path, h, w, codes, table)
        with (
            patch.object(tifffile.TiffPage, "asarray", side_effect=AssertionError("full array decoded")),
            patch("negpy.infrastructure.loaders.rawpy_loader.rawpy.imread") as imread,
            pytest.raises(InterruptedError, match="preview load cancelled"),
        ):
            LoaderFactory().get_loader(path, preview_max_edge=6, should_cancel=lambda: True)

    imread.assert_not_called()


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
