from unittest.mock import Mock, patch

import numpy as np
import pytest
import tifffile
from PIL import Image

from negpy.infrastructure.loaders.factory import LoaderFactory
from negpy.infrastructure.loaders.fff_loader import FffLoader
from negpy.infrastructure.loaders.helpers import bounded_tiff_page_preview, linear_uint16_to_display_uint8
from negpy.infrastructure.loaders.jpeg_loader import JpegLoader
from negpy.infrastructure.loaders.jxl_loader import JxlLoader
from negpy.infrastructure.loaders.nef_loader import NefLoader
from negpy.infrastructure.loaders.noritsu_loader import NoritsuLoader
from negpy.infrastructure.loaders.pakon_loader import PakonLoader
from negpy.infrastructure.loaders.tiff_loader import TiffLoader


def test_factory_delegates_the_bounded_preview_contract():
    factory = LoaderFactory()
    selected = Mock()
    expected = Mock()
    selected.load_bounded_preview.return_value = expected
    cancel = Mock(return_value=False)

    with patch.object(factory, "_select_loader", return_value=selected):
        result = factory.load_bounded_preview("scan.any", 96, fast_only=True, should_cancel=cancel)

    assert result is expected
    selected.load_bounded_preview.assert_called_once_with(
        "scan.any",
        96,
        fast_only=True,
        should_cancel=cancel,
    )


def test_jpeg_loader_returns_a_bounded_preview(tmp_path):
    path = str(tmp_path / "scan.jpg")
    Image.fromarray(np.full((120, 240, 3), 128, dtype=np.uint8)).save(path)

    result = JpegLoader().load_bounded_preview(path, 48)

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (48, 24)


def test_tiff_loader_streams_main_page_without_full_array_decode(tmp_path):
    path = str(tmp_path / "scan.tif")
    source = np.zeros((80, 120, 3), dtype=np.uint16)
    source[..., 1] = 32768
    tifffile.imwrite(path, source, tile=(16, 16), photometric="rgb")

    with (
        patch("negpy.infrastructure.loaders.tiff_loader._tiff_preview_page", return_value=None),
        patch.object(tifffile.TiffPage, "asarray", side_effect=AssertionError("full array decoded")),
    ):
        result = TiffLoader().load_bounded_preview(path, 40)

    assert result is not None
    assert result.size == (40, 27)
    preview = np.asarray(result)
    assert preview[..., 1].mean() > preview[..., 0].mean()


def test_scanner_tiff_loaders_use_the_quick_page_before_the_main_image():
    quick = Image.new("RGB", (120, 80))
    for loader_path, main_path, loader in (
        (
            "negpy.infrastructure.loaders.fff_loader._tiff_preview_page",
            "negpy.infrastructure.loaders.fff_loader.tifffile.TiffFile",
            FffLoader(),
        ),
        (
            "negpy.infrastructure.loaders.nef_loader._tiff_preview_page",
            "negpy.infrastructure.loaders.nef_loader.tifffile.TiffFile",
            NefLoader(),
        ),
    ):
        with patch(loader_path, return_value=quick), patch(main_path) as full:
            result = loader.load_bounded_preview("scan.raw", 48, fast_only=True)

        assert result is not None
        assert result.size == (48, 32)
        full.assert_not_called()


def test_noritsu_loader_samples_a_memory_map(tmp_path):
    path = str(tmp_path / "scan.raw")
    source = np.zeros((4, 8, 3), dtype="<u2")
    source[..., 0] = 65535
    source.tofile(path)

    with patch("negpy.infrastructure.loaders.noritsu_loader.detect_noritsu_dims", return_value=(8, 4)):
        result = NoritsuLoader().load_bounded_preview(path, 4)

    assert result is not None
    assert result.size == (4, 2)
    preview = np.asarray(result)
    assert preview[..., 2].mean() > preview[..., 0].mean()


def test_pakon_loader_samples_a_memory_map(tmp_path):
    path = str(tmp_path / "scan.raw")
    source = np.zeros((4, 8, 3), dtype="<u2")
    source[..., 2] = 65535
    source.tofile(path)

    with patch.object(PakonLoader, "PAKON_SPECS", [{"size": source.nbytes, "res": (4, 8), "desc": "test"}]):
        result = PakonLoader().load_bounded_preview(path, 4)

    assert result is not None
    assert result.size == (4, 2)


def test_direct_jxl_does_not_fall_back_to_a_full_decode():
    codec = Mock()
    with patch("negpy.infrastructure.loaders.jxl_loader.imagecodecs", codec):
        result = JxlLoader().load_bounded_preview("scan.jxl", 48)

    assert result is None
    codec.jpegxl_decode.assert_not_called()


def test_small_non_libraw_loaders_allow_neighbor_prefetch(tmp_path):
    jpeg_path = str(tmp_path / "small.jpg")
    tiff_path = str(tmp_path / "small.tif")
    Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(jpeg_path)
    tifffile.imwrite(tiff_path, np.zeros((20, 30, 3), dtype=np.uint16), photometric="rgb")
    factory = LoaderFactory()

    for path in (jpeg_path, tiff_path):
        assert factory.estimate_linear_preview_prefetch_memory(path, 1600) is not None


def test_large_loader_decode_is_estimated_and_left_to_the_ram_policy():
    factory = LoaderFactory()

    with patch.object(factory, "estimate_preview_memory") as estimate:
        estimate.return_value.temporary_bytes = 4 * 1024 * 1024 * 1024
        assert factory.estimate_linear_preview_prefetch_memory("large.tif", 1600) is estimate.return_value


def _single_strip_tiff(path, source, **kwargs):
    tifffile.imwrite(path, source, rowsperstrip=source.shape[0], photometric="rgb", **kwargs)


def _streamed(path, cancel=None):
    with tifffile.TiffFile(path) as tif:
        return bounded_tiff_page_preview(tif.pages[0], 40, should_cancel=cancel)


def test_single_strip_over_the_cap_streams_in_row_bands(tmp_path):
    yy, xx = np.mgrid[0:80, 0:120]
    source = np.stack([xx * 500, yy * 800, (xx + yy) * 300], axis=-1).astype(np.uint16)
    whole = Image.fromarray(linear_uint16_to_display_uint8(source)).resize((40, 27), Image.Resampling.BOX)
    expected = np.asarray(whole, dtype=np.int16)

    for byteorder in ("<", ">"):
        path = str(tmp_path / f"strip{byteorder == '>'}.tif")
        _single_strip_tiff(path, source, byteorder=byteorder)
        with (
            patch("negpy.infrastructure.loaders.helpers._TIFF_STREAM_PREVIEW_MAX_BYTES", 1024),
            patch("negpy.infrastructure.loaders.helpers._TIFF_STREAM_BAND_BYTES", 1024),
            patch.object(tifffile.TiffPage, "asarray", side_effect=AssertionError("full array decoded")),
            patch.object(tifffile.TiffPage, "segments", side_effect=AssertionError("segment decoded")),
        ):
            result = _streamed(path)

        assert result is not None
        assert result.size == (40, 27)
        assert np.abs(np.asarray(result, dtype=np.int16) - expected).max() <= 1


def test_banded_stream_checks_cancel_between_bands(tmp_path):
    path = str(tmp_path / "strip.tif")
    _single_strip_tiff(path, np.zeros((80, 120, 3), dtype=np.uint16))
    cancel = Mock(side_effect=[False, False, True])

    with (
        patch("negpy.infrastructure.loaders.helpers._TIFF_STREAM_PREVIEW_MAX_BYTES", 1024),
        patch("negpy.infrastructure.loaders.helpers._TIFF_STREAM_BAND_BYTES", 1024),
        pytest.raises(InterruptedError),
    ):
        _streamed(path, cancel)

    assert cancel.call_count == 3


def test_compressed_or_planar_strip_over_the_cap_is_not_streamed(tmp_path):
    source = np.zeros((80, 120, 3), dtype=np.uint16)
    compressed = str(tmp_path / "compressed.tif")
    _single_strip_tiff(compressed, source, compression="zlib")
    planar = str(tmp_path / "planar.tif")
    tifffile.imwrite(planar, source.transpose(2, 0, 1), rowsperstrip=80, photometric="rgb", planarconfig="separate")

    with patch("negpy.infrastructure.loaders.helpers._TIFF_STREAM_PREVIEW_MAX_BYTES", 1024):
        assert _streamed(compressed) is None
        assert _streamed(planar) is None


def _two_page_tiff(path, reduced, software=None, subfiletype=0, orientation=1):
    main = np.zeros((600, 400, 3), dtype=np.uint16)
    extratags = [(274, "H", 1, orientation, True)]
    tifffile.imwrite(path, main, photometric="rgb", software=software, extratags=extratags)
    tifffile.imwrite(path, reduced, photometric="rgb", append=True, subfiletype=subfiletype)


def _reduced_page(height=480, width=320):
    reduced = np.zeros((height, width, 3), dtype=np.uint8)
    reduced[..., 2] = 200
    return reduced


def _quick(path):
    return TiffLoader().load_bounded_preview(path, 64, fast_only=True)


def test_nikon_scan_reduced_page_is_the_quick_preview(tmp_path):
    path = str(tmp_path / "nikon.tif")
    _two_page_tiff(path, _reduced_page(), software="Nikon Scan 4.0.2 W", orientation=8)

    result = _quick(path)

    assert result is not None
    assert result.size == (64, 43)
    preview = np.asarray(result)
    assert abs(preview[..., 2].mean() - int(linear_uint16_to_display_uint8(np.uint16(200 << 8)))) <= 1
    assert preview[..., 0].max() == 0


def test_flagged_reduced_page_is_the_quick_preview_for_any_writer(tmp_path):
    path = str(tmp_path / "flagged.tif")
    _two_page_tiff(path, _reduced_page(), subfiletype=1)

    assert _quick(path) is not None


def test_unrelated_second_page_is_not_a_quick_preview(tmp_path):
    for name, reduced, software in (
        ("plain.tif", _reduced_page(), None),
        ("aspect.tif", _reduced_page(320, 320), "Nikon Scan 4.0.2 W"),
        ("small.tif", _reduced_page(120, 80), "Nikon Scan 4.0.2 W"),
    ):
        path = str(tmp_path / name)
        _two_page_tiff(path, reduced, software=software)

        assert _quick(path) is None, name
