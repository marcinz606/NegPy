from unittest.mock import Mock, patch

import numpy as np
import tifffile
from PIL import Image

from negpy.infrastructure.loaders.factory import LoaderFactory
from negpy.infrastructure.loaders.jpeg_loader import JpegLoader
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


def test_small_non_libraw_loaders_allow_neighbor_prefetch(tmp_path):
    jpeg_path = str(tmp_path / "small.jpg")
    tiff_path = str(tmp_path / "small.tif")
    Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(jpeg_path)
    tifffile.imwrite(tiff_path, np.zeros((20, 30, 3), dtype=np.uint16), photometric="rgb")
    factory = LoaderFactory()

    for path in (jpeg_path, tiff_path):
        assert factory.estimate_linear_preview_prefetch_memory(path, 1600) is not None


def test_large_non_cooperative_loader_does_not_allow_neighbor_prefetch():
    factory = LoaderFactory()

    with patch.object(factory, "estimate_preview_memory") as estimate:
        estimate.return_value.temporary_bytes = 256 * 1024 * 1024 + 1
        assert factory.estimate_linear_preview_prefetch_memory("large.tif", 1600) is None
