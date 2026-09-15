from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import tifffile

from negpy.infrastructure.loaders.factory import LoaderFactory
from negpy.infrastructure.loaders.helpers import _dng_tag_floats, dng_bounded_preview, dng_quick_preview
from negpy.infrastructure.loaders.rawpy_loader import RawpyLoader
from negpy.services.assets.thumbnails import (
    decode_bounded_source_preview,
    get_thumbnail_worker,
)


def test_reduced_dng_ifd_is_used_without_decoding_main_pixels(tmp_path):
    path = str(tmp_path / "pyramid.dng")
    preview = np.zeros((24, 32, 3), dtype=np.uint8)
    preview[..., 1] = 180
    main = np.zeros((240, 320, 3), dtype=np.uint16)
    main[..., 0] = 65535
    with tifffile.TiffWriter(path) as writer:
        writer.write(preview, photometric="rgb", subfiletype=1, subifds=1)
        writer.write(main, photometric=34892, subfiletype=0)

    original_asarray = tifffile.TiffPage.asarray

    def reduced_only(page, *args, **kwargs):
        tag = page.tags.get("NewSubfileType")
        if tag is None or not (int(tag.value) & 1):
            raise AssertionError("full-resolution DNG pixels were decoded")
        return original_asarray(page, *args, **kwargs)

    with patch.object(tifffile.TiffPage, "asarray", reduced_only):
        result = dng_quick_preview(path)

    assert result is not None
    arr = np.asarray(result)
    assert arr.shape == preview.shape
    assert arr[..., 1].min() == 180
    assert arr[..., 0].max() == 0


def test_full_resolution_only_dng_is_not_decoded_for_a_thumbnail(tmp_path):
    path = str(tmp_path / "main-only.dng")
    tifffile.imwrite(path, np.zeros((40, 60, 3), dtype=np.uint16), photometric=34892)
    raw = Mock()
    raw.__enter__ = Mock(return_value=raw)
    raw.__exit__ = Mock(return_value=None)
    raw.extract_thumb.side_effect = RuntimeError("no preview")

    with patch("negpy.infrastructure.loaders.rawpy_loader.rawpy.imread", return_value=raw):
        result = LoaderFactory().load_bounded_preview(path, 30, fast_only=True)

    assert result is None
    raw.postprocess.assert_not_called()


def test_linear_dng_tiles_stream_into_a_bounded_preview(tmp_path):
    path = str(tmp_path / "tiled-linear.dng")
    main = np.zeros((40, 60, 3), dtype=np.uint16)
    main[..., 1] = 32768
    tifffile.imwrite(path, main, tile=(16, 16), photometric="rgb")

    with (
        patch("negpy.infrastructure.loaders.helpers._DNG_LINEAR_RAW", 2),
        patch.object(tifffile.TiffPage, "asarray", side_effect=AssertionError("full array decoded")),
    ):
        handled, result = dng_bounded_preview(path, 30)

    assert handled
    assert result is not None
    assert result.size == (30, 20)
    preview = np.asarray(result)
    assert preview[..., 1].mean() > preview[..., 0].mean()


def test_linear_dng_stream_can_be_cancelled_between_tiles(tmp_path):
    path = str(tmp_path / "tiled-linear.dng")
    main = np.zeros((40, 60, 3), dtype=np.uint16)
    tifffile.imwrite(path, main, tile=(16, 16), photometric="rgb")
    checks = 0

    def cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    with patch("negpy.infrastructure.loaders.helpers._DNG_LINEAR_RAW", 2):
        with pytest.raises(InterruptedError):
            dng_bounded_preview(path, 30, should_cancel=cancel)


def test_dng_rational_tags_are_converted_to_floats():
    tag = SimpleNamespace(dtype=5, value=(1, 2, 3, 4, 5, 0))

    np.testing.assert_allclose(_dng_tag_floats(tag), (0.5, 0.75, 0.0))


def test_thumbnail_service_uses_loader_contract_without_full_decode():
    with patch("negpy.services.assets.thumbnails.loader_factory.load_bounded_preview", return_value=None) as bounded:
        result = decode_bounded_source_preview("camera.arw", max_edge=64)

    assert result is None
    bounded.assert_called_once_with("camera.arw", 64, fast_only=False, should_cancel=None)


def test_linear_dng_is_a_bounded_preview_implementation():
    expected = Mock()
    with (
        patch("negpy.infrastructure.loaders.rawpy_loader.dng_quick_preview", return_value=None),
        patch("negpy.infrastructure.loaders.rawpy_loader.dng_bounded_preview", return_value=(True, expected)) as bounded,
    ):
        result = RawpyLoader().load_bounded_preview("scanner.dng", 64)

    assert result is expected
    bounded.assert_called_once_with("scanner.dng", 64, should_cancel=None)


def test_bounded_preview_does_not_demosaic_raw_without_embedded_preview():
    class RawWithoutPreview:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def extract_thumb(self):
            raise RuntimeError("no preview")

    with (
        patch("negpy.infrastructure.loaders.rawpy_loader.dng_quick_preview", return_value=None),
        patch("negpy.infrastructure.loaders.rawpy_loader.dng_bounded_preview", return_value=(False, None)),
        patch("negpy.infrastructure.loaders.rawpy_loader.rawpy.imread", return_value=RawWithoutPreview()),
        patch.object(RawWithoutPreview, "postprocess", create=True) as demosaic,
    ):
        result = RawpyLoader().load_bounded_preview("camera.raw", 64)

    assert result is None
    demosaic.assert_not_called()


def test_fast_pass_defers_a_missing_preview():
    with patch("negpy.services.assets.thumbnails.decode_bounded_source_preview", return_value=None) as bounded:
        assert get_thumbnail_worker("main-only.dng", "hash", fast_only=True) is None

    bounded.assert_called_once()
    assert bounded.call_args.kwargs["fast_only"] is True


def test_cancelled_slow_pass_returns_without_a_thumbnail():
    with patch(
        "negpy.services.assets.thumbnails.decode_bounded_source_preview",
        side_effect=InterruptedError("cancelled"),
    ):
        assert get_thumbnail_worker("main-only.dng", "hash") is None


def test_missing_preview_is_retried():
    with patch("negpy.services.assets.thumbnails.decode_bounded_source_preview", return_value=None) as bounded:
        assert get_thumbnail_worker("main-only.dng", "hash") is None
        assert get_thumbnail_worker("main-only.dng", "hash") is None

    assert bounded.call_count == 2
