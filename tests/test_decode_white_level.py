"""Every RAW decode must pin its scale to the camera white level, not the frame's own maximum."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rawpy

from negpy.services.assets import thumbnails
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.services.rendering.preview_manager import PreviewManager


class _SpyRaw:
    raw_type = rawpy.RawType.Flat
    raw_pattern = np.zeros((2, 2), dtype=np.uint8)
    sizes = SimpleNamespace(raw_height=8, raw_width=8, iheight=8, iwidth=8)
    white_level = 16383
    camera_white_level_per_channel = None  # most bodies: no calibrated table
    black_level_per_channel = [0, 0, 0, 0]

    def __init__(self) -> None:
        self.seen: dict = {}

    def __enter__(self) -> "_SpyRaw":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def postprocess(self, **kwargs: object) -> np.ndarray:
        self.seen.update(kwargs)
        return np.zeros((8, 8, 3), dtype=np.uint16)


def test_sensor_decode_pins_the_white_level() -> None:
    raw = _SpyRaw()
    with patch("negpy.services.rendering.image_processor.loader_factory") as lf:
        lf.get_loader.return_value = (raw, {})
        ImageProcessor()._decode_sensor_rgb("/x.dng", linear_raw=True)
    assert raw.seen["adjust_maximum_thr"] == 0.0


def test_sensor_decode_pins_the_scale_to_the_camera_calibrated_limit() -> None:
    # A Nikon D800 reports a generic white_level of 16383 but a calibrated
    # camera_white_level_per_channel of 15311 — trusting the generic number lets
    # already non-linear photosites read as clean (issue #906, capture-side fix).
    class _Spy(_SpyRaw):
        white_level = 16383
        camera_white_level_per_channel = [15311, 15311, 15311, 15311]

    raw = _Spy()
    with patch("negpy.services.rendering.image_processor.loader_factory") as lf:
        lf.get_loader.return_value = (raw, {})
        ImageProcessor()._decode_sensor_rgb("/x.dng", linear_raw=True)
    assert raw.seen["user_sat"] == 15311


def test_sensor_decode_subtracts_the_black_level_from_the_calibrated_limit() -> None:
    # user_sat is compared post-black-subtraction; a non-zero black level must come off the
    # calibrated ceiling too, or the decode is silently under-scaled (invisible when black is 0,
    # as every other fixture in this file has it).
    class _Spy(_SpyRaw):
        white_level = 16383
        camera_white_level_per_channel = [15311, 15311, 15311, 15311]
        black_level_per_channel = [512, 512, 512, 512]

    raw = _Spy()
    with patch("negpy.services.rendering.image_processor.loader_factory") as lf:
        lf.get_loader.return_value = (raw, {})
        ImageProcessor()._decode_sensor_rgb("/x.dng", linear_raw=True)
    assert raw.seen["user_sat"] == 15311 - 512


def test_preview_decode_pins_the_white_level() -> None:
    raw = _SpyRaw()
    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (raw, {"color_space": "Adobe RGB"})
        PreviewManager().load_linear_preview("/x.dng", file_hash="abc")
    assert raw.seen["adjust_maximum_thr"] == 0.0


def test_preview_decode_pins_the_scale_to_the_camera_calibrated_limit() -> None:
    # Must match the export decode's scale (test_sensor_decode_pins_the_scale_to_the_camera_calibrated_limit),
    # or what a user tunes bounds against on screen disagrees with what export produces.
    class _Spy(_SpyRaw):
        white_level = 16383
        camera_white_level_per_channel = [15311, 15311, 15311, 15311]

    raw = _Spy()
    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (raw, {"color_space": "Adobe RGB"})
        PreviewManager().load_linear_preview("/x.dng", file_hash="abc")
    assert raw.seen["user_sat"] == 15311


def test_detection_decode_pins_the_white_level() -> None:
    raw = _SpyRaw()
    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (raw, {})
        PreviewManager().decode_for_detection("/x.dng")
    assert raw.seen["adjust_maximum_thr"] == 0.0


def test_detection_decode_pins_the_scale_to_the_camera_calibrated_limit() -> None:
    class _Spy(_SpyRaw):
        white_level = 16383
        camera_white_level_per_channel = [15311, 15311, 15311, 15311]

    raw = _Spy()
    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (raw, {})
        PreviewManager().decode_for_detection("/x.dng")
    assert raw.seen["user_sat"] == 15311


def test_preview_decode_falls_back_to_none_for_a_non_standard_source() -> None:
    from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper

    raw = NonStandardFileWrapper(data=np.zeros((8, 8, 3), dtype=np.float32))
    seen: dict = {}
    original_postprocess = raw.postprocess

    def spy_postprocess(**kwargs: object) -> np.ndarray:
        seen.update(kwargs)
        return original_postprocess(**kwargs)

    raw.postprocess = spy_postprocess  # type: ignore[method-assign]
    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (raw, {"color_space": "Adobe RGB"})
        PreviewManager().load_linear_preview("/x.tiff", file_hash="abc")
    assert seen["user_sat"] is None


def test_thumbnail_decode_pins_the_scale_to_the_camera_calibrated_limit() -> None:
    class _Spy(_SpyRaw):
        white_level = 16383
        camera_white_level_per_channel = [15311, 15311, 15311, 15311]

    raw = _Spy()
    thumbnails._fast_demosaic(raw)
    assert raw.seen["adjust_maximum_thr"] == 0.0
    assert raw.seen["user_sat"] == 15311


def test_thumbnail_decode_falls_back_to_none_for_a_non_standard_source() -> None:
    from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper

    raw = NonStandardFileWrapper(data=np.zeros((8, 8, 3), dtype=np.float32))
    seen: dict = {}
    original_postprocess = raw.postprocess

    def spy_postprocess(**kwargs: object) -> np.ndarray:
        seen.update(kwargs)
        return original_postprocess(**kwargs)

    raw.postprocess = spy_postprocess  # type: ignore[method-assign]
    thumbnails._fast_demosaic(raw)
    assert seen["user_sat"] is None
