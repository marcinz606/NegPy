"""Every RAW decode must pin its scale to the camera white level, not the frame's own maximum."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rawpy

from negpy.services.rendering.image_processor import ImageProcessor
from negpy.services.rendering.preview_manager import PreviewManager


class _SpyRaw:
    raw_type = rawpy.RawType.Flat
    raw_pattern = np.zeros((2, 2), dtype=np.uint8)
    sizes = SimpleNamespace(raw_height=8, raw_width=8, iheight=8, iwidth=8)

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


def test_preview_decode_pins_the_white_level() -> None:
    raw = _SpyRaw()
    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (raw, {"color_space": "Adobe RGB"})
        PreviewManager().load_linear_preview("/x.dng", file_hash="abc")
    assert raw.seen["adjust_maximum_thr"] == 0.0


def test_detection_decode_pins_the_white_level() -> None:
    raw = _SpyRaw()
    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (raw, {})
        PreviewManager().decode_for_detection("/x.dng")
    assert raw.seen["adjust_maximum_thr"] == 0.0
