import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.normalization import resolve_analysis_region
from negpy.features.exposure.processor import NormalizationProcessor
from negpy.features.process.models import ProcessConfig, ProcessMode


def test_resolve_falls_back_to_roi_and_buffer_when_rect_none():
    roi, buf = resolve_analysis_region((100, 200), (10, 90, 20, 180), 0.05, None)
    assert roi == (10, 90, 20, 180)
    assert buf == 0.05


def test_resolve_rect_overrides_roi_and_zeroes_buffer():
    # Rect is normalized in the transformed image (H=100, W=200).
    roi, buf = resolve_analysis_region((100, 200), (10, 90, 20, 180), 0.05, (0.25, 0.5, 0.75, 1.0))
    assert roi == (50, 100, 50, 150)
    assert buf == 0.0


def test_resolve_ignores_degenerate_rect():
    # Zero-area rect can't blank analysis: fall back to the ROI + buffer.
    roi, buf = resolve_analysis_region((100, 100), None, 0.1, (0.5, 0.5, 0.5, 0.5))
    assert roi is None
    assert buf == 0.1


def _split_negative() -> np.ndarray:
    # Linear "negative": a dense (dark) left half and a thin (bright) right half.
    img = np.empty((200, 200, 3), dtype=np.float32)
    img[:, :100] = 0.15
    img[:, 100:] = 0.85
    return img


def _bounds_for_rect(rect):
    img = _split_negative()
    cfg = ProcessConfig(process_mode=ProcessMode.C41, analysis_buffer=0.0, analysis_rect=rect)
    ctx = PipelineContext(scale_factor=1.0, original_size=(200, 200))
    NormalizationProcessor(cfg).process(img, ctx)
    return ctx.metrics["log_bounds"]


def test_analysis_rect_changes_metered_bounds():
    # Metering only the dense (dark) half vs only the thin (bright) half must yield
    # different normalization bounds — proof the rect is wired through the meters.
    dark = _bounds_for_rect((0.0, 0.0, 0.5, 1.0))
    bright = _bounds_for_rect((0.5, 0.0, 1.0, 1.0))
    assert not np.allclose(dark.floors, bright.floors) or not np.allclose(dark.ceils, bright.ceils)
