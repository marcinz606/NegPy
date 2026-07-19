"""Tests for Tier-3 rendering (negpy.services.roll.positive): a thin wrapper
around NegPy's own ImageProcessor.run_pipeline, exercised for real -- no
scanner or GPU involved, `prefer_gpu=False` forces the CPU engine, matching
how the rest of this test suite already drives ImageProcessor directly
(see tests/test_image_processor.py).
"""

from __future__ import annotations

import numpy as np

from negpy.services.rendering.image_processor import ImageProcessor
from negpy.services.roll import positive


def _negative_like(seed: int, shape=(40, 60)) -> np.ndarray:
    """A plausible scanner-linear C41 negative: red-biased (orange mask),
    with real per-pixel variance so the pipeline's percentile-based bounds
    analysis has something non-degenerate to measure."""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((*shape, 3), dtype=np.uint16)
    rgb[..., 0] = rng.integers(40000, 60000, size=shape)
    rgb[..., 1] = rng.integers(25000, 45000, size=shape)
    rgb[..., 2] = rng.integers(15000, 35000, size=shape)
    return rgb


class TestAvailable:
    def test_always_true(self) -> None:
        """Unlike coolscanpy or a Tier-2 repair engine, NegPy's own rendering
        pipeline ships with this application -- there is nothing to install."""
        assert positive.available() is True


class TestRenderPositive:
    def test_returns_uint16_rgb_of_the_same_shape(self) -> None:
        rgb = _negative_like(seed=1)
        result = positive.render_positive(rgb, processor=ImageProcessor())

        assert result.rgb.shape == rgb.shape
        assert result.rgb.dtype == np.uint16

    def test_uses_the_default_c41_print_conversion(self) -> None:
        """Stock WorkspaceConfig(): the same conversion a freshly opened
        negative gets before any slider is touched."""
        rgb = _negative_like(seed=2)
        result = positive.render_positive(rgb, processor=ImageProcessor())

        assert result.process_mode == "C41"
        assert result.render_intent == "print"
        assert result.auto_exposure is True

    def test_records_the_negpy_version(self) -> None:
        import negpy

        rgb = _negative_like(seed=3)
        result = positive.render_positive(rgb, processor=ImageProcessor())

        assert result.negpy_version == negpy.__version__

    def test_actually_inverts_not_a_passthrough(self) -> None:
        """A scanner-linear negative and its rendered positive must not be the
        same array -- catches a wrapper that accidentally short-circuits the
        engine (e.g. a stubbed process()) and returns the source untouched."""
        rgb = _negative_like(seed=4)
        result = positive.render_positive(rgb, processor=ImageProcessor())

        assert not np.array_equal(result.rgb, rgb)

    def test_two_different_frames_on_one_processor_render_differently(self) -> None:
        """Regression guard: render_positive must pass a fresh source_hash per
        call. DarkroomEngine's stage cache reuses a prior render whenever
        source_hash *and* the settings hash both match, and every Tier-3 call
        shares the same stock WorkspaceConfig -- a reused/stable hash across
        two different frames would silently hand back the first frame's
        pixels for the second. Two distinct frames rendered on the *same*
        ImageProcessor (so its cache is actually in play) must differ."""
        processor = ImageProcessor()
        first = positive.render_positive(_negative_like(seed=5), processor=processor)
        second = positive.render_positive(_negative_like(seed=6), processor=processor)

        assert not np.array_equal(first.rgb, second.rgb)
