import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.logic import grade_chroma_damping
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.lab.models import LabConfig
from negpy.features.lab.processor import PhotoLabProcessor
from negpy.kernel.image.logic import rgb_to_lab_working


def _mean_chroma(img: np.ndarray) -> float:
    lab = rgb_to_lab_working(img)
    return float(np.mean(np.hypot(lab[..., 1], lab[..., 2])))


def _color_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0.05, 0.9, (16, 16, 3)).astype(np.float32)


class TestGradeChromaDamping:
    def test_identity_at_zero_strength(self):
        assert grade_chroma_damping(5.0, 0.0) == 1.0

    def test_identity_at_softest_slope(self):
        assert grade_chroma_damping(EXPOSURE_CONSTANTS["slope_min"], 0.7) == 1.0

    def test_decreasing_in_slope(self):
        d = [grade_chroma_damping(s, 0.5) for s in (2.0, 3.0, 5.0, 10.0)]
        assert all(a > b for a, b in zip(d, d[1:]))

    def test_decreasing_in_strength(self):
        d = [grade_chroma_damping(4.0, k) for k in (0.0, 0.25, 0.5, 1.0)]
        assert all(a > b for a, b in zip(d, d[1:]))

    def test_slope_clamped(self):
        c = EXPOSURE_CONSTANTS
        assert grade_chroma_damping(0.5, 0.5) == grade_chroma_damping(c["slope_min"], 0.5)
        assert grade_chroma_damping(50.0, 0.5) == grade_chroma_damping(c["slope_max"], 0.5)


class TestLabProcessorDamping:
    def _run(self, img, slopes):
        config = LabConfig(sharpen=0.0, chroma_damping=0.5)
        ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2])
        if slopes is not None:
            ctx.metrics["print_slopes"] = slopes
        return PhotoLabProcessor(config).process(img, ctx)

    def test_higher_slope_lower_chroma(self):
        img = _color_image()
        soft = self._run(img, (2.0, 2.0, 2.0))
        hard = self._run(img, (8.0, 8.0, 8.0))
        assert _mean_chroma(hard) < _mean_chroma(soft)

    def test_missing_slopes_is_noop(self):
        img = _color_image()
        out = self._run(img, None)
        np.testing.assert_allclose(out, np.clip(img, 0, 1), atol=1e-6)

    def test_strength_zero_is_noop(self):
        img = _color_image()
        config = LabConfig(sharpen=0.0, chroma_damping=0.0)
        ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2])
        ctx.metrics["print_slopes"] = (8.0, 8.0, 8.0)
        out = PhotoLabProcessor(config).process(img, ctx)
        np.testing.assert_allclose(out, np.clip(img, 0, 1), atol=1e-6)
