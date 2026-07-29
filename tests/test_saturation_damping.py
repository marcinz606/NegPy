"""Dye Mute: the grade-tracking damping on density_saturation. Covers the pure
function, the off-by-default no-op, that it acts on density (not CIELAB a*/b*),
and that the retired lab.chroma_damping key drops silently."""

import logging

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import grade_saturation_damping
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.lab.models import LabConfig
from negpy.features.lab.processor import PhotoLabProcessor
from negpy.kernel.image.logic import rgb_to_lab_working
from negpy.services.rendering.engine import DarkroomEngine

_HARDEST_GRADE = 60.0


def _mean_chroma(img: np.ndarray) -> float:
    lab = rgb_to_lab_working(img)
    return float(np.mean(np.hypot(lab[..., 1], lab[..., 2])))


def _color_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0.05, 0.9, (16, 16, 3)).astype(np.float32)


class TestGradeSaturationDamping:
    def test_identity_at_zero_strength(self):
        assert grade_saturation_damping(5.0, 0.0) == 1.0

    def test_identity_at_zero_strength_for_every_slope(self):
        """Exactly 1.0, not approximately -- the default (0.0) has to leave the
        saturation matrix at identity so resolve_saturation_matrix returns None
        and the dye_mix fast path stays allocation-free."""
        assert all(grade_saturation_damping(s, 0.0) == 1.0 for s in (0.5, 2.0, 3.48, 5.8, 10.0, 50.0))

    def test_identity_at_softest_slope(self):
        assert grade_saturation_damping(EXPOSURE_CONSTANTS["slope_min"], 0.7) == 1.0

    def test_decreasing_in_slope(self):
        d = [grade_saturation_damping(s, 0.5) for s in (2.0, 3.0, 5.0, 10.0)]
        assert all(a > b for a, b in zip(d, d[1:]))

    def test_decreasing_in_strength(self):
        d = [grade_saturation_damping(4.0, k) for k in (0.0, 0.25, 0.5, 1.0)]
        assert all(a > b for a, b in zip(d, d[1:]))

    def test_slope_clamped(self):
        c = EXPOSURE_CONSTANTS
        assert grade_saturation_damping(0.5, 0.5) == grade_saturation_damping(c["slope_min"], 0.5)
        assert grade_saturation_damping(50.0, 0.5) == grade_saturation_damping(c["slope_max"], 0.5)


class TestLabSaturationIsGradeIndependent:
    """Dye Mute used to multiply into Lab Saturation, making it track the print
    slope. It now damps density_saturation instead, so the Lab stage must render
    identically whatever the curve did upstream."""

    def _run(self, img, slopes):
        ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2])
        if slopes is not None:
            ctx.metrics["print_slopes"] = slopes
        return PhotoLabProcessor(LabConfig(sharpen=0.0, saturation=1.4)).process(img, ctx)

    def test_slope_does_not_change_lab_saturation(self):
        img = _color_image()
        soft = self._run(img, (2.0, 2.0, 2.0))
        hard = self._run(img, (8.0, 8.0, 8.0))
        np.testing.assert_array_equal(soft, hard)

    def test_missing_slopes_matches_present_slopes(self):
        img = _color_image()
        np.testing.assert_array_equal(self._run(img, None), self._run(img, (8.0, 8.0, 8.0)))


class TestDampingRender:
    def _render(self, overrides):
        engine = DarkroomEngine()
        rng = np.random.default_rng(0)
        img = rng.uniform(0.05, 0.9, (32, 32, 3)).astype(np.float32)
        settings = WorkspaceConfig.from_flat_dict(
            {
                "paper_profile": "neutral",
                # Pin the slope to the Grade slider so damping is deterministic.
                "auto_exposure": False,
                "auto_normalize_contrast": False,
                **overrides,
            }
        )
        return engine.process(img, settings, source_hash="dye-mute-test")

    def test_default_off_is_exact_noop(self):
        baseline = self._render({"grade": _HARDEST_GRADE})
        explicit = self._render({"grade": _HARDEST_GRADE, "density_saturation_damping": 0.0})
        np.testing.assert_array_equal(baseline, explicit)

    def test_hard_grade_damping_reduces_chroma(self):
        off = self._render({"grade": _HARDEST_GRADE, "density_saturation_damping": 0.0})
        on = self._render({"grade": _HARDEST_GRADE, "density_saturation_damping": 0.5})
        assert _mean_chroma(on) < _mean_chroma(off)

    def test_effect_grows_as_grade_hardens(self):
        """The whole reason Dye Mute survives as its own control rather than
        folding into a static Print Saturation value: it tracks the grade."""

        def drop(grade):
            off = self._render({"grade": grade, "density_saturation_damping": 0.0})
            on = self._render({"grade": grade, "density_saturation_damping": 0.5})
            return _mean_chroma(off) - _mean_chroma(on)

        assert drop(_HARDEST_GRADE) > drop(120.0) > 0.0

    def test_damping_scales_global_but_not_trims(self):
        """Trims are absolute crossover corrections -- a measured fix for one
        paper's diverging dye layers must not shrink when the grade hardens."""
        no_trim = self._render({"grade": _HARDEST_GRADE, "density_saturation_damping": 0.5})
        with_trim = self._render({"grade": _HARDEST_GRADE, "density_saturation_damping": 0.5, "density_saturation_trim_red": 0.3})
        assert not np.array_equal(no_trim, with_trim)

    def test_bw_mode_is_inert(self):
        bw_off = self._render({"process_mode": "B&W", "grade": _HARDEST_GRADE, "density_saturation_damping": 0.0})
        bw_on = self._render({"process_mode": "B&W", "grade": _HARDEST_GRADE, "density_saturation_damping": 0.5})
        np.testing.assert_allclose(bw_off, bw_on, atol=1e-6)


class TestRetiredChromaDampingKey:
    def test_legacy_key_drops_without_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = WorkspaceConfig.from_flat_dict({"chroma_damping": 0.5})
        assert "chroma_damping" not in caplog.text
        assert config.exposure.density_saturation_damping == 0.0

    def test_value_is_not_carried_into_the_new_field(self):
        """Dropped, not renamed: the same strength means a different effect
        magnitude in density space, so carrying it would apply something the
        user never set."""
        config = WorkspaceConfig.from_flat_dict({"chroma_damping": 0.9})
        assert config.exposure.density_saturation_damping == 0.0
        assert not hasattr(config.lab, "chroma_damping")
