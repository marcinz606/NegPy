"""Local grade: printing one masked area at its own contrast — the darkroom's burn-in
through the hard filter. Covers the ΔR → slope-ratio map, the pivot-preserving rotation
in the curve, and the end-to-end guarantee that unmasked pixels are untouched."""

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import apply_characteristic_curve, local_grade_factor_map
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.local.models import LocalAdjustmentsConfig, PolygonMask
from negpy.services.rendering.engine import DarkroomEngine

R_MIN = float(EXPOSURE_CONSTANTS["iso_r_min"])
R_MAX = float(EXPOSURE_CONSTANTS["iso_r_max"])


class TestFactorMap:
    def test_no_delta_is_no_change(self):
        out = local_grade_factor_map(np.zeros((4, 4), dtype=np.float32), 115.0)
        np.testing.assert_array_equal(out, np.ones((4, 4), dtype=np.float32))

    def test_harder_grade_steepens_the_slope(self):
        """k is proportional to 1/R, so a negative ΔR (harder paper) must scale slope up."""
        out = local_grade_factor_map(np.full((2, 2), -20.0, dtype=np.float32), 115.0)
        assert out[0, 0] == pytest.approx(115.0 / 95.0, rel=1e-5)

    def test_softer_grade_flattens_the_slope(self):
        out = local_grade_factor_map(np.full((2, 2), 25.0, dtype=np.float32), 115.0)
        assert out[0, 0] == pytest.approx(115.0 / 140.0, rel=1e-5)

    def test_both_ends_clamp_to_the_iso_r_ladder(self):
        deltas = np.array([[-1000.0, 1000.0]], dtype=np.float32)
        out = local_grade_factor_map(deltas, 115.0)
        assert out[0, 0] == pytest.approx(115.0 / R_MIN, rel=1e-5)
        assert out[0, 1] == pytest.approx(115.0 / R_MAX, rel=1e-5)

    def test_frame_grade_off_the_ladder_is_clamped_too(self):
        out = local_grade_factor_map(np.zeros((1, 1), dtype=np.float32), 5000.0)
        assert out[0, 0] == pytest.approx(1.0)


class TestCurveRotation:
    """The rotation happens about the channel pivot, which is the whole reason a
    grade-only mask changes contrast without moving the region's own midtone."""

    PIVOT, SLOPE = 0.5, 4.0

    def _curve(self, vals: np.ndarray, factor: float) -> np.ndarray:
        img = np.stack([vals, vals, vals], axis=-1).astype(np.float32)
        grade_map = np.full(vals.shape, factor, dtype=np.float32)
        params = (self.PIVOT, self.SLOPE)
        return np.asarray(apply_characteristic_curve(img, params, params, params, grade_map=grade_map))[:, :, 1]

    def test_pivot_value_is_untouched(self):
        vals = np.array([[self.PIVOT, self.PIVOT]], dtype=np.float32)
        np.testing.assert_allclose(self._curve(vals, 1.3), self._curve(vals, 1.0), rtol=1e-5)

    def test_harder_local_grade_separates_tones_further(self):
        vals = np.array([[self.PIVOT - 0.1, self.PIVOT + 0.1]], dtype=np.float32)
        flat = self._curve(vals, 1.0)
        hard = self._curve(vals, 1.3)
        assert abs(hard[0, 1] - hard[0, 0]) > abs(flat[0, 1] - flat[0, 0])

    def test_softer_local_grade_pulls_tones_together(self):
        vals = np.array([[self.PIVOT - 0.1, self.PIVOT + 0.1]], dtype=np.float32)
        flat = self._curve(vals, 1.0)
        soft = self._curve(vals, 0.7)
        assert abs(soft[0, 1] - soft[0, 0]) < abs(flat[0, 1] - flat[0, 0])

    def test_no_map_matches_a_map_of_ones(self):
        vals = np.array([[0.3, 0.5, 0.7]], dtype=np.float32)
        img = np.stack([vals, vals, vals], axis=-1)
        params = (self.PIVOT, self.SLOPE)
        without = np.asarray(apply_characteristic_curve(img, params, params, params))
        with_ones = np.asarray(apply_characteristic_curve(img, params, params, params, grade_map=np.ones_like(vals)))
        np.testing.assert_array_equal(without, with_ones)


class TestEndToEnd:
    """Through the real CPU pipeline: a mask over the top half only."""

    def _frame(self) -> np.ndarray:
        ramp = np.linspace(0.05, 0.9, 48, dtype=np.float32)
        img = np.repeat(ramp[None, :], 48, axis=0)
        return np.ascontiguousarray(np.stack([img, img * 0.95, img * 0.9], axis=-1))

    def _config(self, grade_delta: float) -> WorkspaceConfig:
        mask = PolygonMask(vertices=((0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)), stops=0.0, feather=0.0, grade=grade_delta)
        return WorkspaceConfig(local=LocalAdjustmentsConfig(masks=(mask,)))

    def _render(self, grade_delta: float) -> np.ndarray:
        return np.asarray(DarkroomEngine().process(self._frame(), self._config(grade_delta), source_hash=f"local-grade-{grade_delta}"))

    def test_masked_half_gains_contrast(self):
        flat = self._render(0.0)
        hard = self._render(-40.0)
        assert hard[:20].std() > flat[:20].std()

    def test_unmasked_half_is_bit_identical(self):
        """A local grade must stay local — anything else means it leaked into the
        frame-wide curve instead of riding the mask's alpha."""
        flat = self._render(0.0)
        hard = self._render(-40.0)
        np.testing.assert_array_equal(flat[28:], hard[28:])
