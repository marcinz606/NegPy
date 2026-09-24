"""Tone-limited masks: the lith mask registered with the negative. A limited mask acts only
where the unburned print is lighter (Highlights) or darker (Shadows) than its zone, so a
sky burn stops at the skyline instead of darkening a band of it."""

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.analysis import zone_of_encoded
from negpy.features.exposure.logic import apply_characteristic_curve, local_grade_factor_map
from negpy.features.exposure.models import ExposureConfig
from negpy.features.exposure.placement import key_edges, predicted_zone
from negpy.features.lab.models import LabConfig
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskKey
from negpy.services.rendering.engine import DarkroomEngine

_FULL = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def _limited(key: MaskKey, zone: float = 6.0, softness: float = 1.0) -> LocalMask:
    return LocalMask(vertices=_FULL, stops=1.0, feather=0.0, key=key, key_zone=zone, key_softness=softness)


class TestKeyEdges:
    """The edges invert the achromatic print curve, so they must land on the zones the
    zone strip and zone placement read for the same tone."""

    EXPOSURE = ExposureConfig(auto_exposure=False, auto_normalize_contrast=False)

    def _zone(self, val: float) -> float:
        return predicted_zone(self.EXPOSURE, None, {}, val)

    def test_highlight_edges_land_on_the_zone_either_side(self):
        e0, e1 = key_edges(_limited(MaskKey.HIGHLIGHTS, 6.0, 1.0), self.EXPOSURE, None, {})
        assert self._zone(e0) == pytest.approx(5.5, abs=0.02)
        assert self._zone(e1) == pytest.approx(6.5, abs=0.02)

    def test_shadow_edges_are_the_highlight_edges_swapped(self):
        hi = key_edges(_limited(MaskKey.HIGHLIGHTS, 4.0, 2.0), self.EXPOSURE, None, {})
        sh = key_edges(_limited(MaskKey.SHADOWS, 4.0, 2.0), self.EXPOSURE, None, {})
        assert sh == pytest.approx((hi[1], hi[0]))

    def test_a_zone_past_paper_white_keeps_the_edges_apart(self):
        """The curve is flat beyond paper white; equal edges would divide by zero."""
        e0, e1 = key_edges(_limited(MaskKey.HIGHLIGHTS, 10.0, 1.0 / 3.0), self.EXPOSURE, None, {})
        assert e0 > e1


class TestKeyedKernel:
    """Two tones through the curve: 0.3 prints light, 0.7 prints dark. The Highlights
    edges below (key 0 at 0.6, 1 at 0.4) split them cleanly."""

    PARAMS = (0.5, 4.0)
    SCALE = (0.1, 0.1, 0.1)

    def _img(self, vals) -> np.ndarray:
        row = np.array([vals], dtype=np.float32)
        return np.ascontiguousarray(np.stack([row, row, row], axis=-1))

    def _curve(self, img, **kw) -> np.ndarray:
        p = self.PARAMS
        return np.asarray(apply_characteristic_curve(img, p, p, p, ev_scale=self.SCALE, **kw))

    def _keyed(self, img, params, alpha: float = 1.0, grade_deltas=None, frame_grade: float = 100.0) -> np.ndarray:
        h, w = img.shape[:2]
        return self._curve(
            img,
            key_alpha=np.full((h, w, 1), alpha, dtype=np.float32),
            key_params=np.array([params], dtype=np.float64),
            grade_deltas=np.zeros((h, w), dtype=np.float32) if grade_deltas is None else grade_deltas,
            frame_grade=frame_grade,
        )

    def test_the_light_tone_takes_the_burn_and_the_dark_tone_does_not(self):
        img = self._img([0.3, 0.7])
        burned = self._curve(img, ev_map=np.ones((1, 2), dtype=np.float32))
        plain = self._curve(img)
        keyed = self._keyed(img, (1.0, 0.0, 0.6, 0.4))
        np.testing.assert_allclose(keyed[0, 0], burned[0, 0], rtol=1e-6)
        np.testing.assert_array_equal(keyed[0, 1], plain[0, 1])

    def test_a_tone_midway_between_the_edges_takes_half_the_burn(self):
        img = self._img([0.5])
        half = self._curve(img, ev_map=np.full((1, 1), 0.5, dtype=np.float32))
        np.testing.assert_allclose(self._keyed(img, (1.0, 0.0, 0.6, 0.4)), half, rtol=1e-6)

    def test_the_shape_alpha_scales_the_burn(self):
        img = self._img([0.3])
        quarter = self._curve(img, ev_map=np.full((1, 1), 0.25, dtype=np.float32))
        np.testing.assert_allclose(self._keyed(img, (1.0, 0.0, 0.6, 0.4), alpha=0.25), quarter, rtol=1e-6)

    def test_a_limited_grade_prints_the_light_tone_at_its_own_grade(self):
        img = self._img([0.3, 0.7])
        factor = local_grade_factor_map(np.full((1, 2), -20.0, dtype=np.float32), 100.0)
        graded = self._curve(img, grade_map=factor)
        plain = self._curve(img)
        keyed = self._keyed(img, (0.0, -20.0, 0.6, 0.4))
        np.testing.assert_allclose(keyed[0, 0], graded[0, 0], rtol=1e-6)
        np.testing.assert_array_equal(keyed[0, 1], plain[0, 1])

    def test_limited_and_unlimited_grades_add_before_the_ladder_clamps(self):
        """R100 −30 −30 is R40, which clamps to R50: one clamp on the sum, not one per mask."""
        img = self._img([0.3, 0.7])
        deltas = np.full((1, 2), -30.0, dtype=np.float32)
        summed = self._curve(img, grade_map=local_grade_factor_map(np.array([[-60.0, -30.0]], dtype=np.float32), 100.0))
        keyed = self._keyed(img, (0.0, -30.0, 0.6, 0.4), grade_deltas=deltas)
        np.testing.assert_allclose(keyed, summed, rtol=1e-6)


class TestEndToEnd:
    """Through the real CPU pipeline: a full-frame +1 st burn limited to Highlights."""

    def _frame(self) -> np.ndarray:
        ramp = np.linspace(0.05, 0.9, 64, dtype=np.float32)
        img = np.repeat(ramp[None, :], 16, axis=0)
        return np.ascontiguousarray(np.stack([img, img * 0.95, img * 0.9], axis=-1))

    def _render(self, masks, tag: str) -> np.ndarray:
        cfg = WorkspaceConfig(local=LocalAdjustmentsConfig(masks=masks), lab=LabConfig(sharpen=0.0))
        return np.asarray(DarkroomEngine().process(self._frame(), cfg, source_hash=f"local-key-{tag}"))

    def test_the_burn_stops_at_the_zone(self):
        plain = self._render((), "plain")
        keyed = self._render((_limited(MaskKey.HIGHLIGHTS, 5.0, 1.0 / 3.0),), "keyed")
        # The engine's last step is the working OETF, so its output is already encoded.
        zones = zone_of_encoded(plain[..., 1])[0]
        light, dark = zones >= 6.0, zones <= 4.0
        assert light.any() and dark.any()
        np.testing.assert_array_equal(keyed[:, dark], plain[:, dark])
        assert np.all(keyed[:, light, 1] < plain[:, light, 1])
