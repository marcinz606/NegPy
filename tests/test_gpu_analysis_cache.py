"""Per-source auto-exposure analysis cache for the GPU engine.

The GPU preview path recomputed bounds/anchor/textural every frame (the CPU engine
caches them per source). These tests pin the pure cache helpers (no GPU) plus a
GPU-guarded end-to-end check that a creative-slider drag reuses the analysis.
"""

import unittest
from dataclasses import replace

from negpy.domain.models import WorkspaceConfig
from negpy.services.rendering.gpu_engine import (
    _analysis_cache_key,
    _fill_analysis_overrides,
    _update_analysis_cache,
)


class TestAnalysisCacheKey(unittest.TestCase):
    def setUp(self):
        self.cfg = WorkspaceConfig()

    def test_creative_slider_keeps_key(self):
        """density/grade and lab/toning changes must not invalidate the analysis key."""
        k0 = _analysis_cache_key(self.cfg, "src")
        for cfg in (
            replace(self.cfg, exposure=replace(self.cfg.exposure, density=self.cfg.exposure.density + 0.5)),
            replace(self.cfg, exposure=replace(self.cfg.exposure, grade=self.cfg.exposure.grade + 1.0)),
            replace(self.cfg, lab=replace(self.cfg.lab, saturation=0.3)),
            replace(self.cfg, toning=replace(self.cfg.toning, sepia_strength=0.4)),
        ):
            self.assertEqual(k0, _analysis_cache_key(cfg, "src"))

    def test_analysis_settings_change_key(self):
        """Anything feeding the meter must invalidate the key."""
        k0 = _analysis_cache_key(self.cfg, "src")
        variants = [
            replace(self.cfg, process=replace(self.cfg.process, analysis_buffer=self.cfg.process.analysis_buffer + 0.05)),
            replace(self.cfg, process=replace(self.cfg.process, luma_range_clip=0.05)),
            replace(self.cfg, process=replace(self.cfg.process, color_range_clip=0.05)),
            replace(self.cfg, geometry=replace(self.cfg.geometry, rotation=1)),
            replace(self.cfg, exposure=replace(self.cfg.exposure, cast_removal=not self.cfg.exposure.cast_removal)),
            replace(self.cfg, exposure=replace(self.cfg.exposure, auto_exposure=not self.cfg.exposure.auto_exposure)),
        ]
        for cfg in variants:
            self.assertNotEqual(k0, _analysis_cache_key(cfg, "src"))

    def test_source_identity_changes_key(self):
        self.assertNotEqual(_analysis_cache_key(self.cfg, "a"), _analysis_cache_key(self.cfg, "b"))


class TestFillAndUpdate(unittest.TestCase):
    KEY = ("src", "k")
    # Cache layout: (bounds, shadow_refs, anchor, textural, neutral_axis).

    def test_empty_cache_is_a_miss(self):
        self.assertEqual(_fill_analysis_overrides(None, self.KEY, None, None, None, None, None), (None, None, None, None, None))

    def test_matching_key_fills_none_overrides(self):
        cache = (self.KEY, "B", "R", 0.5, 2.0, "N")
        self.assertEqual(
            _fill_analysis_overrides(cache, self.KEY, None, None, None, None, None),
            ("B", "R", 0.5, 2.0, "N"),
        )

    def test_caller_overrides_win(self):
        cache = (self.KEY, "B", "R", 0.5, 2.0, "N")
        self.assertEqual(
            _fill_analysis_overrides(cache, self.KEY, "CALLER", None, 9.0, None, "N2"),
            ("CALLER", "R", 9.0, 2.0, "N2"),
        )

    def test_key_mismatch_is_a_miss(self):
        cache = (("other", "k"), "B", "R", 0.5, 2.0, "N")
        self.assertEqual(_fill_analysis_overrides(cache, self.KEY, None, None, None, None, None), (None, None, None, None, None))

    def test_update_stores_under_key(self):
        cache = _update_analysis_cache(None, self.KEY, "B", "R", 0.5, 2.0, "N")
        self.assertEqual(cache, (self.KEY, "B", "R", 0.5, 2.0, "N"))

    def test_update_merges_none_keeps_old(self):
        cache = (self.KEY, "B", "R", 0.5, 2.0, "N")
        # New frame computed only the anchor; the rest (incl. neutral axis) stay cached.
        merged = _update_analysis_cache(cache, self.KEY, None, None, 0.7, None, None)
        self.assertEqual(merged, (self.KEY, "B", "R", 0.7, 2.0, "N"))

    def test_update_resets_on_key_change(self):
        cache = (self.KEY, "B", "R", 0.5, 2.0, "N")
        merged = _update_analysis_cache(cache, ("new", "k"), "B2", None, None, None, None)
        self.assertEqual(merged, (("new", "k"), "B2", None, None, None, None))


# --------------------------------------------------------------------------- #
# GPU end-to-end: a creative-slider drag must not re-meter the source.
# --------------------------------------------------------------------------- #


def _gpu_available() -> bool:
    from negpy.infrastructure.gpu.device import GPUDevice

    return GPUDevice.get().is_available


class TestAnalysisReuseOnGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _gpu_available():
            raise unittest.SkipTest("GPU not available")
        import numpy as np

        from negpy.services.rendering.gpu_engine import GPUEngine

        cls.GPUEngine = GPUEngine
        rng = np.random.default_rng(0)
        cls.img = rng.random((512, 768, 3), dtype=np.float32) * 0.6 + 0.05

    def _render(self, eng, cfg, **kw):
        return eng.process_to_texture(self.img, cfg, scale_factor=1.0, readback_metrics=True, **kw)

    def test_density_drag_meters_once(self):
        import negpy.services.rendering.gpu_engine as ge

        eng = self.GPUEngine()
        try:
            calls = {"n": 0}
            real = ge.analyze_log_exposure_bounds

            def spy(*a, **k):
                calls["n"] += 1
                return real(*a, **k)

            ge.analyze_log_exposure_bounds = spy
            try:
                cfg = WorkspaceConfig()
                self._render(eng, cfg, analysis_source_hash="frame")
                cfg2 = replace(cfg, exposure=replace(cfg.exposure, density=cfg.exposure.density + 0.4))
                self._render(eng, cfg2, analysis_source_hash="frame")
            finally:
                ge.analyze_log_exposure_bounds = real
            self.assertEqual(calls["n"], 1, "analysis must be cached across a creative-slider change")
        finally:
            eng.destroy_all()

    def test_no_hash_disables_cache(self):
        import negpy.services.rendering.gpu_engine as ge

        eng = self.GPUEngine()
        try:
            calls = {"n": 0}
            real = ge.analyze_log_exposure_bounds

            def spy(*a, **k):
                calls["n"] += 1
                return real(*a, **k)

            ge.analyze_log_exposure_bounds = spy
            try:
                cfg = WorkspaceConfig()
                self._render(eng, cfg)  # no analysis_source_hash -> cache off
                self._render(eng, replace(cfg, exposure=replace(cfg.exposure, density=cfg.exposure.density + 0.4)))
            finally:
                ge.analyze_log_exposure_bounds = real
            self.assertEqual(calls["n"], 2, "without a source hash the cache must stay disabled")
        finally:
            eng.destroy_all()


if __name__ == "__main__":
    unittest.main()
