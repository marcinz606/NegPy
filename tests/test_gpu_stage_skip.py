"""Incremental GPU rendering: stage skipping, the retouch bypass and the autocrop cache.

``process_to_texture`` resumes from the first dirty stage when it is given a
``source_hash``. That path had never run in production (the caller passed only
``analysis_source_hash``), so these tests pin its two contracts: an incremental
render is bit-identical to a full one, and the caches it relies on hit and miss
in the right places.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.altprocess.models import AltProcess
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.gpu.device import GPUDevice


def _gpu_available() -> bool:
    return GPUDevice.get().is_available


def _sub(cfg: WorkspaceConfig, name: str, **kw) -> WorkspaceConfig:
    return replace(cfg, **{name: replace(getattr(cfg, name), **kw)})


@unittest.skipUnless(_gpu_available(), "GPU not available")
class TestStageSkipParity(unittest.TestCase):
    """Every stage boundary: resuming must equal re-running the whole chain."""

    @classmethod
    def setUpClass(cls):
        from negpy.services.rendering.gpu_engine import GPUEngine

        cls.GPUEngine = GPUEngine
        rng = np.random.default_rng(0)
        cls.img = rng.random((384, 512, 3), dtype=np.float32) * 0.6 + 0.05

    def setUp(self):
        self.inc = self.GPUEngine()
        self.ref = self.GPUEngine()
        self.addCleanup(self.inc.destroy_all)
        self.addCleanup(self.ref.destroy_all)

    def _assert_same(self, label, cfg, *, source_hash="frame", size=1600.0, scale=1.0):
        inc_tex, _ = self.inc.process_to_texture(
            self.img,
            cfg,
            scale_factor=scale,
            render_size_ref=size,
            readback_metrics=True,
            source_hash=source_hash,
            analysis_source_hash=source_hash,
        )
        incremental = inc_tex.readback().copy()
        # source_hash=None is the full-rebuild path: upload + start_stage 0.
        ref_tex, _ = self.ref.process_to_texture(
            self.img,
            cfg,
            scale_factor=scale,
            render_size_ref=size,
            readback_metrics=True,
            source_hash=None,
            analysis_source_hash=source_hash,
        )
        full = ref_tex.readback().copy()
        self.assertEqual(incremental.shape, full.shape, f"{label}: shape drift")
        self.assertTrue(np.array_equal(incremental, full), f"{label}: incremental render differs from a full one")

    def test_every_stage_boundary_is_bit_identical(self):
        base = WorkspaceConfig()
        lit = _sub(base, "exposure", density=0.35)
        bw = _sub(lit, "process", process_mode=ProcessMode.BW)
        for label, cfg in (
            ("baseline", base),
            ("exposure.density", lit),
            ("exposure.grade", _sub(lit, "exposure", grade=3.0)),
            ("process.luma_range_clip", _sub(lit, "process", luma_range_clip=0.4)),
            ("lab.clahe on", _sub(lit, "lab", clahe_strength=0.5)),
            ("lab.clahe off", lit),
            ("lab.sharpen", _sub(lit, "lab", sharpen=0.8)),
            ("lab.saturation", _sub(lit, "lab", saturation=1.4)),
            ("retouch spot added", _sub(lit, "retouch", manual_dust_spots=[(0.5, 0.5, 100.0)])),
            ("retouch cleared", _sub(lit, "retouch", manual_dust_spots=[])),
            ("altproc bw", bw),
            ("lith on", _sub(bw, "altproc", alt_process=AltProcess.LITH)),
            ("lith.snatch", _sub(bw, "altproc", alt_process=AltProcess.LITH, lith_snatch=0.8)),
            ("cyanotype on", _sub(bw, "altproc", alt_process=AltProcess.CYANOTYPE)),
            ("cyanotype.scale", _sub(bw, "altproc", alt_process=AltProcess.CYANOTYPE, cyano_scale=2.4)),
            ("altproc off", bw),
            ("toning.sepia", _sub(lit, "toning", sepia_strength=0.4)),
            ("finish.border", _sub(lit, "finish", border_size=4.0)),
            ("geometry.rotation", _sub(lit, "geometry", rotation=1)),
            ("geometry.rotation back", lit),
            ("geometry.flip_horizontal", _sub(lit, "geometry", flip_horizontal=True)),
            ("geometry.autocrop_offset", _sub(lit, "geometry", autocrop_offset=6.0)),
            ("process_mode BW", _sub(lit, "process", process_mode=ProcessMode.BW)),
            ("process_mode C41", _sub(lit, "process", process_mode=ProcessMode.C41)),
            ("back to baseline", base),
        ):
            with self.subTest(change=label):
                self._assert_same(label, cfg)

    def test_render_size_and_scale_changes_rebuild(self):
        """Neither is carried by a config field, so both must force a full re-run."""
        cfg = _sub(WorkspaceConfig(), "exposure", density=0.35)
        self._assert_same("warm-up", cfg)
        self._assert_same("smaller print", cfg, size=900.0)
        self._assert_same("back to full print", cfg, size=1600.0)
        self._assert_same("scale 2.0", cfg, scale=2.0)
        self._assert_same("scale 1.0", cfg, scale=1.0)

    def test_cleanup_forces_a_full_rebuild(self):
        """cleanup() destroys the stage textures a resume would paint onto."""
        cfg = _sub(WorkspaceConfig(), "exposure", density=0.35)
        self._assert_same("warm-up", cfg)
        self.inc.cleanup(collect=False)
        self.assertIsNone(self.inc._current_source_hash)
        self.assertIsNone(self.inc._last_settings)
        self._assert_same("after cleanup", cfg)

    def test_export_render_does_not_poison_the_next_preview(self):
        """process() shares the engine and passes no source_hash."""
        cfg = _sub(WorkspaceConfig(), "exposure", density=0.35)
        self._assert_same("warm-up", cfg)
        self.inc.process(self.img, _sub(cfg, "exposure", density=0.9), scale_factor=1.0, readback_metrics=False)
        self._assert_same("preview after export", cfg)


@unittest.skipUnless(_gpu_available(), "GPU not available")
class TestLocalMapCache(unittest.TestCase):
    """The dodge/burn EV map is rasterised on the CPU and uploaded whole."""

    @classmethod
    def setUpClass(cls):
        from negpy.features.local.models import LocalMask, MaskShape
        from negpy.services.rendering.gpu_engine import GPUEngine

        cls.GPUEngine = GPUEngine
        rng = np.random.default_rng(3)
        cls.img = rng.random((256, 320, 3), dtype=np.float32) * 0.5 + 0.2
        cls.mask = LocalMask(
            vertices=((0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)),
            stops=0.8,
            grade=1.0,
            shape=MaskShape.POLYGON,
        )

    def setUp(self):
        self.eng = self.GPUEngine()
        self.addCleanup(self.eng.destroy_all)
        base = WorkspaceConfig()
        self.cfg = replace(base, local=replace(base.local, masks=(self.mask,)))

    def _render(self, cfg):
        self.eng.process_to_texture(
            self.img, cfg, scale_factor=1.0, readback_metrics=False, source_hash="frame", analysis_source_hash="frame"
        )

    def _count_rasterisations(self, fn):
        import negpy.services.rendering.gpu_engine as ge

        calls = {"n": 0}
        real = ge.compute_local_maps

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        ge.compute_local_maps = spy
        try:
            fn()
        finally:
            ge.compute_local_maps = real
        return calls["n"]

    def test_creative_drag_rasterises_once(self):
        def drag():
            for d in (0.1, 0.2, 0.3, 0.4):
                self._render(_sub(self.cfg, "exposure", density=d))

        self.assertEqual(self._count_rasterisations(drag), 1)

    def test_grade_change_rebuilds_the_map(self):
        """The green lane carries a grade-derived slope factor."""

        def move_grade():
            self._render(self.cfg)
            self._render(_sub(self.cfg, "exposure", grade=4.0))

        self.assertEqual(self._count_rasterisations(move_grade), 2)

    def test_mask_edit_rebuilds_the_map(self):
        def move_mask():
            self._render(self.cfg)
            edited = replace(self.cfg.local, masks=(replace(self.mask, stops=-0.5),))
            self._render(replace(self.cfg, local=edited))

        self.assertEqual(self._count_rasterisations(move_mask), 2)


@unittest.skipUnless(_gpu_available(), "GPU not available")
class TestSourcePreCorrectionCache(unittest.TestCase):
    """Flat-field and sensor unmix are full-buffer passes no creative slider moves."""

    MATRIX = (1.06, -0.04, -0.02, -0.05, 1.08, -0.03, -0.01, -0.04, 1.05)

    @classmethod
    def setUpClass(cls):
        from negpy.services.rendering.image_processor import ImageProcessor

        cls.ImageProcessor = ImageProcessor
        rng = np.random.default_rng(4)
        cls.img = rng.random((256, 320, 3), dtype=np.float32) * 0.5 + 0.2

    def setUp(self):
        self.proc = self.ImageProcessor()
        self.addCleanup(self.proc.destroy_all)
        # The unmix only applies on a Linear RAW basis (see sensor_unmix_available).
        self.cfg = _sub(WorkspaceConfig(), "process", sensor_matrix=self.MATRIX, linear_raw=True)

    def _render(self, cfg, img=None):
        buffer, _ = self.proc.run_pipeline(
            self.img if img is None else img,
            cfg,
            "frame",
            render_size_ref=1600.0,
            prefer_gpu=True,
            readback_metrics=False,
        )
        return buffer.readback().copy()

    def _count_corrections(self, fn):
        import negpy.services.rendering.image_processor as ipm

        calls = {"n": 0}
        real = ipm.apply_sensor_correction

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        ipm.apply_sensor_correction = spy
        try:
            fn()
        finally:
            ipm.apply_sensor_correction = real
        return calls["n"]

    def test_cached_render_matches_an_uncached_one(self):
        from negpy.features.process.sensor import effective_sensor_matrix

        self.assertIsNotNone(effective_sensor_matrix(self.cfg.process), "matrix must be active for this test to mean anything")
        cached = self._render(_sub(self.cfg, "exposure", density=0.4))
        fresh = self.ImageProcessor()
        try:
            uncached, _ = fresh.run_pipeline(
                self.img, _sub(self.cfg, "exposure", density=0.4), "frame", render_size_ref=1600.0, prefer_gpu=True, readback_metrics=False
            )
            self.assertTrue(np.array_equal(cached, uncached.readback()))
        finally:
            fresh.destroy_all()

    def test_creative_drag_corrects_once(self):
        def drag():
            for d in (0.1, 0.2, 0.3, 0.4):
                self._render(_sub(self.cfg, "exposure", density=d))

        self.assertEqual(self._count_corrections(drag), 1)

    def test_matrix_change_recorrects(self):
        def change():
            self._render(self.cfg)
            self._render(_sub(self.cfg, "process", sensor_matrix=(1.1, 0, 0, 0, 1.1, 0, 0, 0, 1.1), linear_raw=True))

        self.assertEqual(self._count_corrections(change), 2)

    def test_resolution_change_recorrects(self):
        """HQ preview re-decodes the same file larger under an unchanged source hash."""
        rng = np.random.default_rng(5)
        big = rng.random((512, 640, 3), dtype=np.float32) * 0.5 + 0.2

        def toggle_hq():
            self._render(self.cfg)
            self._render(self.cfg, img=big)

        self.assertEqual(self._count_corrections(toggle_hq), 2)


if __name__ == "__main__":
    unittest.main()
