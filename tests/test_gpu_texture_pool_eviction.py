"""Generation-based texture-pool eviction across export renders.

The pool survives a batch so a same-dimensions roll reuses its chain, but a
dimension change must free the old chain, or VRAM grows with every size seen.
"""

import unittest

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice


class TestTexturePoolEviction(unittest.TestCase):
    def setUp(self):
        if not GPUDevice.get().is_available:
            self.skipTest("GPU not available")
        from negpy.services.rendering.gpu_engine import GPUEngine

        self.eng = GPUEngine()
        self.addCleanup(self.eng.destroy_all)
        rng = np.random.default_rng(5)
        self.img_a = rng.random((256, 320, 3), dtype=np.float32) * 0.5 + 0.2
        self.img_b = rng.random((320, 200, 3), dtype=np.float32) * 0.5 + 0.2
        self.cfg = WorkspaceConfig()

    def test_same_dimensions_reuse_the_chain(self):
        self.eng.process(self.img_a, self.cfg)
        keys = set(self.eng._tex_cache)
        self.eng.process(self.img_a, self.cfg)
        self.assertEqual(set(self.eng._tex_cache), keys)

    def test_dimension_change_frees_the_old_chain(self):
        self.eng.process(self.img_a, self.cfg)
        keys_a = set(self.eng._tex_cache)

        # Grace render: the A chain is still pooled alongside B's.
        self.eng.process(self.img_b, self.cfg)
        after_first_b = set(self.eng._tex_cache)
        self.assertTrue(keys_a <= after_first_b)

        # Second B render evicts everything the first B render did not touch.
        self.eng.process(self.img_b, self.cfg)
        after_second_b = set(self.eng._tex_cache)
        self.assertLess(len(after_second_b), len(after_first_b))
        for gen in self.eng._tex_gen.values():
            self.assertGreaterEqual(gen, self.eng._render_gen - 1)

    def test_an_evicted_mask_texture_is_uploaded_again(self):
        """An evicted dodge/burn texture comes back empty, so its next upload must not be skipped."""
        from dataclasses import replace

        from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask

        mask = LocalMask(vertices=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)), stops=1.5)
        masked = replace(self.cfg, local=LocalAdjustmentsConfig(masks=(mask,)))
        first, _ = self.eng.process(self.img_a, masked, source_hash="a")
        self.eng.process(self.img_b, self.cfg, source_hash="b")
        self.eng.process(self.img_b, self.cfg, source_hash="b")
        again, _ = self.eng.process(self.img_a, masked, source_hash="a")
        np.testing.assert_allclose(np.asarray(again), np.asarray(first), atol=1e-4)

    def test_preview_renders_at_changing_sizes_keep_the_pool_bounded(self):
        from dataclasses import replace

        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        sizes = []
        for border in range(10):
            cfg = replace(self.cfg, finish=replace(self.cfg.finish, border_size=float(border)))
            processor.run_pipeline(self.img_a, cfg, "src", render_size_ref=320.0, prefer_gpu=True, readback_metrics=False)
            sizes.append(len(processor.engine_gpu._tex_cache))
        self.assertLessEqual(max(sizes[2:]), max(sizes[:2]) + 6, sizes)


if __name__ == "__main__":
    unittest.main()
