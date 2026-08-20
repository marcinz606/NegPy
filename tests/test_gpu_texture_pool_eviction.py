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


if __name__ == "__main__":
    unittest.main()
