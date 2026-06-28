"""Splash (ephemeral) renders must not share an engine cache identity with the real
linear render of the same file — otherwise the embedded-JPEG analysis leaks into (and
gets persisted from) the linear render, leaving a wrong colour cast."""

import unittest
from unittest.mock import patch

import numpy as np

from negpy.domain.models import WorkspaceConfig


class TestEphemeralRenderCacheIsolation(unittest.TestCase):
    def _run(self, *tasks):
        with patch("negpy.desktop.workers.render.ImageProcessor") as MockIP:
            from negpy.desktop.workers.render import RenderTask, RenderWorker

            proc = MockIP.return_value
            proc.run_pipeline.return_value = (np.zeros((2, 2, 3), np.float32), {})
            worker = RenderWorker()
            for t in tasks:
                worker.process(t)
            return [c.args[2] for c in proc.run_pipeline.call_args_list], RenderTask

    def test_splash_render_gets_isolated_source_hash(self):
        from negpy.desktop.workers.render import RenderTask

        common = dict(buffer=np.zeros((2, 2, 3), np.float32), config=WorkspaceConfig(), preview_size=512.0)
        hashes, _ = self._run(
            RenderTask(source_hash="f1", ephemeral=False, **common),
            RenderTask(source_hash="f1", ephemeral=True, **common),
        )
        self.assertEqual(hashes[0], "f1")  # the real render keeps the file's cache identity
        self.assertNotEqual(hashes[1], hashes[0])  # the splash must not collide with it

    def test_non_ephemeral_keeps_source_hash_unchanged(self):
        from negpy.desktop.workers.render import RenderTask

        common = dict(buffer=np.zeros((2, 2, 3), np.float32), config=WorkspaceConfig(), preview_size=512.0)
        hashes, _ = self._run(RenderTask(source_hash="abc", ephemeral=False, **common))
        self.assertEqual(hashes[0], "abc")


if __name__ == "__main__":
    unittest.main()
