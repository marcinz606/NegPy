"""The filmstrip thumbnail must never read a GPU texture back on the Qt main thread.

A full-frame copy there stalls the UI mid-drag. The render worker owns the readback,
and is the only thread that can take it safely: the engine recycles stage textures
from its pool on the next frame.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.desktop.workers.render import RenderTask, RenderWorker
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


class _FakeTexture:
    """Stands in for GPUTexture; counts full readbacks."""

    def __init__(self, array):
        self._array = array
        self.height, self.width = array.shape[:2]
        self.readbacks = 0

    def readback(self):
        self.readbacks += 1
        return self._array


class TestControllerDoesNotReadBack(unittest.TestCase):
    def _controller_stub(self, metrics):
        from negpy.desktop.controller import AppController

        stub = SimpleNamespace(
            state=SimpleNamespace(
                current_file_path="/tmp/a.arw",
                current_file_hash="h1",
                selected_file_idx=0,
                uploaded_files=[{"name": "a.arw", "path": "/tmp/a.arw", "hash": "h1"}],
                last_metrics=metrics,
                metrics_lock=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None),
            ),
            display_transform_params=lambda splash=False: ("Adobe RGB", None, None),
            thumbnail_update_requested=MagicMock(),
        )
        # Attributing a render to its own frame is the caller's first step.
        stub._asset_for_render = lambda m: AppController._asset_for_render(stub, m)
        return AppController._update_thumbnail_from_state, stub

    def test_gpu_texture_is_not_read_back_on_the_ui_thread(self):
        tex = _FakeTexture(np.zeros((8, 8, 4), dtype=np.float32))
        fn, stub = self._controller_stub({"base_positive": tex, "source_hash": "h1"})
        with patch("negpy.desktop.controller.GPUTexture", _FakeTexture):
            fn(stub)
        self.assertEqual(tex.readbacks, 0, "the UI thread must not read back the render texture")
        stub.thumbnail_update_requested.emit.assert_not_called()

    def test_uses_the_host_copy_the_worker_attached(self):
        tex = _FakeTexture(np.zeros((8, 8, 4), dtype=np.float32))
        host = np.full((8, 8, 3), 0.25, dtype=np.float32)
        fn, stub = self._controller_stub({"base_positive": tex, "thumbnail_source": host, "source_hash": "h1"})
        with patch("negpy.desktop.controller.GPUTexture", _FakeTexture):
            fn(stub)
        self.assertEqual(tex.readbacks, 0)
        stub.thumbnail_update_requested.emit.assert_called_once()
        self.assertIs(stub.thumbnail_update_requested.emit.call_args[0][0].buffer, host)


class TestWorkerAttachesTheHostCopy(unittest.TestCase):
    """Only settle frames pay for it — a drag frame must stay readback-free."""

    def _worker(self, result):
        worker = RenderWorker.__new__(RenderWorker)
        super(RenderWorker, worker).__init__()
        worker._processor = SimpleNamespace(run_pipeline=lambda *a, **k: (result, {}))
        return worker

    def _run(self, worker, task):
        got = {}
        worker.finished.connect(lambda _r, m: got.update(m))
        with patch("negpy.desktop.workers.render.GPUTexture", _FakeTexture):
            worker.process(task)
        return got

    def _task(self, buffer, wants_thumbnail, preview_size=8.0):
        return RenderTask(
            buffer=np.zeros((8, 8, 3), dtype=np.float32),
            config=DEFAULT_WORKSPACE_CONFIG,
            source_hash="h",
            preview_size=preview_size,
            readback_metrics=True,
            wants_thumbnail=wants_thumbnail,
        )

    def test_attaches_a_host_copy_when_asked(self):
        tex = _FakeTexture(np.full((8, 8, 4), 0.5, dtype=np.float32))
        metrics = self._run(self._worker(tex), self._task(tex, wants_thumbnail=True))
        self.assertEqual(tex.readbacks, 1)
        self.assertIsInstance(metrics.get("thumbnail_source"), np.ndarray)
        self.assertEqual(metrics["thumbnail_source"].shape[2], 3, "alpha lane dropped for the thumbnail")

    def test_no_copy_when_not_asked(self):
        tex = _FakeTexture(np.full((8, 8, 4), 0.5, dtype=np.float32))
        metrics = self._run(self._worker(tex), self._task(tex, wants_thumbnail=False))
        self.assertEqual(tex.readbacks, 0)
        self.assertIsNone(metrics.get("thumbnail_source"))

    def test_hq_render_still_produces_a_thumbnail(self):
        """Switching images with HQ on never renders that image at preview size."""
        from negpy.kernel.system.config import APP_CONFIG

        tex = _FakeTexture(np.full((8, 8, 4), 0.5, dtype=np.float32))
        task = self._task(tex, wants_thumbnail=True, preview_size=float(APP_CONFIG.preview_render_size * 4))
        metrics = self._run(self._worker(tex), task)
        self.assertEqual(tex.readbacks, 1)
        self.assertIsInstance(metrics.get("thumbnail_source"), np.ndarray)

    def test_key_is_always_assigned_so_a_stale_copy_cannot_survive(self):
        """The controller merges metrics into a running dict, so a render that produced
        no copy must clear the previous image's."""
        tex = _FakeTexture(np.full((8, 8, 4), 0.5, dtype=np.float32))
        metrics = self._run(self._worker(tex), self._task(tex, wants_thumbnail=False))
        self.assertIn("thumbnail_source", metrics)

        running = {"thumbnail_source": np.full((4, 4, 3), 0.9, dtype=np.float32)}  # previous image
        running.update(metrics)
        self.assertIsNone(running["thumbnail_source"])

    def test_cpu_render_needs_no_copy(self):
        arr = np.full((8, 8, 3), 0.5, dtype=np.float32)
        metrics = self._run(self._worker(arr), self._task(arr, wants_thumbnail=True))
        self.assertIsNone(metrics.get("thumbnail_source"), "a host render is already its own thumbnail source")


if __name__ == "__main__":
    unittest.main()
