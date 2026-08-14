"""Discovery's per-file passes run in parallel without disturbing asset order.

Order is what the filmstrip shows, so a faster pass that reorders assets is a
regression, not a win.
"""

import threading
import unittest
from unittest.mock import patch

from PIL import Image

from negpy.desktop.workers.render import AssetDiscoveryWorker, ThumbnailWorker


class _Recorder:
    """Stands in for a pyqtSignal on a worker built without a QObject parent."""

    def __init__(self) -> None:
        self.calls: list = []

    def emit(self, *args) -> None:
        self.calls.append(args)


def _worker() -> AssetDiscoveryWorker:
    worker = AssetDiscoveryWorker.__new__(AssetDiscoveryWorker)
    worker.progress = _Recorder()
    return worker


class TestMapFiles(unittest.TestCase):
    def test_results_keep_input_order(self):
        import time

        paths = [f"f{i}" for i in range(20)]

        def _fn(path: str) -> str:
            # Descending sleeps: completions come back roughly reversed.
            time.sleep(0.002 * (len(paths) - paths.index(path)))
            return path

        self.assertEqual(_worker()._map_files(paths, _fn, str, 4), paths)

    def test_pass_is_actually_concurrent(self):
        """Serial execution can never let two files sit in the barrier at once."""
        barrier = threading.Barrier(2, timeout=5)

        def _fn(path: str) -> str:
            barrier.wait()
            return path

        out = _worker()._map_files(["a", "b"], _fn, str, 2)
        self.assertEqual(out, ["a", "b"])

    def test_progress_counts_every_file(self):
        worker = _worker()
        worker._map_files([f"f{i}" for i in range(6)], lambda p: p, str, 4)
        counts = [c[0] for c in worker.progress.calls]
        self.assertEqual(sorted(counts), [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(c[1] == 6 for c in worker.progress.calls))

    def test_a_failing_file_becomes_none_and_the_rest_survive(self):
        def _fn(path: str) -> str:
            if path == "bad":
                raise OSError("unreadable")
            return path.upper()

        out = _worker()._map_files(["a", "bad", "c"], _fn, str, 4)
        self.assertEqual(out, ["A", None, "C"])

    def test_single_file_skips_the_pool(self):
        out = _worker()._map_files(["only"], lambda p: p.upper(), str, 8)
        self.assertEqual(out, ["ONLY"])


class TestThumbnailStreaming(unittest.TestCase):
    """The filmstrip must fill in during the batch, not only at the end."""

    def _run(self, count: int):
        worker = ThumbnailWorker.__new__(ThumbnailWorker)
        worker._store = None
        worker.progress = _Recorder()
        worker.partial = _Recorder()
        worker.finished = _Recorder()
        worker.error = _Recorder()

        files = [{"name": f"f{i}", "path": f"/tmp/f{i}.arw", "hash": f"h{i}"} for i in range(count)]
        with patch(
            "negpy.services.assets.thumbnails.get_thumbnail_worker",
            lambda *a, **k: Image.new("RGB", (4, 4)),
        ):
            worker.generate(files)
        return worker

    def test_chunks_arrive_before_the_batch_finishes(self):
        worker = self._run(20)
        self.assertTrue(worker.partial.calls, "no chunk was emitted during the batch")
        streamed = {k for (chunk,) in worker.partial.calls for k in chunk}
        (final,) = worker.finished.calls
        self.assertTrue(streamed.issubset(final[0]), "a streamed key is missing from the final map")
        self.assertEqual(len(final[0]), 20)

    def test_small_batch_still_completes(self):
        """Under one chunk nothing streams, and the final map still carries every file."""
        worker = self._run(3)
        self.assertEqual(worker.partial.calls, [])
        self.assertEqual(len(worker.finished.calls[0][0]), 3)


if __name__ == "__main__":
    unittest.main()
