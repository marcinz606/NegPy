"""Discovery's per-file passes run in parallel without disturbing asset order.

Order is what the filmstrip shows, so a faster pass that reorders assets is a
regression, not a win.
"""

import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image
from PyQt6.QtCore import QCoreApplication, QObject, QThread, pyqtSignal

from negpy.desktop.workers.render import AssetDiscoveryWorker, ThumbnailWorker
from negpy.desktop.workers import render as render_workers


class _ThumbnailEmitter(QObject):
    generate = pyqtSignal(list)


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
    """The filmstrip queue yields between files and can be cancelled."""

    def _run(self, count: int):
        worker = ThumbnailWorker(None)
        partial: list[dict] = []
        finished: list[dict] = []
        worker.partial.connect(partial.append)
        worker.finished.connect(finished.append)

        files = [{"name": f"f{i}", "path": f"/tmp/f{i}.arw", "hash": f"h{i}"} for i in range(count)]
        with patch(
            "negpy.services.assets.thumbnails.get_thumbnail_worker",
            lambda *a, **k: Image.new("RGB", (4, 4)),
        ):
            worker.generate(files)
            worker._next_timer.stop()
            while worker._active:
                worker._process_next()
                worker._next_timer.stop()
        return worker, partial, finished

    def test_chunks_arrive_before_the_batch_finishes(self):
        _worker, partial, finished = self._run(20)
        self.assertTrue(partial, "no chunk was emitted during the batch")
        streamed = {key for chunk in partial for key in chunk}
        self.assertEqual(len(streamed | set(finished[0])), 20)

    def test_small_batch_still_completes(self):
        """Under one chunk nothing streams, and the final map still carries every file."""
        _worker, partial, finished = self._run(3)
        self.assertEqual(partial, [])
        self.assertEqual(len(finished[0]), 3)

    def test_slow_fallbacks_wait_until_every_fast_preview_was_tried(self):
        worker = ThumbnailWorker(None)
        finished: list[dict] = []
        worker.finished.connect(finished.append)
        files = [{"name": f"f{i}", "path": f"/tmp/f{i}.dng", "hash": f"h{i}"} for i in range(3)]
        calls: list[tuple[str, bool]] = []

        def thumbnail(path, *args, fast_only=False, **kwargs):
            calls.append((path, fast_only))
            if fast_only and path != "/tmp/f0.dng":
                return None
            return Image.new("RGB", (4, 4))

        with patch("negpy.services.assets.thumbnails.get_thumbnail_worker", side_effect=thumbnail):
            worker.generate(files)
            worker._next_timer.stop()
            while worker._active:
                worker._process_next()
                worker._next_timer.stop()

        self.assertEqual(
            calls,
            [
                ("/tmp/f0.dng", True),
                ("/tmp/f1.dng", True),
                ("/tmp/f2.dng", True),
                ("/tmp/f1.dng", False),
                ("/tmp/f2.dng", False),
            ],
        )
        self.assertEqual(set(finished[0]), {"h0-v3", "h1-v3", "h2-v3"})

    def test_cancel_stops_before_the_next_file(self):
        worker = ThumbnailWorker(None)
        finished: list[dict] = []
        worker.finished.connect(finished.append)
        files = [{"name": f"f{i}", "path": f"/tmp/f{i}.arw", "hash": f"h{i}"} for i in range(3)]
        calls: list[str] = []

        with patch(
            "negpy.services.assets.thumbnails.get_thumbnail_worker",
            side_effect=lambda path, *a, **k: calls.append(path) or Image.new("RGB", (4, 4)),
        ):
            worker.generate(files)
            worker._next_timer.stop()
            worker._process_next()
            worker._next_timer.stop()
            worker.cancel_pending()
            worker._process_next()

        self.assertEqual(calls, ["/tmp/f0.arw"])
        self.assertEqual(set(finished[0]), {"h0-v3"})

    def test_scheduler_is_created_in_the_worker_thread(self):
        app = QCoreApplication.instance() or QCoreApplication([])
        worker = ThumbnailWorker(None)
        thread = QThread()
        emitter = _ThumbnailEmitter()
        worker.moveToThread(thread)
        emitter.generate.connect(worker.generate)
        thread.start()
        try:
            files = [{"name": "f", "path": "/tmp/f.arw", "hash": "h"}]
            with patch(
                "negpy.services.assets.thumbnails.get_thumbnail_worker",
                return_value=Image.new("RGB", (4, 4)),
            ):
                emitter.generate.emit(files)
                deadline = time.monotonic() + 5
                while worker._next_timer is None and time.monotonic() < deadline:
                    time.sleep(0.01)
            self.assertIsNotNone(worker._next_timer)
            self.assertIs(worker._next_timer.thread(), thread)
            self.assertIsNotNone(app)
        finally:
            thread.quit()
            thread.wait()

    def test_thumbnail_decode_waits_for_foreground_memory_gate(self):
        worker = ThumbnailWorker(None)
        worker._files = [{"name": "f", "path": "/tmp/f.arw", "hash": "h"}]
        worker._total = 1
        worker._active = True
        entered = threading.Event()

        def thumbnail(*_args, **_kwargs):
            entered.set()
            worker._cancel_requested.set()
            return Image.new("RGB", (4, 4))

        render_workers._DECODE_MEMORY_GATE.acquire()
        try:
            with patch("negpy.services.assets.thumbnails.get_thumbnail_worker", side_effect=thumbnail):
                thread = threading.Thread(target=worker._process_next)
                thread.start()
                self.assertFalse(entered.wait(0.1), "thumbnail decode crossed the foreground gate")
                render_workers._DECODE_MEMORY_GATE.release()
                self.assertTrue(entered.wait(5))
                thread.join(5)
                self.assertFalse(thread.is_alive())
        finally:
            if render_workers._DECODE_MEMORY_GATE.locked():
                render_workers._DECODE_MEMORY_GATE.release()


if __name__ == "__main__":
    unittest.main()
