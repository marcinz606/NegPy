"""Discovery's per-file passes run in parallel without disturbing asset order.

Order is what the filmstrip shows, so a faster pass that reorders assets is a
regression, not a win.
"""

import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image
from PyQt6.QtCore import QCoreApplication, QEventLoop, QObject, QThread, QTimer, pyqtSignal

from negpy.desktop.workers.render import AssetDiscoveryWorker, ThumbnailWorker
from negpy.desktop.workers import render as render_workers


class _ThumbnailEmitter(QObject):
    generate = pyqtSignal(list)
    requested = pyqtSignal(list)


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
        activity: list[str] = []
        worker.partial.connect(partial.append)
        worker.activity.connect(activity.append)

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
        return worker, partial, activity

    def test_each_completed_thumbnail_streams_before_the_next_source(self):
        _worker, partial, activity = self._run(3)
        self.assertEqual([len(chunk) for chunk in partial], [1, 1, 1])
        self.assertEqual(activity, ["h0-v3", "h1-v3", "h2-v3", ""])

    def test_real_worker_thread_drains_the_timer_queue(self):
        worker = ThumbnailWorker(None)
        thread = QThread()
        emitter = _ThumbnailEmitter()
        loop = QEventLoop()
        streamed: list[dict] = []
        activity: list[str] = []
        timed_out = False

        def record_activity(key: str) -> None:
            activity.append(key)
            if not key:
                loop.quit()

        def timeout() -> None:
            nonlocal timed_out
            timed_out = True
            loop.quit()

        worker.moveToThread(thread)
        emitter.requested.connect(worker.generate)
        worker.partial.connect(streamed.append)
        worker.activity.connect(record_activity)
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(timeout)
        files = [{"name": f"f{i}", "path": f"/tmp/f{i}.arw", "hash": f"h{i}"} for i in range(3)]

        thread.start()
        try:
            with patch(
                "negpy.services.assets.thumbnails.get_thumbnail_worker",
                lambda *a, **k: Image.new("RGB", (4, 4)),
            ):
                emitter.requested.emit(files)
                watchdog.start(2000)
                loop.exec()
        finally:
            watchdog.stop()
            worker.cancel_pending()
            thread.quit()
            thread.wait()

        self.assertFalse(timed_out, "the queued timer stopped before the batch finished")
        self.assertEqual([len(result) for result in streamed], [1, 1, 1])
        self.assertEqual(activity, ["h0-v3", "h1-v3", "h2-v3", ""])

    def test_cancel_stops_before_the_next_file(self):
        worker = ThumbnailWorker(None)
        activity: list[str] = []
        worker.activity.connect(activity.append)
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
        self.assertEqual(activity, ["h0-v3", ""])

    def test_quick_sources_finish_before_slow_fallbacks(self):
        worker = ThumbnailWorker(None)
        files = [{"name": f"f{i}", "path": f"/tmp/f{i}.dng", "hash": f"h{i}"} for i in range(2)]
        calls: list[tuple[str, bool]] = []

        def generate(path, *args, fast_only=False, **kwargs):
            calls.append((path, fast_only))
            return None if fast_only else Image.new("RGB", (4, 4))

        with patch("negpy.services.assets.thumbnails.get_thumbnail_worker", side_effect=generate):
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
                ("/tmp/f0.dng", False),
                ("/tmp/f1.dng", False),
            ],
        )

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
