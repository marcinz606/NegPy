import threading
import unittest
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.workers.render import _DecodeGate, PreviewLoadState, PreviewLoadTask, PreviewLoadWorker


def _enters_promptly(gate: _DecodeGate, kind: str, path: str, timeout: float = 1.0) -> bool:
    entered = threading.Event()

    def attempt() -> None:
        with gate.hold(kind, path):
            entered.set()

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    ok = entered.wait(timeout)
    thread.join(1.0)
    return ok


class TestDecodeGate(unittest.TestCase):
    def test_foreground_enters_alongside_a_prefetch_of_another_file(self):
        gate = _DecodeGate()
        with gate.hold("prefetch", "/neighbor.arw"):
            self.assertTrue(_enters_promptly(gate, "foreground", "/clicked.arw"))

    def test_foreground_waits_for_a_prefetch_of_the_same_file(self):
        gate = _DecodeGate()
        with gate.hold("prefetch", "/clicked.arw"):
            self.assertFalse(_enters_promptly(gate, "foreground", "/clicked.arw", timeout=0.1))
        self.assertTrue(_enters_promptly(gate, "foreground", "/clicked.arw"))

    def test_prefetch_waits_for_a_foreground_decode(self):
        gate = _DecodeGate()
        with gate.hold("foreground", "/clicked.arw"):
            self.assertFalse(_enters_promptly(gate, "prefetch", "/neighbor.arw", timeout=0.1))
        self.assertTrue(_enters_promptly(gate, "prefetch", "/neighbor.arw"))

    def test_thumbnail_waits_even_for_an_abandoned_prefetch(self):
        gate = _DecodeGate()
        with gate.hold("prefetch", "/neighbor.arw"):
            self.assertFalse(_enters_promptly(gate, "thumbnail", "", timeout=0.1))
        self.assertTrue(_enters_promptly(gate, "thumbnail", ""))

    def test_a_parked_foreground_wait_gives_up_once_its_task_is_stale(self):
        gate = _DecodeGate()
        with gate.hold("prefetch", "/clicked.arw"):
            with gate.hold("foreground", "/clicked.arw", abandoned=lambda: True) as entered:
                self.assertFalse(entered)
        # Giving up held nothing; the gate stays reusable.
        with gate.hold("foreground", "/clicked.arw") as entered:
            self.assertTrue(entered)


def _foreground_task(path: str, generation: int) -> PreviewLoadTask:
    return PreviewLoadTask(
        file_path=path,
        workspace_color_space="Adobe RGB",
        use_camera_wb=True,
        generation=generation,
        use_splash=False,
    )


def _prefetch_task(path: str, generation: int) -> PreviewLoadTask:
    return PreviewLoadTask(
        file_path=path,
        workspace_color_space="Adobe RGB",
        use_camera_wb=True,
        generation=generation,
        use_splash=False,
        for_cache_warm=True,
        file_hash="neighbor-hash",
    )


class TestForegroundOverlapsPrefetch(unittest.TestCase):
    def test_click_completes_while_a_prefetch_decode_is_still_running(self):
        state = PreviewLoadState()
        prefetch_started = threading.Event()
        release_prefetch = threading.Event()

        prefetch_service = MagicMock()

        def blocked_prefetch(*_args, **_kwargs):
            prefetch_started.set()
            if not release_prefetch.wait(10):
                raise AssertionError("prefetch was never released")
            return True

        prefetch_service.prefetch_linear_preview.side_effect = blocked_prefetch
        prefetch_worker = PreviewLoadWorker(prefetch_service, state=state)

        foreground_service = MagicMock()
        buf = np.ones((4, 4, 3), dtype=np.float32)
        foreground_service.load_linear_preview.return_value = (buf, (4, 4), {})
        foreground_worker = PreviewLoadWorker(foreground_service, state=state)

        state.expect_generation(1)
        prefetch_thread = threading.Thread(target=prefetch_worker.process, args=(_prefetch_task("/neighbor.arw", 1),), daemon=True)
        prefetch_thread.start()
        self.assertTrue(prefetch_started.wait(5))

        # The click, while the prefetch is stuck mid-read.
        state.expect_generation(2, "/clicked.arw")
        state.cancel_prefetch(1)
        finished = threading.Event()
        foreground_worker.finished.connect(lambda *_: finished.set())
        try:
            foreground_worker.process(_foreground_task("/clicked.arw", 2))
            self.assertTrue(finished.is_set(), "foreground load did not complete")
            self.assertTrue(prefetch_thread.is_alive(), "the test prefetch finished early; overlap was not exercised")
            foreground_service.load_linear_preview.assert_called_once()
        finally:
            release_prefetch.set()
            prefetch_thread.join(5)
        self.assertFalse(prefetch_thread.is_alive())

    def test_click_on_the_prefetched_file_waits_and_reuses_the_decode(self):
        state = PreviewLoadState()
        prefetch_started = threading.Event()
        release_prefetch = threading.Event()

        prefetch_service = MagicMock()

        def blocked_prefetch(*_args, **_kwargs):
            prefetch_started.set()
            release_prefetch.wait(10)
            return True

        prefetch_service.prefetch_linear_preview.side_effect = blocked_prefetch
        prefetch_worker = PreviewLoadWorker(prefetch_service, state=state)

        foreground_service = MagicMock()
        buf = np.ones((4, 4, 3), dtype=np.float32)
        foreground_service.load_linear_preview.return_value = (buf, (4, 4), {})
        foreground_worker = PreviewLoadWorker(foreground_service, state=state)

        state.expect_generation(1)
        prefetch_thread = threading.Thread(target=prefetch_worker.process, args=(_prefetch_task("/neighbor.arw", 1),), daemon=True)
        prefetch_thread.start()
        self.assertTrue(prefetch_started.wait(5))

        state.expect_generation(2, "/neighbor.arw")
        blocked = threading.Event()
        done = threading.Event()

        def click() -> None:
            blocked.set()
            foreground_worker.process(_foreground_task("/neighbor.arw", 2))
            done.set()

        click_thread = threading.Thread(target=click, daemon=True)
        click_thread.start()
        self.assertTrue(blocked.wait(5))
        self.assertFalse(done.wait(0.1), "the click crossed the gate while its own file was still decoding")
        release_prefetch.set()
        self.assertTrue(done.wait(5))
        prefetch_thread.join(5)
        click_thread.join(5)


if __name__ == "__main__":
    unittest.main()
