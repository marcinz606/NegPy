"""Test-strip lifecycle: print → pick → commit, and every way it gets dropped.

The strip is a proof of the config as it stood when the patches were printed. If the edit
moves underneath it, the patches on screen no longer say what they claim to — so any real
render has to drop it, including one that lands while the strip is still printing.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.analysis import STRIP_DENSITIES, STRIP_GRADES, strip_cells


class TestStripWorker(unittest.TestCase):
    def test_every_patch_is_rendered_at_its_own_density_and_grade(self):
        with patch("negpy.desktop.workers.render.ImageProcessor") as MockIP:
            from negpy.desktop.workers.render import RenderWorker, TestStripTask

            seen: list = []

            def fake_pipeline(_buf, config, *a, **k):
                seen.append((config.exposure.density, config.exposure.grade))
                # Flat frame valued by call order, so the mosaic reveals which tile won.
                return np.full((8, 8, 3), float(len(seen) - 1), np.float32), {"content_rect": (0, 0, 8, 8)}

            MockIP.return_value.run_pipeline.side_effect = fake_pipeline
            worker = RenderWorker()
            done: list = []
            worker.strip_finished.connect(lambda m, r: done.append((m, r)))

            worker.build_strip(
                TestStripTask(
                    buffer=np.zeros((8, 8, 3), np.float32),
                    config=WorkspaceConfig(),
                    source_hash="f1",
                    preview_size=512.0,
                )
            )

        self.assertEqual(seen, [(d, g) for _, _, d, g in strip_cells()])
        mosaic, content_rect = done[0]
        self.assertEqual(content_rect, (0, 0, 8, 8))
        # Top-left patch came from the first render, bottom-right from the last.
        self.assertEqual(mosaic[0, 0, 0], 0.0)
        self.assertEqual(mosaic[-1, -1, 0], float(len(seen) - 1))

    def test_metrics_never_escape_the_strip(self):
        """A proof must not disturb the histogram/bounds writeback the real render owns."""
        with patch("negpy.desktop.workers.render.ImageProcessor") as MockIP:
            from negpy.desktop.workers.render import RenderWorker, TestStripTask

            MockIP.return_value.run_pipeline.side_effect = lambda *a, **k: (np.zeros((4, 4, 3), np.float32), {})
            worker = RenderWorker()
            leaked: list = []
            worker.metrics_updated.connect(leaked.append)
            worker.finished.connect(lambda *a: leaked.append(a))

            worker.build_strip(
                TestStripTask(
                    buffer=np.zeros((4, 4, 3), np.float32),
                    config=WorkspaceConfig(),
                    source_hash="f1",
                    preview_size=512.0,
                )
            )
            # readback_metrics must be off for every variant.
            for call in MockIP.return_value.run_pipeline.call_args_list:
                self.assertFalse(call.kwargs["readback_metrics"])

        self.assertEqual(leaked, [])


class TestStripLifecycle(unittest.TestCase):
    def setUp(self):
        from negpy.desktop.controller import AppController
        from negpy.desktop.session import AppState, DesktopSessionManager
        from negpy.services.rendering.preview_manager import PreviewManager

        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)
        self.controller.state.preview_raw = np.empty((8, 8, 3), dtype=np.float32)
        self.strip_tasks: list = []
        self.render_tasks: list = []
        self.announced: list = []
        self.controller.strip_requested.connect(self.strip_tasks.append)
        self.controller.render_requested.connect(self.render_tasks.append)
        self.controller.test_strip_changed.connect(self.announced.append)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _mosaic(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.float32)

    def _print_strip(self) -> None:
        self.controller.toggle_test_strip()
        self.controller.on_strip_finished(self._mosaic(), (0, 0, 8, 8))

    def test_toggling_on_dispatches_one_job_and_shows_nothing_yet(self):
        self.controller.toggle_test_strip()
        self.assertEqual(len(self.strip_tasks), 1)
        self.assertTrue(self.controller.state.test_strip_pending)
        # Nothing to paint until the patches land.
        self.assertFalse(self.controller.state.test_strip)
        self.assertEqual(self.announced, [False])

        self.controller.on_strip_finished(self._mosaic(), (0, 0, 8, 8))
        self.assertTrue(self.controller.state.test_strip)
        self.assertFalse(self.controller.state.test_strip_pending)
        self.assertEqual(self.controller.state.test_strip_content_rect, (0, 0, 8, 8))
        self.assertEqual(self.announced, [False, True])

    def test_toggling_off_while_printing_cancels_the_job(self):
        self.controller.toggle_test_strip()
        self.controller.toggle_test_strip()
        self.assertFalse(self.controller.state.test_strip_pending)

        # The in-flight job still finishes; its mosaic must be dropped, not shown.
        self.controller.on_strip_finished(self._mosaic(), None)
        self.assertFalse(self.controller.state.test_strip)
        self.assertIsNone(self.controller.state.test_strip_mosaic)

    def test_a_frame_with_nothing_loaded_prints_no_strip(self):
        self.controller.state.preview_raw = None
        self.controller.toggle_test_strip()
        self.assertEqual(self.strip_tasks, [])
        self.assertFalse(self.controller.state.test_strip_pending)

    def test_picking_a_patch_commits_its_settings_and_clears_the_strip(self):
        self._print_strip()
        self.controller.apply_test_strip_pick(0, 3)

        committed = self.mock_session_manager.update_config.call_args
        exposure = committed.args[0].exposure
        self.assertEqual(exposure.density, STRIP_DENSITIES[3])
        self.assertEqual(exposure.grade, STRIP_GRADES[0])
        self.assertTrue(committed.kwargs["persist"])
        self.assertFalse(self.controller.state.test_strip)
        self.assertIsNone(self.controller.state.test_strip_mosaic)

    def test_picking_leaves_the_auto_toggles_alone(self):
        """The patches were printed under the autos — flipping them would render
        something other than the patch that was clicked."""
        before = self.controller.state.config.exposure
        self._print_strip()
        self.controller.apply_test_strip_pick(2, 1)

        exposure = self.mock_session_manager.update_config.call_args.args[0].exposure
        self.assertEqual(exposure.auto_exposure, before.auto_exposure)
        self.assertEqual(exposure.auto_normalize_contrast, before.auto_normalize_contrast)

    def test_a_click_with_no_strip_up_commits_nothing(self):
        self.controller.apply_test_strip_pick(1, 1)
        self.mock_session_manager.update_config.assert_not_called()

    def test_any_real_render_drops_the_strip(self):
        self._print_strip()
        self.controller._is_rendering = False
        self.controller.request_render()

        self.assertFalse(self.controller.state.test_strip)
        self.assertIsNone(self.controller.state.test_strip_mosaic)
        self.assertIn(False, self.announced[1:])

    def test_an_override_render_on_its_own_leaves_the_strip_alone(self):
        """An override paints an alternate config without touching the edit, so it is not
        the invalidating kind of render — the modes that use one clear the strip themselves."""
        self._print_strip()
        self.controller._is_rendering = False
        self.controller.request_render(readback_metrics=False, config_override=WorkspaceConfig())

        self.assertTrue(self.controller.state.test_strip)

    def test_loading_another_frame_drops_the_strip(self):
        """load_file's navigate-back memo path repaints without request_render, so a strip
        left up would hang over a different frame."""
        self._print_strip()
        with patch.object(self.controller, "_file_hash_for_path", return_value=None):
            self.controller.load_file("/nowhere/other.raw")

        self.assertFalse(self.controller.state.test_strip)
        self.assertIsNone(self.controller.state.test_strip_mosaic)

    def test_compare_and_flat_peek_take_the_canvas_from_the_strip(self):
        for enter_mode in (self.controller.toggle_compare, lambda: self.controller.toggle_flat_peek(force=True)):
            self.controller.state.compare_mode = False
            self.controller.state.flat_peek = False
            self.controller._is_rendering = False
            self._print_strip()

            enter_mode()
            self.assertFalse(self.controller.state.test_strip)
            self.assertIsNone(self.controller.state.test_strip_mosaic)


if __name__ == "__main__":
    unittest.main()
