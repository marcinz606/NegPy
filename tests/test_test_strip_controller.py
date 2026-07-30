"""Test-strip lifecycle: print → pick → commit, and every way it gets dropped.

The strip is a proof of the config as it stood when the patches were printed. If the edit
moves underneath it, the patches on screen no longer say what they claim to — so any real
render has to drop it, including one that lands while the strip is still printing.
"""

import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.analysis import (
    STRIP_DENSITIES,
    STRIP_GRADES,
    STRIP_GRID,
    strip_cells,
    strip_overrides,
)


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
            ticks: list = []
            worker.strip_finished.connect(lambda m, r: done.append((m, r)))
            worker.strip_progress.connect(lambda n, t: ticks.append((n, t)))

            worker.build_strip(
                TestStripTask(
                    buffer=np.zeros((8, 8, 3), np.float32),
                    config=WorkspaceConfig(),
                    source_hash="f1",
                    preview_size=512.0,
                    overrides=tuple(strip_overrides()),
                    grid=STRIP_GRID,
                )
            )

        self.assertEqual(seen, [(d, g) for _, _, d, g in strip_cells()])
        # One progress tick per patch, counting up to the total.
        self.assertEqual(ticks, [(i + 1, len(seen)) for i in range(len(seen))])
        mosaics, content_rect = done[0]
        self.assertEqual(content_rect, (0, 0, 8, 8))
        self.assertEqual(len(mosaics), 4, "one assembled orientation per quarter-turn")
        # Unrotated: top-left patch came from the first render, bottom-right from the last.
        self.assertEqual(mosaics[0][0, 0, 0], 0.0)
        self.assertEqual(mosaics[0][-1, -1, 0], float(len(seen) - 1))
        # A quarter-turn CCW brings the right column up to the top row — the same 25 renders,
        # sliced differently, which is why rotating never re-renders.
        cols = STRIP_GRID[1]
        self.assertEqual(mosaics[1][0, 0, 0], float(cols - 1))  # was top-right
        self.assertEqual(mosaics[1][0, -1, 0], float(len(seen) - 1))  # was bottom-right

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
                    overrides=tuple(strip_overrides()),
                    grid=STRIP_GRID,
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
        # RenderMemo is keyed by file hash and no-ops without one, as the app always has.
        self.controller.state.current_file_hash = "f1"
        self.strip_tasks: list = []
        self.render_tasks: list = []
        self.announced: list = []
        self.toasts: list = []
        self.progress: list = []
        self.controller.strip_requested.connect(self.strip_tasks.append)
        self.controller.render_requested.connect(self.render_tasks.append)
        self.controller.test_strip_changed.connect(self.announced.append)
        self.controller.status_message_requested.connect(lambda msg, _ms: self.toasts.append(msg))
        self.controller.status_progress_requested.connect(lambda done, total: self.progress.append((done, total)))

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

    def _mosaic(self) -> tuple:
        """One mosaic per quarter-turn, as the worker emits them, valued by orientation."""
        return tuple(np.full((8, 8, 3), float(k), dtype=np.float32) for k in range(4))

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

    def test_printing_says_so_and_ticks_the_progress_bar(self):
        """36 renders take a few seconds — the HUD has to show it is working."""
        total = len(strip_cells())
        self.controller.toggle_test_strip()
        self.assertIn("Printing test strip…", self.toasts)
        self.assertEqual(self.progress, [(0, total)])

        self.controller.on_strip_progress(12, total)
        self.controller.on_strip_progress(total, total)
        self.assertEqual(self.progress[-2:], [(12, total), (total, total)])

        self.controller.on_strip_finished(self._mosaic(), None)
        self.assertTrue(any("ready" in msg for msg in self.toasts))

    def test_cancelling_mid_print_hides_the_progress_bar(self):
        self.controller.toggle_test_strip()
        self.controller.toggle_test_strip()
        self.assertEqual(self.progress[-1], (0, 0))  # total <= 0 hides it

        # Late progress from the cancelled job must not resurrect the bar.
        self.progress.clear()
        self.controller.on_strip_progress(30, len(strip_cells()))
        self.assertEqual(self.progress, [])

    def test_reprinting_an_unchanged_strip_is_a_cache_hit(self):
        self._print_strip()
        self.controller.toggle_test_strip()  # off
        self.controller.toggle_test_strip()  # on again

        self.assertEqual(len(self.strip_tasks), 1, "no second render job")
        self.assertTrue(self.controller.state.test_strip)
        self.assertIsNotNone(self.controller.state.test_strip_mosaic)
        self.assertEqual(self.controller.state.test_strip_content_rect, (0, 0, 8, 8))

    def test_picking_a_patch_leaves_the_cache_valid(self):
        """The reason this cache is worth having: the mosaic varies density and grade
        itself, so it is invariant to them — and a pick changes nothing else."""
        self._print_strip()
        self.controller.apply_test_strip_pick(0, 3)
        self.controller.state.config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertNotEqual(self.controller.state.config.exposure.density, 1.0)

        self.controller._is_rendering = False
        self.controller.toggle_test_strip()

        self.assertEqual(len(self.strip_tasks), 1, "refining after a pick must not re-render")
        self.assertTrue(self.controller.state.test_strip)

    def test_any_other_edit_invalidates_the_cache(self):
        from negpy.features.geometry.models import GeometryConfig

        self._print_strip()
        self.controller.toggle_test_strip()  # off
        # A crop changes the pixels under every patch, so the strip must be re-printed.
        self.controller.state.config = replace(self.controller.state.config, geometry=replace(GeometryConfig(), fine_rotation=3.0))
        self.controller.toggle_test_strip()

        self.assertEqual(len(self.strip_tasks), 2)
        self.assertTrue(self.controller.state.test_strip_pending)

    def test_a_cache_hit_shows_the_strip_without_a_progress_bar(self):
        self._print_strip()
        self.controller.toggle_test_strip()
        self.progress.clear()
        self.controller.toggle_test_strip()

        self.assertEqual(self.progress, [], "nothing to report progress on")
        self.assertFalse(self.controller.state.test_strip_pending)

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

    def test_rotation_is_declined_when_no_proof_is_up_so_the_image_still_turns(self):
        self.assertFalse(self.controller.rotate_test_strip(1))
        self.assertEqual(self.controller.state.test_strip_rotation, 0)
        self.assertEqual(self.strip_tasks, [])

    def test_rotating_a_printed_proof_swaps_orientation_without_re_rendering(self):
        """The print assembled all four, so a turn is a swap rather than 25 fresh renders."""
        self._print_strip()
        self.assertTrue(self.controller.rotate_test_strip(1))

        self.assertEqual(self.controller.state.test_strip_rotation, 1)
        self.assertEqual(len(self.strip_tasks), 1, "no reprint")
        self.assertFalse(self.controller.state.test_strip_pending)
        self.assertIs(self.controller.state.test_strip_mosaic, self.controller.state.test_strip_mosaics[1])
        self.assertEqual(self.announced[-1], True)
        # The ladder itself goes to the worker unrotated — every orientation comes off one print.
        self.assertEqual(list(self.strip_tasks[0].overrides), strip_overrides())
        self.assertEqual(self.strip_tasks[0].grid, STRIP_GRID)

    def test_rotation_wraps_at_a_full_turn(self):
        self._print_strip()
        for expected in (3, 2, 1, 0):
            self.controller.rotate_test_strip(-1)
            self.assertEqual(self.controller.state.test_strip_rotation, expected)
        self.assertIs(self.controller.state.test_strip_mosaic, self.controller.state.test_strip_mosaics[0])

    def test_a_proof_that_is_only_expected_cannot_swallow_the_rotate_controls(self):
        """Consuming on `pending` left `[` / `]` dead for the session if a print never landed:
        nothing on screen to turn, and the image no longer turning either."""
        self.controller.toggle_test_strip()
        self.assertTrue(self.controller.state.test_strip_pending)
        self.assertFalse(self.controller.rotate_test_strip(1))
        self.assertEqual(self.controller.state.test_strip_rotation, 0)

    def test_a_failed_print_stops_waiting_for_a_proof_that_will_never_land(self):
        self.controller.toggle_test_strip()
        self.controller._on_strip_error("boom")

        self.assertFalse(self.controller.state.test_strip_pending)
        self.assertEqual(self.progress[-1], (0, 0))  # total <= 0 hides the bar

    def test_a_failed_render_elsewhere_leaves_a_printed_proof_alone(self):
        self._print_strip()
        self.controller._on_strip_error("boom")
        self.assertTrue(self.controller.state.test_strip)

    def test_picking_under_a_rotated_ladder_commits_the_patch_that_was_clicked(self):
        self._print_strip()
        self.controller.state.test_strip_rotation = 1
        self.controller.apply_test_strip_pick(0, 3)

        # A quarter-turn CCW puts the grade ladder on the columns and the densest rung on row 0.
        exposure = self.mock_session_manager.update_config.call_args.args[0].exposure
        self.assertEqual(exposure.density, STRIP_DENSITIES[-1])
        self.assertEqual(exposure.grade, STRIP_GRADES[3])

    def test_the_chosen_orientation_outlives_the_proof_it_was_set_on(self):
        self._print_strip()
        self.controller.rotate_test_strip(1)
        self.controller.toggle_test_strip(force=False)
        self.assertEqual(self.controller.state.test_strip_rotation, 1)

    def test_one_cache_entry_serves_every_orientation(self):
        """RenderMemo holds one entry per file hash, so folding rotation into the key would make
        turning the ladder and turning it back two full reprints."""
        keys = set()
        for rotation in range(4):
            self.controller.state.test_strip_rotation = rotation
            keys.add(self.controller._strip_memo_key())
        self.assertEqual(len(keys), 1)
        self.controller.state.test_strip_rotation = 0

        self._print_strip()
        self.controller.rotate_test_strip(1)
        self.controller.toggle_test_strip(force=False)
        self.controller.toggle_test_strip()
        self.assertEqual(len(self.strip_tasks), 1, "a rotated proof must still be a cache hit")
        self.assertIs(self.controller.state.test_strip_mosaic, self.controller.state.test_strip_mosaics[1])

    def test_the_grain_focuser_toggles_without_rendering_anything(self):
        # It magnifies the frame the canvas already holds, so unlike the strip it must never
        # dispatch a job, and a render must not clear it.
        announced: list = []
        self.controller.grain_focuser_changed.connect(announced.append)

        self.controller.toggle_grain_focuser()
        self.assertTrue(self.controller.state.grain_focuser)
        self.controller.toggle_grain_focuser()
        self.assertFalse(self.controller.state.grain_focuser)

        self.controller.toggle_grain_focuser(force=True)
        self.controller.toggle_grain_focuser(force=True)  # idempotent
        self.assertTrue(self.controller.state.grain_focuser)
        self.assertEqual(announced, [True, False, True, True])
        self.assertEqual(self.strip_tasks, [])

        self.controller.request_render()
        self.assertTrue(self.controller.state.grain_focuser)


if __name__ == "__main__":
    unittest.main()
