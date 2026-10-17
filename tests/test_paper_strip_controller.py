"""Paper test strip lifecycle: print → show one paper full frame → keep, sharing one proof slot
with the test strip and the ring-around.

Each band is a whole-frame render on one print stock, so a band opens full frame without a
re-render. The memo pins the paper: keeping one leaves the cache valid.
"""

import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.papers import profiles_for_mode
from negpy.features.process.models import ProcessMode

C41_KEYS = [k for k, _ in profiles_for_mode(ProcessMode.C41)]


class PaperStripWorker(unittest.TestCase):
    def _build(self, keep_tiles: bool) -> list:
        with patch("negpy.desktop.workers.render.ImageProcessor") as MockIP:
            from negpy.desktop.workers.render import RenderWorker, TestStripTask

            calls: list = []

            def fake_pipeline(_buf, config, *a, **k):
                calls.append(config.exposure.paper_profile)
                return np.full((6, 6, 3), float(len(calls) - 1), np.float32), {"content_rect": (0, 0, 6, 6)}

            MockIP.return_value.run_pipeline.side_effect = fake_pipeline
            worker = RenderWorker()
            done: list = []
            worker.strip_finished.connect(lambda m, r, t: done.append((m, r, t)))
            worker.build_strip(
                TestStripTask(
                    buffer=np.zeros((6, 6, 3), np.float32),
                    config=WorkspaceConfig(),
                    source_hash="f1",
                    preview_size=512.0,
                    overrides=tuple({"paper_profile": k} for k in C41_KEYS[:3]),
                    grid=(1, 3),
                    keep_tiles=keep_tiles,
                )
            )
        self.assertEqual(calls, C41_KEYS[:3])
        return done

    def test_each_band_keeps_its_whole_frame_render(self):
        mosaics, _rect, tiles = self._build(keep_tiles=True)[0]
        self.assertEqual(len(mosaics), 4)
        self.assertEqual([float(t[0, 0, 0]) for t in tiles], [0.0, 1.0, 2.0])
        self.assertEqual(tiles[2].shape, (6, 6, 3))

    def test_a_tone_strip_keeps_no_renders(self):
        _mosaics, _rect, tiles = self._build(keep_tiles=False)[0]
        self.assertIsNone(tiles)


class PaperStripLifecycle(unittest.TestCase):
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
        self.controller.state.current_file_hash = "f1"
        self.strip_tasks: list = []
        self.announced: list = []
        self.controller.strip_requested.connect(self.strip_tasks.append)
        self.controller.test_strip_changed.connect(self.announced.append)
        self.mock_session_manager.update_config.side_effect = self._commit

    def _commit(self, config, persist=False, **kwargs):
        self.controller.state.config = config

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.prefetch_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _set_process(self, **fields) -> None:
        cfg = self.controller.state.config
        self.controller.state.config = replace(cfg, process=replace(cfg.process, **fields))

    def _print(self) -> tuple:
        self.controller.toggle_paper_strip()
        n = len(self.strip_tasks[-1].overrides)
        tiles = tuple(np.full((8, 8, 3), float(i), np.float32) for i in range(n))
        mosaics = tuple(np.zeros((8, 8, 3), np.float32) for _ in range(4))
        self.controller.on_strip_finished(mosaics, (0, 0, 8, 8), tiles)
        return tiles

    def test_printing_renders_every_print_stock_of_the_film_mode(self):
        for mode in (ProcessMode.C41, ProcessMode.BW):
            self.strip_tasks.clear()
            self.controller.toggle_paper_strip(force=False)
            self._set_process(process_mode=mode)
            self.controller.toggle_paper_strip()

            keys = [k for k, _ in profiles_for_mode(mode)]
            task = self.strip_tasks[0]
            self.assertEqual(list(task.overrides), [{"paper_profile": k} for k in keys])
            self.assertEqual(task.grid, (1, len(keys)))
            self.assertTrue(task.keep_tiles)
            self.assertEqual(self.controller.state.test_strip_kind, "paper")

    def test_a_slide_or_a_positive_prints_no_paper_strip(self):
        self._set_process(process_mode=ProcessMode.E6)
        self.controller.toggle_paper_strip()
        self._set_process(process_mode=ProcessMode.E6, positive_source=True)
        self.controller.toggle_paper_strip()
        self.assertEqual(self.strip_tasks, [])

    def test_a_click_on_a_band_shows_that_paper_and_keeps_nothing(self):
        self._print()
        self.controller.apply_test_strip_pick(0, 3)

        self.assertEqual(self.controller.state.test_strip_expanded, 3)
        self.assertTrue(self.controller.state.test_strip)
        self.mock_session_manager.update_config.assert_not_called()
        self.assertEqual(self.announced[-1], True)

    def test_a_click_on_a_turned_strip_finds_its_band(self):
        """A quarter-turn CCW stacks the bands with the last paper on top."""
        self._print()
        self.controller.rotate_test_strip(1)
        self.controller.apply_test_strip_pick(0, 0)
        self.assertEqual(self.controller.state.test_strip_expanded, len(C41_KEYS) - 1)

    def test_keep_commits_only_the_shown_paper_and_drops_the_proof(self):
        cfg = self.controller.state.config
        self.controller.state.config = replace(cfg, exposure=replace(cfg.exposure, toe=0.3, density=1.2))
        self._print()
        self.controller.show_paper_strip(2)
        self.controller.keep_paper_strip()

        committed = self.mock_session_manager.update_config.call_args
        self.assertTrue(committed.kwargs["persist"])
        exposure = self.controller.state.config.exposure
        self.assertEqual(exposure.paper_profile, C41_KEYS[2])
        self.assertEqual((exposure.toe, exposure.density), (0.3, 1.2))
        self.assertFalse(self.controller.state.test_strip)
        self.assertIsNone(self.controller.state.test_strip_expanded)
        self.assertIsNone(self.controller.state.test_strip_tiles)

    def test_keep_on_the_strip_itself_commits_nothing(self):
        self._print()
        self.controller.keep_paper_strip()
        self.mock_session_manager.update_config.assert_not_called()
        self.assertTrue(self.controller.state.test_strip)

    def test_the_arrows_step_from_the_current_paper_and_stop_at_the_ends(self):
        cfg = self.controller.state.config
        self.controller.state.config = replace(cfg, exposure=replace(cfg.exposure, paper_profile=C41_KEYS[1]))
        self._print()

        self.assertTrue(self.controller.step_paper_strip(1))
        self.assertEqual(self.controller.state.test_strip_expanded, 2)
        self.assertTrue(self.controller.step_paper_strip(-1))
        self.assertTrue(self.controller.step_paper_strip(-1))
        self.assertTrue(self.controller.step_paper_strip(-1))
        self.assertEqual(self.controller.state.test_strip_expanded, 0)

        self.controller.show_paper_strip(len(C41_KEYS) - 1)
        self.assertTrue(self.controller.step_paper_strip(1))
        self.assertEqual(self.controller.state.test_strip_expanded, len(C41_KEYS) - 1)

    def test_the_arrows_pass_through_without_a_paper_strip(self):
        self.assertFalse(self.controller.step_paper_strip(1))
        self.controller.toggle_test_strip()
        self.controller.on_strip_finished(tuple(np.zeros((8, 8, 3), np.float32) for _ in range(4)), None)
        self.assertFalse(self.controller.step_paper_strip(1))
        self.assertIsNone(self.controller.state.test_strip_expanded)

    def test_back_to_the_strip_keeps_the_proof_up(self):
        self._print()
        self.controller.show_paper_strip(4)
        self.controller.show_paper_strip(None)
        self.assertIsNone(self.controller.state.test_strip_expanded)
        self.assertTrue(self.controller.state.test_strip)

    def test_reprinting_after_a_keep_is_a_cache_hit(self):
        tiles = self._print()
        self.controller.show_paper_strip(3)
        self.controller.keep_paper_strip()
        self.assertEqual(self.controller.state.config.exposure.paper_profile, C41_KEYS[3])

        self.controller._is_rendering = False
        self.controller.toggle_paper_strip()

        self.assertEqual(len(self.strip_tasks), 1, "keeping a paper must not force a reprint")
        self.assertTrue(self.controller.state.test_strip)
        self.assertIs(self.controller.state.test_strip_tiles, tiles)
        self.assertIsNone(self.controller.state.test_strip_expanded)

    def test_another_proof_kind_drops_the_shown_paper(self):
        self._print()
        self.controller.set_paper_strip_layout("grid")
        self.controller.show_paper_strip(2)
        self.controller.toggle_test_strip()
        self.assertIsNone(self.controller.state.test_strip_expanded)
        self.assertIsNone(self.controller.state.test_strip_tiles)
        self.assertIsNone(self.controller.state.test_strip_grid)

    def test_the_grid_shows_every_paper_whole_and_a_click_finds_its_cell(self):
        tiles = self._print()
        self.controller.set_paper_strip_layout("grid")

        self.assertEqual(self.controller.state.test_strip_layout, "grid")
        self.assertEqual(self.controller.state.test_strip_grid.shape, tiles[0].shape)
        self.assertEqual(self.announced[-1], True)
        self.controller.apply_test_strip_pick(1, 2)
        self.assertEqual(self.controller.state.test_strip_expanded, 5)

    def test_a_click_on_an_empty_grid_cell_shows_nothing(self):
        self._set_process(process_mode=ProcessMode.BW)
        self._print()
        self.controller.set_paper_strip_layout("grid")
        self.controller.apply_test_strip_pick(2, 2)
        self.assertIsNone(self.controller.state.test_strip_expanded)

    def test_the_grid_sticks_across_a_reprint(self):
        self._print()
        self.controller.set_paper_strip_layout("grid")
        self.controller.toggle_paper_strip()
        self.assertIsNone(self.controller.state.test_strip_grid)
        self.controller.toggle_paper_strip()

        self.assertEqual(self.controller.state.test_strip_layout, "grid")
        self.assertIsNotNone(self.controller.state.test_strip_grid)


def test_esc_on_a_paper_shown_full_frame_goes_back_to_the_strip():
    from negpy.desktop.view.keyboard_shortcuts import _context_cancel

    controller, window = MagicMock(), MagicMock()
    window.light_table_active.return_value = False
    controller.state.test_strip = True
    controller.state.test_strip_expanded = 2

    _context_cancel(controller, window)

    controller.show_paper_strip.assert_called_once_with(None)
    controller.toggle_test_strip.assert_not_called()


def test_the_file_keys_step_papers_while_the_paper_strip_is_up():
    from negpy.desktop.view.keyboard_shortcuts import _step_file

    controller = MagicMock()
    controller.step_paper_strip.return_value = True
    _step_file(controller, 1)
    controller.step_paper_strip.assert_called_once_with(1)
    controller.session.next_file.assert_not_called()

    controller.step_paper_strip.return_value = False
    _step_file(controller, -1)
    _step_file(controller, 1)
    controller.session.prev_file.assert_called_once()
    controller.session.next_file.assert_called_once()
