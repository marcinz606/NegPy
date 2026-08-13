"""Ring-around lifecycle: print → pick → commit, sharing one proof slot with the test strip.

Its ladder is absolute, so the mosaic is invariant to the filtration in force and the memo
pins M/Y: picking a patch leaves the cache valid. Asking for the other kind swaps the proof
rather than dismissing it.
"""

import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.analysis import RING_GRID, STRIP_GRID, ring_cells, ring_overrides, rotate_grid


class RingAroundWorker(unittest.TestCase):
    def test_every_patch_is_rendered_at_its_own_filtration(self):
        with patch("negpy.desktop.workers.render.ImageProcessor") as MockIP:
            from negpy.desktop.workers.render import RenderWorker, TestStripTask

            seen: list = []

            def fake_pipeline(_buf, config, *a, **k):
                e = config.exposure
                seen.append((e.wb_magenta, e.wb_yellow, e.wb_cyan, e.density, e.grade))
                return np.full((10, 10, 3), float(len(seen) - 1), np.float32), {"content_rect": (0, 0, 10, 10)}

            MockIP.return_value.run_pipeline.side_effect = fake_pipeline
            worker = RenderWorker()
            done: list = []
            worker.strip_finished.connect(lambda m, r: done.append((m, r)))

            base = WorkspaceConfig()
            base = replace(base, exposure=replace(base.exposure, wb_magenta=0.2, wb_yellow=-0.1))
            worker.build_strip(
                TestStripTask(
                    buffer=np.zeros((10, 10, 3), np.float32),
                    config=base,
                    source_hash="f1",
                    preview_size=512.0,
                    overrides=tuple(ring_overrides()),
                    grid=RING_GRID,
                )
            )

        expected = [(m, y) for _, _, m, y in ring_cells()]
        self.assertEqual([(m, y) for m, y, _, _, _ in seen], expected)
        # Nothing else moved: cyan, density and grade are the base config's on every patch.
        for _m, _y, cyan, density, grade in seen:
            self.assertEqual(cyan, base.exposure.wb_cyan)
            self.assertEqual(density, base.exposure.density)
            self.assertEqual(grade, base.exposure.grade)

        mosaics, _rect = done[0]
        # Top-left patch from the first render, bottom-right from the last — slicing is right.
        self.assertEqual(mosaics[0][0, 0, 0], 0.0)
        self.assertEqual(mosaics[0][-1, -1, 0], float(RING_GRID[0] * RING_GRID[1] - 1))


class RingAroundLifecycle(unittest.TestCase):
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
        self.controller.strip_requested.connect(self.strip_tasks.append)
        # update_config on a MagicMock session doesn't write through; mirror it so the
        # controller sees the committed config the real session would have stored.
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
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _mosaic(self) -> tuple:
        """One mosaic per quarter-turn, as the worker emits them."""
        return tuple(np.full((10, 10, 3), float(k), dtype=np.float32) for k in range(4))

    def _print_ring(self) -> None:
        self.controller.toggle_ring_around()
        self.controller.on_strip_finished(self._mosaic(), (0, 0, 10, 10))

    def _set_filtration(self, magenta: float, yellow: float) -> None:
        cfg = self.controller.state.config
        self.controller.state.config = replace(cfg, exposure=replace(cfg.exposure, wb_magenta=magenta, wb_yellow=yellow))

    def test_toggling_on_dispatches_a_full_grid_job(self):
        self._set_filtration(0.2, -0.1)
        self.controller.toggle_ring_around()

        self.assertEqual(len(self.strip_tasks), 1)
        task = self.strip_tasks[0]
        self.assertEqual(task.grid, RING_GRID)
        self.assertEqual(len(task.overrides), RING_GRID[0] * RING_GRID[1])
        self.assertEqual(self.controller.state.test_strip_kind, "color")
        # Absolute ladder: the rungs don't depend on what is currently dialled in.
        self.assertEqual(tuple(task.overrides), tuple(ring_overrides()))

    def test_picking_a_patch_commits_only_its_filtration(self):
        self._set_filtration(0.2, -0.1)
        self._print_ring()
        before = self.controller.state.config.exposure

        self.controller.apply_test_strip_pick(0, 0)

        after = self.controller.state.config.exposure
        _, _, m, y = ring_cells()[0]
        self.assertAlmostEqual(after.wb_magenta, m)
        self.assertAlmostEqual(after.wb_yellow, y)
        # Everything the patches were rendered under is left exactly as it was.
        self.assertEqual(after.wb_cyan, before.wb_cyan)
        self.assertEqual(after.density, before.density)
        self.assertEqual(after.grade, before.grade)
        self.assertEqual(after.cast_removal_strength, before.cast_removal_strength)
        self.assertEqual(after.auto_exposure, before.auto_exposure)
        self.assertEqual(after.auto_normalize_contrast, before.auto_normalize_contrast)
        self.assertFalse(self.controller.state.test_strip)

    def test_picking_the_centre_commits_neutral_filtration(self):
        self._set_filtration(0.2, -0.1)
        self._print_ring()
        mid = RING_GRID[0] // 2
        self.controller.apply_test_strip_pick(mid, mid)

        after = self.controller.state.config.exposure
        self.assertEqual((after.wb_magenta, after.wb_yellow), (0.0, 0.0))
        self.assertFalse(self.controller.state.test_strip)

    def test_reprinting_an_unchanged_ring_is_a_cache_hit(self):
        """Same deal as the tone strip: 25 renders for pixels already in hand is worth
        avoiding, so toggling the ring off and back on with no edit between must not reprint."""
        self._print_ring()
        self.controller._clear_test_strip()
        self.controller.toggle_ring_around()

        self.assertEqual(len(self.strip_tasks), 1)
        self.assertTrue(self.controller.state.test_strip)
        self.assertIsNotNone(self.controller.state.test_strip_mosaic)

    def test_any_other_edit_invalidates_the_ring_cache(self):
        self._print_ring()
        self.controller._clear_test_strip()
        cfg = self.controller.state.config
        self.controller.state.config = replace(cfg, exposure=replace(cfg.exposure, density=1.4))
        self.controller.toggle_ring_around()
        self.assertEqual(len(self.strip_tasks), 2)

    def test_picking_a_patch_leaves_the_ring_cache_valid(self):
        """What the absolute ladder buys: the mosaic varies M/Y itself, so it is invariant to
        them and picking a patch changes nothing the patches depended on."""
        self._print_ring()
        self.controller.apply_test_strip_pick(0, 0)
        self.controller.toggle_ring_around()
        self.assertEqual(len(self.strip_tasks), 1)  # cache hit, no reprint
        self.assertTrue(self.controller.state.test_strip)

    def test_the_two_proof_kinds_never_share_a_cache_entry(self):
        """RenderMemo holds one entry per file hash, so the kind is folded into the key —
        otherwise printing the ring then the strip would paint the ring's mosaic."""
        self._print_ring()
        self.controller._clear_test_strip()
        self.controller.toggle_test_strip()
        self.assertEqual(len(self.strip_tasks), 2)
        self.assertEqual(self.strip_tasks[1].grid, STRIP_GRID)

    def test_asking_for_the_other_kind_swaps_the_proof(self):
        self._print_ring()
        self.assertEqual(self.controller.state.test_strip_kind, "color")

        self.controller.toggle_test_strip()
        self.assertEqual(self.controller.state.test_strip_kind, "tone")
        self.assertEqual(len(self.strip_tasks), 2)

        self.controller.on_strip_finished(self._mosaic(), (0, 0, 10, 10))
        self.controller.toggle_ring_around()
        self.assertEqual(self.controller.state.test_strip_kind, "color")
        self.assertEqual(len(self.strip_tasks), 3)

    def test_toggling_the_same_kind_again_dismisses_it(self):
        self._print_ring()
        self.controller.toggle_ring_around()
        self.assertFalse(self.controller.state.test_strip)

    def test_a_render_drops_the_ring(self):
        self._print_ring()
        self.controller.request_render()
        self.assertFalse(self.controller.state.test_strip)
        self.assertIsNone(self.controller.state.test_strip_mosaic)

    def test_rotating_the_ring_turns_its_axes_and_the_pick_follows(self):
        """The magenta axis is the rows unrotated; a quarter-turn puts it on the columns, and
        a click has to read the cell that is actually there."""
        self._print_ring()
        self.assertTrue(self.controller.rotate_test_strip(1))
        self.assertEqual(len(self.strip_tasks), 1, "the ring's orientations come off one print too")
        self.assertIs(self.controller.state.test_strip_mosaic, self.controller.state.test_strip_mosaics[1])

        self.controller.apply_test_strip_pick(0, 0)
        after = self.controller.state.config.exposure
        _, _, m, y = rotate_grid(ring_cells(), RING_GRID, 1)[0]
        self.assertAlmostEqual(after.wb_magenta, m)
        self.assertAlmostEqual(after.wb_yellow, y)

    def test_compare_and_flat_peek_take_the_canvas_from_the_ring(self):
        for enter_mode in (self.controller.toggle_compare, lambda: self.controller.toggle_flat_peek(force=True)):
            self.controller.state.compare_mode = False
            self.controller.state.flat_peek = False
            self.controller._is_rendering = False
            self._print_ring()

            enter_mode()
            self.assertFalse(self.controller.state.test_strip)
            self.assertIsNone(self.controller.state.test_strip_mosaic)


if __name__ == "__main__":
    unittest.main()
