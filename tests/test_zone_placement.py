import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.features.exposure.analysis import encoded_of_zone, zone_of_encoded
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.placement import (
    DENSITY_RANGE,
    ZonePin,
    predicted_zone,
    solve_placement,
)

METRICS = {"norm_density_range": 1.4}


def _pin(val: float, target: float) -> ZonePin:
    return ZonePin(nx=0.5, ny=0.5, val_rgb=(val, val, val), val_luma=val, target_zone=target)


class TestZoneRuler(unittest.TestCase):
    def test_encoded_zone_roundtrip(self):
        enc = np.linspace(0.0, 1.0, 21)
        for e in enc:
            self.assertAlmostEqual(encoded_of_zone(float(zone_of_encoded(e))), float(e), places=6)
        for z in np.linspace(0.0, 10.0, 31):
            self.assertAlmostEqual(float(zone_of_encoded(encoded_of_zone(float(z)))), float(z), places=6)


class TestPlacementSolver(unittest.TestCase):
    def setUp(self):
        self.exposure = ExposureConfig()

    def test_one_pin_lands_on_target(self):
        pin = _pin(0.46, 5.0)
        sol = solve_placement(self.exposure, None, METRICS, [pin])
        self.assertIsNotNone(sol)
        self.assertFalse(sol.clamped)
        self.assertGreaterEqual(sol.fields["density"], DENSITY_RANGE[0])
        self.assertLessEqual(sol.fields["density"], DENSITY_RANGE[1])
        self.assertAlmostEqual(sol.achieved[0], 5.0, delta=0.05)

    def test_one_pin_fields(self):
        sol = solve_placement(self.exposure, None, METRICS, [_pin(0.46, 5.0)])
        self.assertEqual(set(sol.fields), {"density", "auto_exposure"})
        self.assertFalse(sol.fields["auto_exposure"])

    def test_two_pins_land_on_both_targets(self):
        pins = [_pin(0.65, 3.0), _pin(0.30, 8.0)]
        sol = solve_placement(self.exposure, None, METRICS, pins)
        self.assertIsNotNone(sol)
        self.assertEqual(set(sol.fields), {"density", "grade", "auto_exposure", "auto_normalize_contrast"})
        self.assertFalse(sol.fields["auto_exposure"])
        self.assertFalse(sol.fields["auto_normalize_contrast"])
        self.assertEqual(sol.fields["grade"], round(sol.fields["grade"]))
        self.assertGreaterEqual(sol.fields["grade"], EXPOSURE_CONSTANTS["iso_r_min"])
        self.assertLessEqual(sol.fields["grade"], EXPOSURE_CONSTANTS["iso_r_max"])
        self.assertAlmostEqual(sol.achieved[0], 3.0, delta=0.15)
        self.assertAlmostEqual(sol.achieved[1], 8.0, delta=0.15)

    def test_unreachable_target_clamps(self):
        # Zone 0 needs paper density beyond d_max: unreachable at any exposure.
        sol = solve_placement(self.exposure, None, METRICS, [_pin(0.2, 0.0)])
        self.assertTrue(sol.clamped)
        self.assertEqual(sol.fields["density"], DENSITY_RANGE[1])
        self.assertGreater(sol.achieved[0], 0.3)

    def test_split_grade_still_converges(self):
        exposure = replace(self.exposure, shadow_grade=20.0, highlight_grade=-15.0)
        pins = [_pin(0.65, 3.0), _pin(0.30, 8.0)]
        sol = solve_placement(exposure, None, METRICS, pins)
        self.assertIsNotNone(sol)
        self.assertAlmostEqual(sol.achieved[0], 3.0, delta=0.15)
        self.assertAlmostEqual(sol.achieved[1], 8.0, delta=0.15)

    def test_degenerate_two_pins_returns_none(self):
        sol = solve_placement(self.exposure, None, METRICS, [_pin(0.5, 3.0), _pin(0.5, 8.0)])
        self.assertIsNone(sol)

    def test_achieved_recomputed_at_rounded_values(self):
        pins = [_pin(0.65, 3.0), _pin(0.30, 8.0)]
        sol = solve_placement(self.exposure, None, METRICS, pins)
        self.assertEqual(sol.fields["density"], round(sol.fields["density"], 2))
        applied = replace(self.exposure, **sol.fields)
        for i, pin in enumerate(pins):
            self.assertAlmostEqual(sol.achieved[i], predicted_zone(applied, None, METRICS, pin.val_luma), places=6)


class TestZonePlacementLifecycle(unittest.TestCase):
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
        self.controller.canvas = MagicMock()
        self.controller.canvas.display_size.return_value = (8, 8)
        # Vertical ramp: a click's ny picks its normalized-log value, so two pins
        # at different heights read two different tones.
        ramp = np.tile(np.linspace(0.15, 0.85, 8, dtype=np.float32).reshape(8, 1, 1), (1, 8, 3))
        self.controller.state.last_metrics["normalized_log"] = ramp
        self.render_tasks: list = []
        self.controller.render_requested.connect(self.render_tasks.append)

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

    def _enter(self) -> None:
        self.controller.toggle_zone_placement()

    def _pin(self, nx: float, ny: float) -> None:
        self.controller.handle_canvas_clicked(nx, ny)

    def test_click_with_tool_active_pins_a_spot(self):
        self._enter()
        self._pin(0.5, 0.2)
        pins = self.controller.state.zone_pins
        self.assertEqual(len(pins), 1)
        # Default target is the measured zone rounded to the nearest third.
        self.assertAlmostEqual(pins[0].target_zone * 3.0, round(pins[0].target_zone * 3.0))

    def test_clicks_without_the_tool_do_not_pin(self):
        self._pin(0.5, 0.2)
        self.assertEqual(self.controller.state.zone_pins, [])

    def test_third_click_replaces_the_nearest_pin(self):
        self._enter()
        self._pin(0.2, 0.2)
        self._pin(0.8, 0.8)
        self._pin(0.3, 0.3)
        pins = self.controller.state.zone_pins
        self.assertEqual(len(pins), 2)
        self.assertEqual(sorted(round(p.nx, 1) for p in pins), [0.3, 0.8])

    def test_apply_commits_and_flips_the_right_autos(self):
        self._enter()
        self._pin(0.5, 0.2)
        self.controller.apply_zone_placement()
        committed = self.mock_session_manager.update_config.call_args
        exposure = committed.args[0].exposure
        self.assertTrue(committed.kwargs["persist"])
        self.assertFalse(exposure.auto_exposure)
        self.assertTrue(exposure.auto_normalize_contrast, "one pin must not touch Auto Grade")
        self.assertGreaterEqual(exposure.density, 0.0)
        self.assertLessEqual(exposure.density, 2.0)

    def test_two_pin_apply_also_writes_grade(self):
        self._enter()
        self._pin(0.5, 0.2)
        self._pin(0.5, 0.8)
        self.controller.apply_zone_placement()
        exposure = self.mock_session_manager.update_config.call_args.args[0].exposure
        self.assertFalse(exposure.auto_exposure)
        self.assertFalse(exposure.auto_normalize_contrast)
        self.assertEqual(exposure.grade, round(exposure.grade))

    def test_apply_keeps_pins_and_the_next_real_render_drops_them(self):
        self._enter()
        self._pin(0.5, 0.2)
        self.controller.apply_zone_placement()
        self.assertEqual(len(self.controller.state.zone_pins), 1)
        self.controller._is_rendering = False
        self.controller.request_render()
        self.assertEqual(self.controller.state.zone_pins, [])

    def test_an_override_render_leaves_the_pins_alone(self):
        from negpy.domain.models import WorkspaceConfig

        self._enter()
        self._pin(0.5, 0.2)
        self.controller._is_rendering = False
        self.controller.request_render(readback_metrics=False, config_override=WorkspaceConfig())
        self.assertEqual(len(self.controller.state.zone_pins), 1)

    def test_loading_another_frame_drops_the_pins(self):
        self._enter()
        self._pin(0.5, 0.2)
        with patch.object(self.controller, "_file_hash_for_path", return_value=None):
            self.controller.load_file("/nowhere/other.raw")
        self.assertEqual(self.controller.state.zone_pins, [])

    def test_leaving_the_tool_drops_the_pins(self):
        from negpy.desktop.session import ToolMode

        self._enter()
        self._pin(0.5, 0.2)
        self.controller.set_active_tool(ToolMode.NONE)
        self.assertEqual(self.controller.state.zone_pins, [])

    def test_target_change_previews_without_committing(self):
        self._enter()
        self._pin(0.5, 0.2)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.set_zone_pin_target(0, 3.0)
        self.assertEqual(len(self.render_tasks), n + 1)
        previewed = self.render_tasks[-1]
        self.assertFalse(previewed.config.exposure.auto_exposure)
        self.assertFalse(previewed.readback_metrics)
        self.mock_session_manager.update_config.assert_not_called()
        self.assertEqual(self.controller.state.zone_pins[0].target_zone, 3.0)

    def test_clear_restores_the_committed_render_after_a_preview(self):
        self._enter()
        self._pin(0.5, 0.2)
        self.controller._is_rendering = False
        self.controller.set_zone_pin_target(0, 3.0)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.clear_zone_pins()
        self.assertEqual(self.controller.state.zone_pins, [])
        self.assertEqual(len(self.render_tasks), n + 1)
        self.assertIs(self.render_tasks[-1].config, self.controller.state.config)

    def test_entering_the_tool_takes_the_canvas_from_the_strip(self):
        self.controller.toggle_test_strip()
        mosaics = tuple(np.zeros((8, 8, 3), np.float32) for _ in range(4))
        self.controller.on_strip_finished(mosaics, (0, 0, 8, 8))
        self.assertTrue(self.controller.state.test_strip)
        self._enter()
        self.assertFalse(self.controller.state.test_strip)

    def test_apply_with_no_pins_commits_nothing(self):
        self._enter()
        self.controller.apply_zone_placement()
        self.mock_session_manager.update_config.assert_not_called()

    def test_dragging_a_pin_rereads_the_tone_under_it(self):
        self._enter()
        self._pin(0.5, 0.15)
        before = self.controller.state.zone_pins[0]
        self.controller.move_zone_pin(0, 0.5, 0.85)
        pin = self.controller.state.zone_pins[0]
        self.assertAlmostEqual(pin.ny, 0.85)
        self.assertNotAlmostEqual(pin.val_luma, before.val_luma)
        self.assertNotEqual(pin.label, "")

    def test_dragging_an_untargeted_pin_follows_the_new_reading(self):
        self._enter()
        self._pin(0.5, 0.15)
        self.controller.move_zone_pin(0, 0.5, 0.85)
        pin = self.controller.state.zone_pins[0]
        measured = self.controller._pin_zone(pin.val_luma)
        self.assertAlmostEqual(pin.target_zone, round(measured * 3.0) / 3.0)

    def test_dragging_a_retargeted_pin_keeps_its_target(self):
        self._enter()
        self._pin(0.5, 0.15)
        self.controller.set_zone_pin_target(0, 3.0)
        self.controller.move_zone_pin(0, 0.5, 0.85)
        self.assertEqual(self.controller.state.zone_pins[0].target_zone, 3.0)

    def test_the_preview_refreshes_when_the_drag_ends_not_during_it(self):
        self._enter()
        self._pin(0.5, 0.15)
        self.controller._is_rendering = False
        self.controller.set_zone_pin_target(0, 3.0)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.move_zone_pin(0, 0.5, 0.5)
        self.assertEqual(len(self.render_tasks), n)
        self.controller.move_zone_pin(0, 0.5, 0.85, final=True)
        self.assertEqual(len(self.render_tasks), n + 1)
        self.assertFalse(self.render_tasks[-1].readback_metrics)

    def test_a_drag_without_a_preview_up_renders_nothing(self):
        self._enter()
        self._pin(0.5, 0.15)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.move_zone_pin(0, 0.5, 0.85, final=True)
        self.assertEqual(len(self.render_tasks), n)

    def test_readouts_skip_the_solve_mid_drag(self):
        self._enter()
        self._pin(0.5, 0.15)
        self._pin(0.5, 0.85)
        with patch.object(self.controller, "_solve_zone_placement", wraps=self.controller._solve_zone_placement) as solve:
            self.controller.zone_pin_readouts()
            self.assertEqual(solve.call_count, 1)
            self.controller.move_zone_pin(0, 0.5, 0.5)
            self.controller.zone_pin_readouts()
            self.assertEqual(solve.call_count, 1, "the nested bisection is too slow to run per mouse-move")
            self.controller.move_zone_pin(0, 0.5, 0.45, final=True)
            self.controller.zone_pin_readouts()
            self.assertEqual(solve.call_count, 2)

    def test_degenerate_pins_neither_preview_nor_apply(self):
        self._enter()
        self._pin(0.2, 0.5)
        self._pin(0.8, 0.5)  # same ramp row -> same tone
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.set_zone_pin_target(0, 3.0)
        self.assertEqual(len(self.render_tasks), n)
        self.controller.apply_zone_placement()
        self.mock_session_manager.update_config.assert_not_called()


class TestZonePlacementRows(unittest.TestCase):
    def _widget(self):
        from negpy.desktop.view.widgets.stats import ZonePlacementRows

        return ZonePlacementRows()

    def test_hidden_when_empty_and_shown_with_pins(self):
        w = self._widget()
        self.assertFalse(w.isVisibleTo(w.parentWidget()) if w.parentWidget() else w.isVisible())
        w.refresh([(0, "IV⅓", 4.33, None, True)])
        self.assertTrue(w._rows[0].isVisibleTo(w))
        self.assertFalse(w._rows[1].isVisibleTo(w))
        w.refresh([])
        self.assertFalse(w.isVisible())

    def test_stepper_emits_absolute_third_step_targets(self):
        w = self._widget()
        w.refresh([(0, "V", 5.0, None, True)])
        emitted: list = []
        w.target_changed.connect(lambda i, z: emitted.append((i, z)))
        w._steppers[0][1].click()  # plus
        w._steppers[0][0].click()  # minus
        self.assertEqual(len(emitted), 2)
        self.assertEqual(emitted[0][0], 0)
        self.assertAlmostEqual(emitted[0][1], 5.0 + 1.0 / 3.0)
        self.assertAlmostEqual(emitted[1][1], 5.0 - 1.0 / 3.0)

    def test_clamped_target_shows_where_it_lands_and_unsolvable_disables_apply(self):
        w = self._widget()
        w.refresh([(0, "II", 0.0, "I", True)])
        self.assertIn("lands I", w._lands[0].text())
        self.assertTrue(w.apply_btn.isEnabled())
        w.refresh([(0, "V", 5.0, None, False), (1, "V", 5.0, None, False)])
        self.assertFalse(w.apply_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
