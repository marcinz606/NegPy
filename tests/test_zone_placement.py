import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.features.exposure.analysis import encoded_of_zone, zone_of_encoded
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.placement import (
    DENSITY_RANGE,
    KNEE_CANDIDATES,
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


class TestKneeSolve(unittest.TestCase):
    """Third pin: Density + Grade from the extremes, plus one knee control for the middle."""

    def setUp(self):
        self.exposure = ExposureConfig()
        self.extremes = [_pin(0.65, 3.0), _pin(0.30, 8.0)]

    def _two_pin_zone_of(self, val: float) -> float:
        """Where `val` lands once the two extremes are placed — the zone a third pin
        starts from, so a test can ask for something a known distance away."""
        two = solve_placement(self.exposure, None, METRICS, self.extremes)
        return predicted_zone(replace(self.exposure, **two.fields), None, METRICS, val)

    def test_three_pins_solve_one_knee_control(self):
        mid = _pin(0.48, self._two_pin_zone_of(0.48) + 0.4)
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, mid])
        self.assertIsNotNone(sol)
        self.assertIn(sol.knee, [c.field for c in KNEE_CANDIDATES])
        self.assertEqual(
            set(sol.fields),
            {"density", "grade", sol.knee, "auto_exposure", "auto_normalize_contrast"},
        )

    def test_solved_knee_stays_inside_its_slider(self):
        mid = _pin(0.48, self._two_pin_zone_of(0.48) + 0.4)
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, mid])
        candidate = next(c for c in KNEE_CANDIDATES if c.field == sol.knee)
        self.assertGreaterEqual(sol.fields[sol.knee], candidate.lo)
        self.assertLessEqual(sol.fields[sol.knee], candidate.hi)

    def test_third_pin_gets_closer_than_a_two_pin_solve_leaves_it(self):
        where = self._two_pin_zone_of(0.48)
        ask = where + 0.7
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, _pin(0.48, ask)])
        self.assertLess(abs(sol.achieved[2] - ask), abs(where - ask))

    def test_the_extremes_still_land_with_a_third_pin_in_play(self):
        mid = _pin(0.48, self._two_pin_zone_of(0.48) + 0.4)
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, mid])
        self.assertAlmostEqual(sol.achieved[0], 3.0, delta=0.2)
        self.assertAlmostEqual(sol.achieved[1], 8.0, delta=0.2)

    def test_a_knee_pin_either_side_of_its_centre_both_converge(self):
        for val in (0.72, 0.36):
            with self.subTest(val=val):
                ask = self._two_pin_zone_of(val) + 0.3
                sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, _pin(val, ask)])
                self.assertIsNotNone(sol)
                self.assertAlmostEqual(sol.achieved[2], ask, delta=0.35)

    def test_a_third_pin_already_on_target_leaves_the_knee_alone(self):
        mid = _pin(0.48, self._two_pin_zone_of(0.48))
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, mid])
        self.assertEqual(sol.knee, "")
        self.assertEqual(set(sol.fields), {"density", "grade", "auto_exposure", "auto_normalize_contrast"})
        self.assertEqual(len(sol.achieved), 3)

    def test_an_unreachable_third_target_reads_as_clamped(self):
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, _pin(0.48, 0.0)])
        self.assertTrue(sol.clamped)
        self.assertNotAlmostEqual(sol.achieved[2], 0.0, delta=0.2)

    def test_three_pins_sharing_one_tone_are_unsolvable(self):
        pins = [_pin(0.5, 3.0), _pin(0.5, 5.0), _pin(0.5, 8.0)]
        self.assertIsNone(solve_placement(self.exposure, None, METRICS, pins))

    def test_the_solve_is_bounded_by_the_pass_cap(self):
        import negpy.features.exposure.placement as placement

        calls = 0
        real = placement.predicted_zone

        def counted(*args):
            nonlocal calls
            calls += 1
            return real(*args)

        mid = _pin(0.48, self._two_pin_zone_of(0.48) + 0.4)
        with patch.object(placement, "predicted_zone", counted):
            solve_placement(self.exposure, None, METRICS, [*self.extremes, mid])
        # A 2-pin solve is ~1.5k evaluations; the cap keeps three pins within a small
        # multiple of that rather than a third nested bisection's blow-up.
        self.assertLess(calls, 20_000)


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

    def _arm(self, zone: float = 5.0) -> None:
        self.controller.arm_zone_target(zone)

    def _pin(self, nx: float, ny: float) -> None:
        self.controller.handle_canvas_clicked(nx, ny)

    def _place(self, nx: float, ny: float, zone: float = 5.0) -> None:
        self._arm(zone)
        self._pin(nx, ny)

    def test_arming_a_zone_puts_the_tool_up(self):
        from negpy.desktop.session import ToolMode

        self._arm(5.0)
        self.assertEqual(self.controller.state.active_tool, ToolMode.ZONE_PLACE)
        self.assertEqual(self.controller.state.zone_arm_target, 5.0)

    def test_arming_the_same_zone_again_disarms_and_puts_the_tool_down(self):
        from negpy.desktop.session import ToolMode

        self._arm(5.0)
        self._arm(5.0)
        self.assertIsNone(self.controller.state.zone_arm_target)
        self.assertEqual(self.controller.state.active_tool, ToolMode.NONE)

    def test_arming_another_zone_replaces_the_arm(self):
        self._arm(5.0)
        self._arm(3.0)
        self.assertEqual(self.controller.state.zone_arm_target, 3.0)

    def test_an_armed_click_places_the_pin_on_that_zone(self):
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self._place(0.5, 0.2, zone=3.0)
        pins = self.controller.state.zone_pins
        self.assertEqual(len(pins), 1)
        self.assertEqual(pins[0].target_zone, 3.0)
        self.assertTrue(pins[0].retargeted, "an asked-for zone must survive a later drag")
        self.assertIsNone(self.controller.state.zone_arm_target, "placing spends the arm")
        self.assertEqual(len(self.render_tasks), n + 1, "the solve previews at once")
        self.mock_session_manager.update_config.assert_not_called()

    def test_an_unarmed_click_meters_without_touching_the_image(self):
        self._place(0.5, 0.2, zone=3.0)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self._pin(0.5, 0.8)
        pins = self.controller.state.zone_pins
        self.assertEqual(len(pins), 2)
        self.assertFalse(pins[1].retargeted)
        self.assertAlmostEqual(pins[1].target_zone, round(self.controller._pin_zone(pins[1].val_luma) * 3.0) / 3.0)
        self.assertEqual(len(self.render_tasks), n, "a bare read must not move the print")

    def test_clicks_without_the_tool_do_not_pin(self):
        self._pin(0.5, 0.2)
        self.assertEqual(self.controller.state.zone_pins, [])

    def test_a_third_click_pins_a_third_tone(self):
        self._place(0.2, 0.2)
        self._place(0.8, 0.8)
        self._place(0.5, 0.5)
        self.assertEqual(len(self.controller.state.zone_pins), 3)

    def test_fourth_click_replaces_the_nearest_pin(self):
        self._place(0.2, 0.2)
        self._place(0.8, 0.8)
        self._place(0.5, 0.5)
        self._place(0.3, 0.3)
        pins = self.controller.state.zone_pins
        self.assertEqual(len(pins), 3)
        self.assertEqual(sorted(round(p.nx, 1) for p in pins), [0.3, 0.5, 0.8])

    def test_removing_a_pin_keeps_the_others_and_re_solves(self):
        self._place(0.5, 0.2, zone=7.0)
        self._place(0.5, 0.8, zone=3.0)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.remove_zone_pin(0)
        pins = self.controller.state.zone_pins
        self.assertEqual(len(pins), 1)
        self.assertEqual(pins[0].target_zone, 3.0)
        self.assertEqual(len(self.render_tasks), n + 1, "the survivor re-solves on its own")
        self.mock_session_manager.update_config.assert_not_called()

    def test_removing_the_last_pin_puts_the_committed_print_back_and_the_tool_down(self):
        from negpy.desktop.session import ToolMode

        self._place(0.5, 0.2, zone=3.0)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.remove_zone_pin(0)
        self.assertEqual(self.controller.state.zone_pins, [])
        self.assertEqual(len(self.render_tasks), n + 1)
        self.assertIs(self.render_tasks[-1].config, self.controller.state.config)
        self.assertEqual(self.controller.state.active_tool, ToolMode.NONE)

    def test_esc_disarms_then_clears_then_puts_the_tool_down(self):
        from negpy.desktop.session import ToolMode
        from negpy.desktop.view.keyboard_shortcuts import _context_cancel

        window = MagicMock()
        window.canvas.overlay.cancel_in_progress.return_value = False
        self._place(0.5, 0.2)
        self._arm(7.0)

        _context_cancel(self.controller, window)
        self.assertIsNone(self.controller.state.zone_arm_target)
        self.assertEqual(len(self.controller.state.zone_pins), 1)

        _context_cancel(self.controller, window)
        self.assertEqual(self.controller.state.zone_pins, [])
        self.assertEqual(self.controller.state.active_tool, ToolMode.ZONE_PLACE)

        _context_cancel(self.controller, window)
        self.assertEqual(self.controller.state.active_tool, ToolMode.NONE)

    def test_apply_commits_and_flips_the_right_autos(self):
        self._place(0.5, 0.2)
        self.controller.apply_zone_placement()
        committed = self.mock_session_manager.update_config.call_args
        exposure = committed.args[0].exposure
        self.assertTrue(committed.kwargs["persist"])
        self.assertFalse(exposure.auto_exposure)
        self.assertTrue(exposure.auto_normalize_contrast, "one pin must not touch Auto Grade")
        self.assertGreaterEqual(exposure.density, 0.0)
        self.assertLessEqual(exposure.density, 2.0)

    def test_two_pin_apply_also_writes_grade(self):
        self._place(0.5, 0.2, zone=7.0)
        self._place(0.5, 0.8, zone=3.0)
        self.controller.apply_zone_placement()
        exposure = self.mock_session_manager.update_config.call_args.args[0].exposure
        self.assertFalse(exposure.auto_exposure)
        self.assertFalse(exposure.auto_normalize_contrast)
        self.assertEqual(exposure.grade, round(exposure.grade))

    def test_the_caption_names_what_is_being_solved(self):
        self.assertEqual(self.controller.zone_solve_caption(), "")
        self._place(0.5, 0.2, zone=7.0)
        self.controller.zone_pin_readouts()
        self.assertEqual(self.controller.zone_solve_caption(), "Solving Print Density")
        self._place(0.5, 0.8, zone=3.0)
        self.controller.zone_pin_readouts()
        self.assertEqual(self.controller.zone_solve_caption(), "Solving Print Density + ISO-R Grade")

    def test_a_third_pin_extends_the_caption(self):
        self._place(0.5, 0.2, zone=7.0)
        self._place(0.5, 0.8, zone=3.0)
        self._place(0.5, 0.5, zone=6.0)
        self.controller.zone_pin_readouts()
        caption = self.controller.zone_solve_caption()
        self.assertTrue(caption.startswith("Solving Print Density + ISO-R Grade"), caption)

    def test_a_three_pin_apply_writes_the_solved_knee(self):
        self._place(0.5, 0.2, zone=7.0)
        self._place(0.5, 0.8, zone=3.0)
        self._place(0.5, 0.5, zone=6.0)
        sol = self.controller._solve_zone_placement()
        self.controller.apply_zone_placement()
        exposure = self.mock_session_manager.update_config.call_args.args[0].exposure
        if sol.knee:
            self.assertEqual(getattr(exposure, sol.knee), sol.fields[sol.knee])

    def test_accepting_commits_and_puts_the_tool_down(self):
        from negpy.desktop.session import ToolMode

        self._place(0.5, 0.2)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.apply_zone_placement()
        self.mock_session_manager.update_config.assert_called_once()
        self.assertEqual(self.controller.state.zone_pins, [], "the placement is made; the proof is spent")
        self.assertEqual(self.controller.state.active_tool, ToolMode.NONE)
        self.assertEqual(len(self.render_tasks), n + 1, "one render of the committed edit")
        self.assertIsNone(self.render_tasks[-1].config_override if hasattr(self.render_tasks[-1], "config_override") else None)

    def test_esc_discards_the_preview_without_committing(self):
        from negpy.desktop.view.keyboard_shortcuts import _context_cancel

        window = MagicMock()
        window.canvas.overlay.cancel_in_progress.return_value = False
        self._place(0.5, 0.2, zone=3.0)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        _context_cancel(self.controller, window)
        self.mock_session_manager.update_config.assert_not_called()
        self.assertEqual(self.controller.state.zone_pins, [])
        self.assertEqual(len(self.render_tasks), n + 1, "the committed print comes back")
        self.assertIs(self.render_tasks[-1].config, self.controller.state.config)

    def test_an_override_render_leaves_the_pins_alone(self):
        from negpy.domain.models import WorkspaceConfig

        self._place(0.5, 0.2)
        self.controller._is_rendering = False
        self.controller.request_render(readback_metrics=False, config_override=WorkspaceConfig())
        self.assertEqual(len(self.controller.state.zone_pins), 1)

    def test_loading_another_frame_drops_the_pins(self):
        self._place(0.5, 0.2)
        with patch.object(self.controller, "_file_hash_for_path", return_value=None):
            self.controller.load_file("/nowhere/other.raw")
        self.assertEqual(self.controller.state.zone_pins, [])

    def test_leaving_the_tool_drops_the_pins(self):
        from negpy.desktop.session import ToolMode

        self._place(0.5, 0.2)
        self.controller.set_active_tool(ToolMode.NONE)
        self.assertEqual(self.controller.state.zone_pins, [])

    def test_target_change_previews_without_committing(self):
        self._place(0.5, 0.2)
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
        self._place(0.5, 0.2)
        self.controller._is_rendering = False
        self.controller.set_zone_pin_target(0, 3.0)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.clear_zone_pins()
        self.assertEqual(self.controller.state.zone_pins, [])
        self.assertEqual(len(self.render_tasks), n + 1)
        self.assertIs(self.render_tasks[-1].config, self.controller.state.config)

    def test_arming_a_zone_takes_the_canvas_from_the_strip(self):
        self.controller.toggle_test_strip()
        mosaics = tuple(np.zeros((8, 8, 3), np.float32) for _ in range(4))
        self.controller.on_strip_finished(mosaics, (0, 0, 8, 8))
        self.assertTrue(self.controller.state.test_strip)
        self._arm(5.0)
        self.assertFalse(self.controller.state.test_strip)

    def test_apply_with_no_pins_commits_nothing(self):
        self._arm()
        self.controller.apply_zone_placement()
        self.mock_session_manager.update_config.assert_not_called()

    def test_dragging_a_pin_rereads_the_tone_under_it(self):
        self._place(0.5, 0.15)
        before = self.controller.state.zone_pins[0]
        self.controller.move_zone_pin(0, 0.5, 0.85)
        pin = self.controller.state.zone_pins[0]
        self.assertAlmostEqual(pin.ny, 0.85)
        self.assertNotAlmostEqual(pin.val_luma, before.val_luma)
        self.assertNotEqual(pin.label, "")

    def test_dragging_an_untargeted_pin_follows_the_new_reading(self):
        self._place(0.5, 0.15)
        self._pin(0.5, 0.5)  # unarmed: a metering pin, no target of its own
        self.controller.move_zone_pin(1, 0.5, 0.85)
        pin = self.controller.state.zone_pins[1]
        measured = self.controller._pin_zone(pin.val_luma)
        self.assertAlmostEqual(pin.target_zone, round(measured * 3.0) / 3.0)

    def test_dragging_a_retargeted_pin_keeps_its_target(self):
        self._place(0.5, 0.15)
        self.controller.set_zone_pin_target(0, 3.0)
        self.controller.move_zone_pin(0, 0.5, 0.85)
        self.assertEqual(self.controller.state.zone_pins[0].target_zone, 3.0)

    def test_the_preview_refreshes_when_the_drag_ends_not_during_it(self):
        self._place(0.5, 0.15)
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
        self._arm()
        self._pin(0.5, 0.15)
        self.controller.clear_zone_pins()
        self._pin(0.5, 0.15)
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.move_zone_pin(0, 0.5, 0.85, final=True)
        self.assertEqual(len(self.render_tasks), n)

    def test_readouts_skip_the_solve_mid_drag(self):
        self._place(0.5, 0.15)
        self._pin(0.5, 0.85)
        with patch.object(self.controller, "_solve_zone_placement", wraps=self.controller._solve_zone_placement) as solve:
            self.controller.zone_pin_readouts()
            self.assertEqual(solve.call_count, 1)
            self.controller.move_zone_pin(0, 0.5, 0.5)
            self.controller.zone_pin_readouts()
            self.assertEqual(solve.call_count, 1, "the nested bisection is too slow to run per mouse-move")
            self.controller.move_zone_pin(0, 0.5, 0.45, final=True)
            self.assertEqual(solve.call_count, 2, "the drag's end re-solves for the preview")
            self.controller.zone_pin_readouts()
            self.assertEqual(solve.call_count, 3)

    def test_degenerate_pins_neither_preview_nor_apply(self):
        self._place(0.2, 0.5)
        self._pin(0.8, 0.5)  # same ramp row -> same tone
        self.controller._is_rendering = False
        n = len(self.render_tasks)
        self.controller.set_zone_pin_target(0, 3.0)
        self.assertEqual(len(self.render_tasks), n)
        self.controller.apply_zone_placement()
        self.mock_session_manager.update_config.assert_not_called()


class TestZoneStripArming(unittest.TestCase):
    """The strip is the control: a cell click says which zone the next pin prints as."""

    def _strip(self):
        from negpy.desktop.view.widgets.charts import ZoneStripWidget

        strip = ZoneStripWidget()
        strip.resize(100, 24)
        strip.update_data(np.full(10, 0.1))
        return strip

    def _click(self, strip, x: float):
        from PyQt6.QtCore import QEvent, QPointF, Qt
        from PyQt6.QtGui import QMouseEvent

        strip.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(x, 12.0),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    def test_a_cell_click_reports_its_zone(self):
        strip = self._strip()
        zones: list = []
        strip.zone_clicked.connect(zones.append)
        self._click(strip, 5.0)  # cell 0
        self._click(strip, 55.0)  # cell V
        self._click(strip, 99.0)  # cell IX
        self.assertEqual(zones, [0, 5, 9])

    def test_the_armed_cell_is_remembered_for_the_repaint(self):
        strip = self._strip()
        strip.set_armed(5)
        self.assertEqual(strip._armed, 5)
        strip.set_armed(None)
        self.assertIsNone(strip._armed)

    def test_a_click_with_no_data_reports_nothing(self):
        from negpy.desktop.view.widgets.charts import ZoneStripWidget

        strip = ZoneStripWidget()
        strip.resize(100, 24)
        zones: list = []
        strip.zone_clicked.connect(zones.append)
        self._click(strip, 55.0)
        self.assertEqual(zones, [])


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

    def test_each_row_can_drop_its_own_pin(self):
        w = self._widget()
        w.refresh([(0, "V", 5.0, None, True), (1, "III", 3.0, None, True)])
        removed: list = []
        w.remove_clicked.connect(removed.append)
        w._removes[1].click()
        self.assertEqual(removed, [1])


class TestZonePlacementConfirmKey(unittest.TestCase):
    """Enter accepts the placement from the canvas, with one pin or two."""

    def _overlay(self, pins: int):
        from negpy.desktop.session import AppState, ToolMode
        from negpy.desktop.view.canvas.overlay import CanvasOverlay
        from negpy.features.exposure.placement import ZonePin

        state = AppState()
        state.zone_pins = [ZonePin(nx=0.5, ny=0.5, val_rgb=(0.5,) * 3, val_luma=0.5, target_zone=5.0) for _ in range(pins)]
        overlay = CanvasOverlay(state)
        overlay.set_tool_mode(ToolMode.ZONE_PLACE)
        return overlay

    def test_enter_accepts_one_pin_and_two(self):
        for count in (1, 2):
            overlay = self._overlay(count)
            confirmed: list = []
            overlay.zone_placement_confirmed.connect(lambda: confirmed.append(True))
            overlay._finish_draw_if_active()
            self.assertEqual(len(confirmed), 1, f"{count} pin(s) must accept on Enter")

    def test_enter_with_no_pins_accepts_nothing(self):
        overlay = self._overlay(0)
        confirmed: list = []
        overlay.zone_placement_confirmed.connect(lambda: confirmed.append(True))
        overlay._finish_draw_if_active()
        self.assertEqual(confirmed, [])


if __name__ == "__main__":
    unittest.main()
