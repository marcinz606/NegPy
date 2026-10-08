"""_sync_scope_buttons: each card's Frame/Roll pair and the roll_override_summary
one-liner that answers "roll-wide or this frame's own". _reset_process_fields: a card's
reset scoped to only the fields it shows.

Stub-on-unbound-method, like test_right_panel_wiring.py: ControlsPanel pulls in every
sidebar in the app, so no test here constructs a real one.
"""

from dataclasses import replace
from unittest.mock import MagicMock, patch

from negpy.desktop.session import AppState
from negpy.desktop.settings_catalog import rows_for_fields
from negpy.desktop.settings_catalog import TONE_FIELDS
from negpy.desktop.view.sidebar.controls_panel import _DEMOSAIC_FIELDS, _SENSOR_FIELDS, ControlsPanel
from negpy.features.exposure.models import ExposureConfig
from negpy.features.process.models import DemosaicMode, ProcessConfig, ProcessMode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _panel_stub(*, active_roll_id="roll1", locked_cards=(), hand_drawn=False) -> MagicMock:
    panel = MagicMock()
    panel._ROLL_CARD_LABELS = ControlsPanel._ROLL_CARD_LABELS
    panel.film_section = MagicMock()
    panel.sensor_section = MagicMock()
    panel.demosaic_section = MagicMock()
    panel.baseline_section = MagicMock()
    panel.process_section = MagicMock()
    panel.autocrop_section = MagicMock()
    panel.optics_section = MagicMock()
    panel.roll_override_summary = MagicMock()
    panel._roll_sections = lambda: ControlsPanel._roll_sections(panel)
    panel._frame_sections = lambda: ControlsPanel._frame_sections(panel)
    panel.controller.state.active_roll_id = active_roll_id
    panel.controller.locked_roll_cards.side_effect = lambda: set(locked_cards)
    panel.controller.frame_section_scopes.side_effect = lambda keys: {key: "frame" for key in keys}
    panel.controller.hand_drawn_crop_departs_from_roll.return_value = hand_drawn
    return panel


def _scope(section) -> str:
    return section.set_scope_buttons.call_args[0][1]


def test_sync_scope_buttons_lists_a_hand_drawn_crop_under_a_rolls_auto_crop():
    panel = _panel_stub(hand_drawn=True)

    ControlsPanel._sync_scope_buttons(panel)

    panel.roll_override_summary.setText.assert_called_with("This frame overrides: Crop (hand-drawn)")
    assert _scope(panel.autocrop_section) == "roll"


def test_sync_scope_buttons_blank_summary_without_an_active_roll():
    panel = _panel_stub(active_roll_id=None)

    ControlsPanel._sync_scope_buttons(panel)

    panel.roll_override_summary.setText.assert_called_once_with("")
    panel.roll_override_summary.setVisible.assert_called_once_with(False)


def test_every_card_reads_frame_with_no_roll_spanning_the_frames():
    """A library search's results are not one roll, so no card can hold a value shared
    across them: every one is that frame's own. Reading Roll there would promise a
    shared value that cannot exist, and the click would have no roll to act on."""
    panel = _panel_stub(active_roll_id=None)

    ControlsPanel._sync_scope_buttons(panel)

    for _key, section in panel._roll_sections():
        assert _scope(section) == "frame"
        assert section.set_scope_buttons.call_args.kwargs["roll_enabled"] is False


def test_the_pair_stays_visible_with_no_roll_so_the_scope_is_still_stated():
    panel = _panel_stub(active_roll_id=None)

    ControlsPanel._sync_scope_buttons(panel)

    for _key, section in panel._roll_sections():
        assert section.set_scope_buttons.call_args[0][0] is True


def test_an_active_roll_leaves_the_roll_half_usable():
    panel = _panel_stub(locked_cards=())

    ControlsPanel._sync_scope_buttons(panel)

    for _key, section in panel._roll_sections():
        assert _scope(section) == "roll"
        assert section.set_scope_buttons.call_args.kwargs["roll_enabled"] is True


def test_sync_scope_buttons_blank_summary_when_nothing_is_overridden():
    panel = _panel_stub(locked_cards=())

    ControlsPanel._sync_scope_buttons(panel)

    panel.roll_override_summary.setText.assert_called_once_with("")


def test_sync_scope_buttons_names_every_overridden_card():
    panel = _panel_stub(locked_cards={"sensor", "process"})

    ControlsPanel._sync_scope_buttons(panel)

    panel.roll_override_summary.setText.assert_called_once_with("This frame overrides: Calibration, Metering")
    panel.roll_override_summary.setVisible.assert_called_once_with(True)


def test_sync_scope_buttons_names_film_mode_too():
    panel = _panel_stub(locked_cards={"film"})

    ControlsPanel._sync_scope_buttons(panel)

    panel.roll_override_summary.setText.assert_called_once_with("This frame overrides: Film Mode")


def test_sync_scope_buttons_marks_only_the_diverged_card_as_frame():
    panel = _panel_stub(locked_cards={"demosaic"})

    ControlsPanel._sync_scope_buttons(panel)

    assert _scope(panel.demosaic_section) == "frame"
    for section in (panel.film_section, panel.sensor_section, panel.process_section, panel.autocrop_section):
        assert _scope(section) == "roll"


def test_sync_scope_buttons_reads_each_frame_cards_own_scope():
    """A frame card reads Roll only once a whole-roll apply put its values there and the
    frame still matches them."""
    panel = _panel_stub()
    panel.controller.frame_section_scopes.side_effect = lambda keys: {key: "roll" if key == "tone" else "frame" for key in keys}

    ControlsPanel._sync_scope_buttons(panel)

    assert _scope(panel.tone_section) == "roll"
    assert _scope(panel.finish_section) == "frame"


def test_sync_scope_buttons_disables_rather_than_hides_the_pair_with_no_roll_open():
    """Frame cards answer the same way Roll-tab cards do: the pair stays readable so the
    scope is stated, and only the half with nothing to act on is turned off."""
    panel = _panel_stub(active_roll_id=None)

    ControlsPanel._sync_scope_buttons(panel)

    for section in (panel.tone_section, panel.sensor_section):
        assert section.set_scope_buttons.call_args[0][0] is True
        assert section.set_scope_buttons.call_args.kwargs["roll_enabled"] is False


def test_sync_scope_buttons_names_the_optics_and_crop_cards():
    panel = _panel_stub(locked_cards={"autocrop", "lens", "flatfield"})

    ControlsPanel._sync_scope_buttons(panel)

    panel.roll_override_summary.setText.assert_called_once_with("This frame overrides: Crop, Optics")
    assert _scope(panel.autocrop_section) == "frame"


def test_optics_reads_frame_when_either_of_its_cards_is_locked():
    for locked in ({"lens"}, {"flatfield"}):
        panel = _panel_stub(locked_cards=locked)

        ControlsPanel._sync_scope_buttons(panel)

        assert _scope(panel.optics_section) == "frame"


def test_the_optics_pair_drives_both_of_its_cards():
    panel = _panel_stub()

    ControlsPanel._on_scope_selected(panel, "optics", "roll")

    panel.controller.set_card_scope.assert_called_once_with(("lens", "flatfield"), "roll")


def test_a_roll_card_routes_both_halves_to_the_controller():
    """Roll-tab cards share one entry point with the Metadata tab's, so the two panels
    cannot drift on what a click means."""
    panel = _panel_stub(locked_cards={"demosaic"})

    ControlsPanel._on_scope_selected(panel, "demosaic", "roll")
    panel.controller.set_card_scope.assert_called_once_with("demosaic", "roll")

    panel.controller.set_card_scope.reset_mock()
    ControlsPanel._on_scope_selected(panel, "sensor", "roll")
    panel.controller.set_card_scope.assert_called_once_with(("sensor", "cast_removal"), "roll")

    panel.controller.set_card_scope.reset_mock()
    ControlsPanel._on_scope_selected(panel, "process", "frame")
    panel.controller.set_card_scope.assert_called_once_with("process", "frame")


def test_a_whole_roll_apply_from_a_frame_card_is_recorded():
    """The record is what lets the card read Roll afterwards; a selection apply is not
    roll-wide and leaves it alone."""
    panel = _panel_stub()
    rows = [r for r in rows_for_fields(("dye_separation",))]

    with patch("negpy.desktop.view.sidebar.controls_panel.open_apply_dialog", return_value=(rows, "roll")):
        ControlsPanel._on_scope_selected(panel, "tone", "roll")
    panel.controller.record_roll_apply.assert_called_once_with(rows)

    panel.controller.record_roll_apply.reset_mock()
    with patch("negpy.desktop.view.sidebar.controls_panel.open_apply_dialog", return_value=(rows, "selection")):
        ControlsPanel._on_scope_selected(panel, "tone", "roll")
    panel.controller.record_roll_apply.assert_not_called()


def test_a_cancelled_apply_records_nothing():
    panel = _panel_stub()

    with patch("negpy.desktop.view.sidebar.controls_panel.open_apply_dialog", return_value=None):
        ControlsPanel._on_scope_selected(panel, "tone", "roll")

    panel.controller.record_roll_apply.assert_not_called()


def test_reset_process_fields_only_touches_the_given_fields():
    """Regression: Normalization's own reset must not also reset Positive (moved
    beside Film Mode) or a Calibration/Demosaic field -- all three cards, and Film
    Mode, live on the same ProcessConfig, but each reset is scoped to its own card."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6, analysis_buffer=0.2, positive_source=True, sensor_profile="Custom"),
    )

    ControlsPanel._reset_process_fields(panel, ("analysis_buffer",))

    new_cfg = panel.controller.apply_config.call_args[0][0]
    assert new_cfg.process.analysis_buffer == ProcessConfig().analysis_buffer
    assert new_cfg.process.positive_source is True
    assert new_cfg.process.sensor_profile == "Custom"


def test_reset_exposure_fields_turns_auto_off_for_a_slide():
    """Tone's reset restores Auto Density/Auto Grade to the mode's own default
    (auto_meter_for_mode), not ExposureConfig's flat one."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6, positive_source=True),
        exposure=replace(cfg.exposure, auto_exposure=True, auto_normalize_contrast=True, density=1.4),
    )

    ControlsPanel._reset_exposure_fields(panel, ("auto_exposure", "auto_normalize_contrast", "density"))

    new_cfg = panel.controller.session.update_config.call_args[0][0]
    assert new_cfg.exposure.auto_exposure is False
    assert new_cfg.exposure.auto_normalize_contrast is False
    assert new_cfg.exposure.density == ExposureConfig().density


def test_reset_exposure_fields_turns_auto_on_for_a_negative_frame():
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, positive_source=False),
        exposure=replace(cfg.exposure, auto_exposure=False, auto_normalize_contrast=False),
    )

    ControlsPanel._reset_exposure_fields(panel, ("auto_exposure", "auto_normalize_contrast"))

    new_cfg = panel.controller.session.update_config.call_args[0][0]
    assert new_cfg.exposure.auto_exposure is True
    assert new_cfg.exposure.auto_normalize_contrast is True


def test_sync_modified_dots_does_not_flag_a_positive_frames_own_auto_default():
    """A Positive frame with Auto Density/Grade correctly off is at its own default,
    not "modified" -- the Tone header's dot must not count it."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = DEFAULT_WORKSPACE_CONFIG
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6, positive_source=True),
        exposure=replace(cfg.exposure, auto_exposure=False, auto_normalize_contrast=False),
    )
    panel.tone_section = MagicMock()

    ControlsPanel._sync_modified_dots(panel)

    panel.tone_section.set_modified.assert_called_once_with(0)


def test_sync_modified_dots_counts_film_mode_on_its_own_card():
    """Film Mode and Positive live on ProcessConfig with Normalization's own fields,
    so a badge that counts the whole config lights the wrong card."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6, positive_source=True),
    )
    panel.film_section = MagicMock()
    panel.process_section = MagicMock()

    ControlsPanel._sync_modified_dots(panel)

    panel.film_section.set_modified.assert_called_once_with(2)
    panel.process_section.set_modified.assert_called_once_with(0)


def test_sync_modified_dots_counts_tonal_range_on_normalization():
    """White/Black Point are Normalization's Tonal Range block, counted with that card."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = DEFAULT_WORKSPACE_CONFIG
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, white_point_offset=0.2, black_point_trim_red=0.1),
    )
    panel.tone_section = MagicMock()
    panel.process_section = MagicMock()

    ControlsPanel._sync_modified_dots(panel)

    panel.tone_section.set_modified.assert_called_once_with(0)
    panel.process_section.set_modified.assert_called_once_with(2)


def test_sync_modified_dots_leaves_the_scanning_setup_uncounted():
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = DEFAULT_WORKSPACE_CONFIG
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, linear_raw=True, narrowband_scan=True, demosaic_preview=DemosaicMode.DHT),
    )
    panel.sensor_section = MagicMock()
    panel.demosaic_section = MagicMock()
    panel.process_section = MagicMock()

    ControlsPanel._sync_modified_dots(panel)

    panel.sensor_section.set_modified.assert_called_once_with(0)
    panel.demosaic_section.set_modified.assert_called_once_with(0)
    panel.process_section.set_modified.assert_called_once_with(0)


def test_a_card_reset_leaves_the_scanning_setup_alone():
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = DEFAULT_WORKSPACE_CONFIG
    panel.controller.state.config = replace(
        cfg,
        process=replace(
            cfg.process,
            linear_raw=True,
            narrowband_scan=True,
            demosaic_preview=DemosaicMode.DHT,
            hue_trim=0.3,
            highlight_reconstruction=True,
        ),
    )

    ControlsPanel._reset_process_fields(panel, _SENSOR_FIELDS)
    after_calibration = panel.controller.apply_config.call_args[0][0].process
    ControlsPanel._reset_process_fields(panel, _DEMOSAIC_FIELDS)
    after_demosaic = panel.controller.apply_config.call_args[0][0].process

    for process in (after_calibration, after_demosaic):
        assert process.linear_raw is True
        assert process.narrowband_scan is True
        assert process.demosaic_preview == DemosaicMode.DHT
    assert after_calibration.hue_trim == DEFAULT_WORKSPACE_CONFIG.process.hue_trim
    assert after_demosaic.highlight_reconstruction == DEFAULT_WORKSPACE_CONFIG.process.highlight_reconstruction


def test_reset_tone_fields_clears_the_print_controls_alone():
    """Tonal Range resets with Normalization, the card it sits on."""
    panel = MagicMock()
    panel.controller.state = AppState()

    ControlsPanel._reset_tone_fields(panel)

    assert panel._reset_exposure_fields.call_args[0][0] == TONE_FIELDS
    panel._reset_process_fields.assert_not_called()


def test_reset_film_fields_routes_through_the_controls_own_setters():
    """Positive rewrites the auto-meter defaults and Film Mode rewrites Cast Removal,
    so a plain field reset would leave both behind."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6, positive_source=True),
    )

    ControlsPanel._reset_film_fields(panel)

    panel.controller.set_positive_source.assert_called_once_with(False)
    panel.controller.set_process_mode.assert_called_once_with(ProcessConfig().process_mode)


def test_the_crop_card_reset_turns_auto_crop_off_and_keeps_a_hand_drawn_crop():
    panel = MagicMock()
    panel.controller.refuse_crop_edit.return_value = False
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(cfg, geometry=replace(cfg.geometry, crop_from_auto=True))

    ControlsPanel._reset_crop_card(panel)

    panel._reset_card_fields.assert_called_once_with("autocrop")
    panel.controller.reset_crop.assert_called_once_with()

    panel.reset_mock()
    panel.controller.state.config = replace(cfg, geometry=replace(cfg.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9)))
    ControlsPanel._reset_crop_card(panel)
    panel.controller.reset_crop.assert_not_called()


def test_the_geometry_reset_clears_a_hand_drawn_crop_and_keeps_auto_crop():
    panel = MagicMock()
    panel.controller.refuse_crop_edit.return_value = False
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    rotated = replace(cfg.geometry, rotation=1, crop_rect=(0.1, 0.1, 0.9, 0.9))

    panel.controller.state.config = replace(cfg, geometry=replace(rotated, crop_from_auto=True, crop_detect_key="k"))
    ControlsPanel._reset_geometry_fields(panel)
    geo = panel.controller.apply_config.call_args.args[0].geometry
    assert geo.rotation == 0 and geo.crop_from_auto and geo.crop_rect == (0.1, 0.1, 0.9, 0.9)
    panel.controller.sync_auto_crop_lock.assert_not_called()

    panel.controller.state.config = replace(cfg, geometry=rotated)
    ControlsPanel._reset_geometry_fields(panel)
    geo = panel.controller.apply_config.call_args.args[0].geometry
    assert geo.rotation == 0 and not geo.crop_from_auto and geo.crop_rect is None
    panel.controller.sync_auto_crop_lock.assert_called_once_with()


def test_reset_film_fields_does_nothing_at_the_defaults():
    panel = MagicMock()
    panel.controller.state = AppState()

    ControlsPanel._reset_film_fields(panel)

    panel.controller.set_positive_source.assert_not_called()
    panel.controller.set_process_mode.assert_not_called()


def test_sync_scope_buttons_offers_reset_to_roll_where_the_controller_finds_one():
    panel = _panel_stub(locked_cards=("process",))
    panel.controller.roll_revert_cards.side_effect = lambda keys: {"process", "lens", "tone"} & set(keys)

    ControlsPanel._sync_scope_buttons(panel)

    panel.process_section.set_roll_revert.assert_called_with(True)
    panel.optics_section.set_roll_revert.assert_called_with(True)
    panel.film_section.set_roll_revert.assert_called_with(False)
    panel.tone_section.set_roll_revert.assert_called_with(True)
    panel.lab_section.set_roll_revert.assert_called_with(False)


def test_the_frame_half_does_not_promise_a_rejoin():
    """Clicking the lit Frame half does nothing; Reset to Roll is the way back."""
    panel = _panel_stub(locked_cards=("film",))

    ControlsPanel._sync_scope_buttons(panel)

    assert "rejoin" not in panel.film_section.set_scope_buttons.call_args.kwargs["frame_tooltip"]


def test_the_calibration_reset_clears_the_unmix_matrices():
    """A profile reset alone leaves the matrix it loaded, and the unmix runs on."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = DEFAULT_WORKSPACE_CONFIG
    panel.controller.state.config = replace(
        cfg,
        process=replace(
            cfg.process,
            sensor_profile="custom",
            sensor_matrix=(1.0, -0.1, 0.0, -0.1, 1.0, -0.1, 0.0, -0.1, 1.0),
            crosstalk_matrix=(1.0, -0.05, 0.0, -0.2, 1.0, -0.05, 0.0, -0.1, 1.0),
            crosstalk_process=ProcessMode.E6,
        ),
    )

    ControlsPanel._reset_process_fields(panel, _SENSOR_FIELDS)

    after = panel.controller.apply_config.call_args[0][0].process
    for field in ("sensor_profile", "sensor_matrix", "crosstalk_matrix", "crosstalk_process"):
        assert getattr(after, field) == getattr(DEFAULT_WORKSPACE_CONFIG.process, field), field


def test_the_metering_header_follows_the_print_histogram():
    import numpy as np

    from negpy.desktop.view.sidebar.controls_panel import ControlsPanel

    panel = MagicMock()
    panel._last_histogram_buf = None
    bins = np.zeros((4, 256))
    panel.controller.state.last_metrics = {"histogram_raw": bins}

    ControlsPanel._update_histogram(panel)

    panel.metering_histogram.update_data.assert_called_once_with(bins)


def _card_menu(monkeypatch, key: str):
    from types import SimpleNamespace

    from PyQt6.QtCore import QPoint

    from negpy.desktop.view.sidebar import controls_panel as cp

    actions = []

    class FakeAction:
        def __init__(self, text):
            self.text, self.enabled, self.slot = text, True, None
            self.triggered = SimpleNamespace(connect=lambda fn: setattr(self, "slot", fn))

        def setToolTip(self, _tip):
            pass

        def setEnabled(self, enabled):
            self.enabled = enabled

    class FakeMenu:
        def __init__(self, *_a):
            pass

        def setToolTipsVisible(self, _on):
            pass

        def addAction(self, text):
            actions.append(FakeAction(text))
            return actions[-1]

        def exec(self, *_a):
            pass

    monkeypatch.setattr(cp, "QMenu", FakeMenu)
    panel = SimpleNamespace(_roll_sections=lambda: (("optics", None),), controller=MagicMock())
    panel.controller.can_undo_roll_push.return_value = False
    cp.ControlsPanel._show_card_menu(panel, key, SimpleNamespace(toggle_button=MagicMock()), QPoint())
    return panel, actions


def test_a_roll_card_menu_offers_undo_only_for_its_own_cards(monkeypatch):
    panel, actions = _card_menu(monkeypatch, "optics")
    assert [a.text for a in actions] == ["Undo Apply to Roll"]
    panel.controller.can_undo_roll_push.assert_called_once_with(("lens", "flatfield"))
    assert not actions[0].enabled


def test_a_frame_card_menu_copies_that_card(monkeypatch):
    from negpy.desktop.settings_catalog import frame_card_rows

    panel, actions = _card_menu(monkeypatch, "lab")
    assert [a.text for a in actions] == ["Copy Card Settings"]
    actions[0].slot()
    panel.controller.session.copy_card_settings.assert_called_once_with(frame_card_rows("lab"))


def test_the_geometry_card_copies_a_crop_with_its_auto_crop():
    """A rect copied without its Auto Crop reads as the other kind on the target frame."""
    from negpy.desktop.settings_catalog import frame_card_rows

    fields = {f for row in frame_card_rows("geometry") for f in row.fields}
    assert {"crop_rect", "crop_detect_key", "crop_from_auto"} <= fields
