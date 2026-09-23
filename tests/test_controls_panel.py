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
from negpy.desktop.view.sidebar.controls_panel import _TONE_FIELDS, ControlsPanel
from negpy.features.exposure.models import ExposureConfig
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _panel_stub(*, active_roll_id="roll1", locked_cards=()) -> MagicMock:
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
    return panel


def _scope(section) -> str:
    return section.set_scope_buttons.call_args[0][1]


def test_sync_scope_buttons_blank_summary_without_an_active_roll():
    panel = _panel_stub(active_roll_id=None)

    ControlsPanel._sync_scope_buttons(panel)

    panel.roll_override_summary.setText.assert_called_once_with("")


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
    panel = _panel_stub(locked_cards={"sensor"})

    ControlsPanel._on_scope_selected(panel, "sensor", "roll")
    panel.controller.set_card_scope.assert_called_once_with("sensor", "roll")

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
    assert panel.controller.record_section_push.call_args[0][0] == "tone"

    panel.controller.record_section_push.reset_mock()
    with patch("negpy.desktop.view.sidebar.controls_panel.open_apply_dialog", return_value=(rows, "selection")):
        ControlsPanel._on_scope_selected(panel, "tone", "roll")
    panel.controller.record_section_push.assert_not_called()


def test_a_cancelled_apply_records_nothing():
    panel = _panel_stub()

    with patch("negpy.desktop.view.sidebar.controls_panel.open_apply_dialog", return_value=None):
        ControlsPanel._on_scope_selected(panel, "tone", "roll")

    panel.controller.record_section_push.assert_not_called()


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


def test_reset_exposure_fields_turns_auto_off_for_a_positive_frame():
    """Regression: Tone's reset used to restore ExposureConfig's own flat default
    (on) regardless of Positive, so resetting a Positive frame turned Auto Density/
    Auto Grade back on instead of to the value auto_meter_for_positive_source gives it."""
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


def test_sync_modified_dots_counts_linear_raw_on_calibration():
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = DEFAULT_WORKSPACE_CONFIG
    panel.controller.state.config = replace(cfg, process=replace(cfg.process, linear_raw=True))
    panel.sensor_section = MagicMock()
    panel.process_section = MagicMock()

    ControlsPanel._sync_modified_dots(panel)

    panel.sensor_section.set_modified.assert_called_once_with(1)
    panel.process_section.set_modified.assert_called_once_with(0)


def test_reset_tone_fields_clears_the_print_controls_alone():
    """Tonal Range resets with Normalization, the card it sits on."""
    panel = MagicMock()
    panel.controller.state = AppState()

    ControlsPanel._reset_tone_fields(panel)

    assert panel._reset_exposure_fields.call_args[0][0] == _TONE_FIELDS
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


def test_reset_film_fields_does_nothing_at_the_defaults():
    panel = MagicMock()
    panel.controller.state = AppState()

    ControlsPanel._reset_film_fields(panel)

    panel.controller.set_positive_source.assert_not_called()
    panel.controller.set_process_mode.assert_not_called()
