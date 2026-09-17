"""Signal wiring of the right panel's analysis refresh, and the outer Roll / Frame /
Metadata / Gear / Export / Scan tab switch.

_paint_negative_peek emits image_updated only, never metrics_available, so the
image_updated path must refresh the histograms itself or entering Peek Negative
leaves the chart in print mode.

Stubs on the unbound methods throughout: no test in this repo constructs a real
RightPanel, since it pulls in a full chain of sidebars that need a real controller.
"""

from unittest.mock import MagicMock

from negpy.desktop.view.sidebar.right_panel import RightPanel


def _panel_stub(last_metrics: dict) -> MagicMock:
    panel = MagicMock()
    panel.controller.session.state.last_metrics = last_metrics
    panel.controller.state.flat_peek = False
    panel.controller.state.negative_peek = True
    panel._clip_fracs = (None, None)
    # Nothing is being soft-proofed in these stubs, so the printability row is absent.
    panel._gamut_fraction.return_value = None
    return panel


def test_update_analysis_refreshes_histograms() -> None:
    metrics = {"interactive": False, "histogram_density": [1.0]}
    panel = _panel_stub(metrics)

    RightPanel._update_analysis(panel)

    panel._update_histograms.assert_called_once_with(metrics)


def test_update_analysis_skips_mid_gesture_frames() -> None:
    panel = _panel_stub({"interactive": True})

    RightPanel._update_analysis(panel)

    panel._update_histograms.assert_not_called()


def _group_panel_stub(*, scan_index: int = 5, active_group: int = 0, n_groups: int = 6) -> MagicMock:
    panel = MagicMock()
    panel._group_buttons = [MagicMock() for _ in range(n_groups)]
    panel._group_icons = ["mdi6.film", "fa5s.image", "fa5s.tags", "fa5s.toolbox", "fa5s.file-export", "fa5s.camera-retro"][:n_groups]
    panel._group_keys = ["roll", "frame", "metadata", "gear", "export", "scan"][:n_groups]
    panel._scan_group_index = scan_index
    panel._active_group = active_group
    return panel


def test_switch_group_persists_the_index_and_updates_the_stack():
    panel = _group_panel_stub()

    RightPanel._switch_group(panel, 1)

    panel.controller.session.repo.save_global_setting.assert_called_once_with("right_panel_group", 1)
    panel.group_stack.setCurrentIndex.assert_called_once_with(1)
    panel.group_switcher.set_pinned.assert_called_once_with(1)
    assert panel._active_group == 1
    panel._group_buttons[1].setChecked.assert_called_once_with(True)
    panel._group_buttons[0].setChecked.assert_called_once_with(False)


def test_switch_group_activates_scan_sidebars_only_on_the_scan_tab():
    panel = _group_panel_stub(scan_index=5)

    RightPanel._switch_group(panel, 0)
    panel.scan_sidebar.on_activated.assert_not_called()
    panel.scanlight_sidebar.on_activated.assert_not_called()

    RightPanel._switch_group(panel, 5)
    panel.scan_sidebar.on_activated.assert_called_once_with()
    panel.scanlight_sidebar.on_activated.assert_called_once_with()


def test_show_tab_by_key_dispatches_to_a_group_tab():
    panel = _group_panel_stub()
    panel._tab_keys = ["favourites", "geometry"]

    RightPanel.show_tab_by_key(panel, "metadata")

    panel._switch_group.assert_called_once_with(2)
    panel._switch_tab.assert_not_called()


def test_show_tab_by_key_dispatches_to_a_frame_tab():
    panel = _group_panel_stub()
    panel._tab_keys = ["favourites", "geometry"]

    RightPanel.show_tab_by_key(panel, "geometry")

    panel._switch_group.assert_called_once_with(1)
    panel._switch_tab.assert_called_once_with(1)


def test_show_tab_by_key_ignores_an_unknown_key():
    panel = _group_panel_stub()
    panel._tab_keys = ["favourites", "geometry"]

    RightPanel.show_tab_by_key(panel, "not-a-real-tab")

    panel._switch_group.assert_not_called()
    panel._switch_tab.assert_not_called()


def test_reveal_section_switches_to_frame_then_the_section_tab():
    panel = _group_panel_stub()
    panel._section_tab_index = {"retouch_section": 3}

    RightPanel.reveal_section(panel, "retouch_section")

    panel._switch_group.assert_called_once_with(1)
    panel._switch_tab.assert_called_once_with(3)


def test_reveal_section_switches_to_roll_for_a_roll_section():
    """sensor_section (Calibration) lives on the Roll tab, not as a Frame sub-tab --
    switching group is the whole job, since Roll has no inner switcher to land on."""
    panel = _group_panel_stub()
    panel._section_tab_index = {}

    RightPanel.reveal_section(panel, "sensor_section")

    panel._switch_group.assert_called_once_with(0)
    panel._switch_tab.assert_not_called()


def test_reveal_section_ignores_an_unknown_section():
    panel = _group_panel_stub()
    panel._section_tab_index = {}

    RightPanel.reveal_section(panel, "nope")

    panel._switch_group.assert_not_called()
    panel._switch_tab.assert_not_called()


def _roll_scope_panel_stub() -> MagicMock:
    panel = MagicMock()
    panel._roll_scope_actions = {k: MagicMock() for k in ("all", "selected")}
    return panel


def test_set_roll_edit_scope_checks_the_matching_action_and_labels_the_button():
    panel = _roll_scope_panel_stub()

    RightPanel._set_roll_edit_scope(panel, "selected")

    panel._roll_scope_actions["selected"].setChecked.assert_called_once_with(True)
    panel.roll_scope_btn.setText.assert_called_once_with(" Apply to Selected")
    panel.controller.set_roll_edit_scope.assert_called_once_with("selected")


def test_set_roll_edit_scope_enables_force_settings_only_for_all():
    panel = _roll_scope_panel_stub()

    RightPanel._set_roll_edit_scope(panel, "all")
    panel.roll_force_btn.setEnabled.assert_called_once_with(True)

    panel.roll_force_btn.reset_mock()
    RightPanel._set_roll_edit_scope(panel, "selected")
    panel.roll_force_btn.setEnabled.assert_called_once_with(False)


def test_set_roll_edit_scope_with_persist_false_does_not_write_the_setting():
    panel = _roll_scope_panel_stub()

    RightPanel._set_roll_edit_scope(panel, "all", persist=False)

    panel.controller.set_roll_edit_scope.assert_not_called()


def test_apply_clicked_runs_apply_to_roll_by_default():
    panel = _roll_scope_panel_stub()
    panel.controller.roll_edit_scope.return_value = "all"

    RightPanel._on_roll_apply_clicked(panel)

    panel.controller.apply_roll_cards_to_roll.assert_called_once()
    panel.controller.apply_roll_cards_to_selected.assert_not_called()


def test_apply_clicked_runs_apply_to_selected_when_that_is_the_current_scope():
    panel = _roll_scope_panel_stub()
    panel.controller.roll_edit_scope.return_value = "selected"

    RightPanel._on_roll_apply_clicked(panel)

    panel.controller.apply_roll_cards_to_selected.assert_called_once()
    panel.controller.apply_roll_cards_to_roll.assert_not_called()


def test_set_roll_edit_scope_refreshes_the_apply_buttons_enabled_state():
    """A scope switch can change whether the current scope has anything to apply, so
    it must re-check, not just relabel the button."""
    panel = _roll_scope_panel_stub()

    RightPanel._set_roll_edit_scope(panel, "selected")

    panel._sync_roll_apply_enabled.assert_called_once()


def test_force_toggled_sets_the_override_and_refreshes_apply_enabled():
    panel = _roll_scope_panel_stub()

    RightPanel._on_roll_force_toggled(panel, True)

    panel.controller.set_roll_override_locked_frames.assert_called_once_with(True)
    panel._sync_roll_apply_enabled.assert_called_once()


def test_sync_roll_apply_enabled_enables_the_button_when_something_can_apply():
    panel = _roll_scope_panel_stub()
    panel.controller.roll_edit_scope.return_value = "all"
    panel.controller.can_apply_roll_cards.return_value = True

    RightPanel._sync_roll_apply_enabled(panel)

    panel.roll_scope_btn.setEnabled.assert_called_once_with(True)
    panel.roll_scope_btn.setToolTip.assert_called_once_with("Apply to all frames in the roll")


def test_sync_roll_apply_enabled_disables_and_explains_when_nothing_can_apply():
    panel = _roll_scope_panel_stub()
    panel.controller.roll_edit_scope.return_value = "selected"
    panel.controller.can_apply_roll_cards.return_value = False

    RightPanel._sync_roll_apply_enabled(panel)

    panel.roll_scope_btn.setEnabled.assert_called_once_with(False)
    panel.roll_scope_btn.setToolTip.assert_called_once_with("Nothing to apply — every card already follows the roll")
