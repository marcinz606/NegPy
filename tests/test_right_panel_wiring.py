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


def test_reveal_section_switches_to_roll_for_the_geometry_and_flat_field_cards():
    """Crop, Optics and Frame Assembly are Roll-tab cards, so the ⓘ
    and the search must not look for them among Frame's sub-tabs."""
    for attr in ("autocrop_section", "optics_section", "assembly_section"):
        panel = _group_panel_stub()
        panel._section_tab_index = {}

        RightPanel.reveal_section(panel, attr)

        panel._switch_group.assert_called_once_with(0)
        panel._switch_tab.assert_not_called()


def test_reveal_section_ignores_an_unknown_section():
    panel = _group_panel_stub()
    panel._section_tab_index = {}

    RightPanel.reveal_section(panel, "nope")

    panel._switch_group.assert_not_called()
    panel._switch_tab.assert_not_called()


def test_scroll_to_centered_moves_a_row_already_in_view(qapp):
    from PyQt6.QtWidgets import QApplication, QLabel, QScrollArea, QVBoxLayout, QWidget

    area = QScrollArea()
    area.setWidgetResizable(True)
    body = QWidget()
    layout = QVBoxLayout(body)
    rows = [QLabel(f"row {i}") for i in range(60)]
    for row in rows:
        row.setFixedHeight(20)
        layout.addWidget(row)
    area.setWidget(body)
    area.resize(200, 300)
    area.show()
    QApplication.processEvents()
    target = rows[12]

    RightPanel.scroll_to(MagicMock(), target, centered=True)

    y = target.mapTo(area.viewport(), target.rect().topLeft()).y()
    assert abs(y - area.viewport().height() // 3) <= 1


def test_first_run_analysis_height_shrinks_on_a_short_screen():
    from negpy.desktop.view.sidebar.right_panel import default_analysis_split

    def screen(height):
        return MagicMock(availableGeometry=lambda: MagicMock(height=lambda: height))

    assert default_analysis_split(screen(1440))[0] == 320
    assert default_analysis_split(screen(900))[0] == 270
    assert default_analysis_split(None)[0] == 320
