"""RollAnalysisSidebar: a plain picker over the library's rolls. Picking one is the
whole action -- it loads that roll's saved Roll Analysis baseline immediately, no
separate Apply. Reanalyze, beside the picker, runs Roll Analysis itself (the metering
run that fills the tick in) -- the same action the Library's "Analyze Roll…" offers
(tested in test_library_tree.py), reachable here too, enabled only for the loaded roll."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.roll import RollAnalysisSidebar
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls


def _repo() -> MagicMock:
    """A repository whose global settings live in a dict, so a write is readable back."""
    repo = MagicMock(spec=StorageRepository)
    store: dict = {}
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    return repo


def _sidebar(roll_names=(), analyzed=(), active_name=None):
    """Builds a real repo with one virtual roll per name, a controller stub pointed at
    it, and the sidebar under test. Returns (controller, sidebar, {name: roll_id})."""
    repo = _repo()
    ids = {}
    for name in roll_names:
        roll_id = rolls.create_virtual_roll(repo, name, [])
        ids[name] = roll_id
        if name in analyzed:
            rolls.set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))

    controller = MagicMock()
    controller.session.repo = repo
    controller.state = AppState()
    controller.state.active_roll_id = ids.get(active_name)
    return controller, RollAnalysisSidebar(controller), ids


def test_picker_owns_its_batch_analysis_subheader(qapp):
    """The subheader belongs to the picker itself, not the composite Normalization
    body -- everything else in that card (Analysis Buffer, clip sliders, White/Black
    Point) isn't Roll Analysis, and labeling it that way would mislead."""
    _, sidebar, _ids = _sidebar()
    assert sidebar.layout.itemAt(0).widget() is not sidebar.roll_combo
    assert sidebar.layout.itemAt(1).layout().itemAt(0).widget() is sidebar.roll_combo


def test_picker_defaults_to_the_active_roll_and_lists_every_library_roll(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Portra 400", "Tri-X"], active_name="Tri-X")
    assert sidebar.roll_combo.selected_id() == ids["Tri-X"]
    assert sidebar.roll_combo.line_edit().text() == "Tri-X"
    sidebar.roll_combo.set_selected_id(ids["Portra 400"])
    assert sidebar.roll_combo.line_edit().text() == "Portra 400"


def test_reanalyze_button_runs_batch_normalization(qapp):
    controller, sidebar, _ids = _sidebar(roll_names=["Tri-X"], active_name="Tri-X")
    sidebar.reanalyze_btn.click()
    controller.request_batch_normalization.assert_called_once()


def test_reanalyze_button_enabled_only_for_the_loaded_roll(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Portra 400", "Tri-X"], active_name="Tri-X")
    assert sidebar.reanalyze_btn.isEnabled()

    # set_selected_id alone only updates the display; _on_roll_picked is what
    # selection_changed actually fires, so drive it directly here.
    sidebar._on_roll_picked(ids["Portra 400"])

    assert not sidebar.reanalyze_btn.isEnabled()
    assert "Open this roll first" in sidebar.reanalyze_btn.toolTip()


def test_reanalyze_button_disabled_with_no_roll_loaded(qapp):
    _, sidebar, _ids = _sidebar(roll_names=["Tri-X"])
    assert not sidebar.reanalyze_btn.isEnabled()


def test_use_this_frame_writes_the_frames_bounds_as_the_baseline(qapp):
    controller, sidebar, ids = _sidebar(roll_names=["Tri-X"], active_name="Tri-X")
    sidebar.from_frame_btn.click()
    controller.set_roll_baseline_from_frame.assert_called_once_with(ids["Tri-X"])


def test_use_this_frame_is_gated_the_same_way_reanalyze_is(qapp):
    """Both write onto the files currently loaded, so neither is offered for a roll
    that is only being looked at."""
    _, sidebar, ids = _sidebar(roll_names=["Portra 400", "Tri-X"], active_name="Tri-X")
    assert sidebar.from_frame_btn.isEnabled()

    sidebar._on_roll_picked(ids["Portra 400"])

    assert not sidebar.from_frame_btn.isEnabled()
    assert "Open this roll first" in sidebar.from_frame_btn.toolTip()


def test_the_baseline_buttons_sit_right_of_the_picker(qapp):
    """The same shape as a scene row: the field, then its icon actions."""
    _, sidebar, _ids = _sidebar()

    row = sidebar.layout.itemAt(1).layout()
    widgets = [row.itemAt(i).widget() for i in range(row.count())]
    assert widgets == [sidebar.roll_combo, sidebar.reanalyze_btn, sidebar.from_frame_btn]
    assert sidebar.layout.itemAt(0).widget().text() == "ROLLS"


def test_the_baseline_icon_buttons_name_themselves_in_the_tooltip(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Portra 400"], active_name="Portra 400")
    sidebar._on_roll_picked(ids["Portra 400"])
    assert sidebar.reanalyze_btn.text() == ""
    assert "Roll Analysis" in sidebar.reanalyze_btn.toolTip()
    assert "Use This Frame" in sidebar.from_frame_btn.toolTip()


def test_the_loaded_roll_is_pinned_first_in_the_dropdown(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Agfa", "Tri-X", "Velvia"], active_name="Velvia")
    listed_ids = [item_id for _label, item_id, _search in sidebar.roll_combo._entries]
    assert listed_ids == [ids["Velvia"], ids["Agfa"], ids["Tri-X"]]


def test_dropdown_stays_alphabetical_without_an_active_roll(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Agfa", "Tri-X", "Velvia"])
    listed_ids = [item_id for _label, item_id, _search in sidebar.roll_combo._entries]
    assert listed_ids == [ids["Agfa"], ids["Tri-X"], ids["Velvia"]]


def test_analyzed_rolls_get_a_tick_in_their_label(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Tri-X", "Portra 400"], analyzed=["Tri-X"], active_name="Tri-X")
    assert sidebar.roll_combo.line_edit().text() == "✓ Tri-X"
    sidebar.roll_combo.set_selected_id(ids["Portra 400"])
    assert sidebar.roll_combo.line_edit().text() == "Portra 400"


def test_picking_a_roll_loads_its_saved_baseline(qapp):
    controller, sidebar, ids = _sidebar(roll_names=["Tri-X", "Portra 400"], analyzed=["Tri-X"], active_name="Portra 400")

    sidebar._on_roll_picked(ids["Tri-X"])

    controller.apply_normalization_roll.assert_called_once_with(ids["Tri-X"])


def test_picking_blank_does_not_touch_the_controller(qapp):
    controller, sidebar, _ids = _sidebar(roll_names=["Tri-X"], active_name="Tri-X")

    sidebar._on_roll_picked("")

    controller.apply_normalization_roll.assert_not_called()


def test_sync_ui_selects_the_config_roll_name_when_it_still_exists(qapp):
    controller, sidebar, ids = _sidebar(roll_names=["Tri-X", "Portra 400"], active_name="Portra 400")
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Tri-X"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == ids["Tri-X"]


def test_sync_ui_falls_back_to_the_active_roll_when_the_saved_name_is_gone(qapp):
    """A deleted or renamed roll_name must not leave the picker pointed at nothing
    while a roll is loaded."""
    controller, sidebar, ids = _sidebar(roll_names=["Portra 400"], active_name="Portra 400")
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Deleted Roll"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == ids["Portra 400"]


def test_sync_ui_falls_back_to_blank_without_an_active_roll(qapp):
    controller, sidebar, _ids = _sidebar(roll_names=["Portra 400"])
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Deleted Roll"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == ""


def test_status_hint_warns_when_a_different_roll_is_picked(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Tri-X", "Portra 400"], active_name="Portra 400")

    sidebar._on_roll_picked(ids["Tri-X"])

    assert "Tri-X" in sidebar.roll_status_hint.text()
    assert "different roll" in sidebar.roll_status_hint.text()
    assert sidebar.roll_status_hint.property("hint") == "warning"


def test_status_hint_is_blank_when_the_active_roll_is_picked(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Tri-X", "Portra 400"], active_name="Portra 400")

    sidebar._on_roll_picked(ids["Portra 400"])

    assert sidebar.roll_status_hint.text() == ""


def test_status_hint_is_blank_without_an_active_roll(qapp):
    _, sidebar, _ids = _sidebar(roll_names=["Portra 400"])
    assert sidebar.roll_status_hint.text() == ""


def _scene_labels(sidebar):
    layout = sidebar._scene_rows_layout
    return [layout.itemAt(i).widget().layout().itemAt(0).widget().text() for i in range(layout.count())]


def test_scene_list_shows_each_scene_with_its_loaded_frames(qapp):
    controller, sidebar, ids = _sidebar(roll_names=["Tri-X"], active_name="Tri-X")
    repo = controller.session.repo
    beach = rolls.create_scene(repo, ids["Tri-X"], "Beach", ["h1", "h2"])
    rolls.create_scene(repo, ids["Tri-X"], "Night", ["h3"])
    rolls.set_scene_normalization(repo, ids["Tri-X"], beach, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
    controller.state.uploaded_files = [{"hash": "h1", "scene": (1, beach, "Beach")}, {"hash": "h2", "scene": (1, beach, "Beach")}]

    sidebar._refresh_scenes()

    labels = _scene_labels(sidebar)
    assert "Beach · 2 frames · ✓" in labels[0]
    assert "Night · 0 frames" in labels[1] and "✓" not in labels[1]
    assert not sidebar.scenes_hint.isVisibleTo(sidebar)


def test_scene_row_analyze_runs_scene_analysis(qapp):
    controller, sidebar, ids = _sidebar(roll_names=["Tri-X"], active_name="Tri-X")
    sid = rolls.create_scene(controller.session.repo, ids["Tri-X"], "Beach", ["h1"])
    controller.state.uploaded_files = [{"hash": "h1", "scene": (1, sid, "Beach")}]
    sidebar._refresh_scenes()

    sidebar._scene_rows_layout.itemAt(0).widget().layout().itemAt(1).widget().click()
    controller.request_scene_analysis.assert_called_once_with(sid)


def test_empty_roll_shows_the_grouping_hint(qapp):
    _, sidebar, _ids = _sidebar(roll_names=["Tri-X"], active_name="Tri-X")
    assert sidebar.scenes_hint.isVisibleTo(sidebar)


def test_scene_block_is_hidden_outside_a_roll(qapp):
    _, sidebar, _ids = _sidebar()
    assert not sidebar.scenes_header.isVisibleTo(sidebar)
    assert not sidebar.scenes_hint.isVisibleTo(sidebar)


def test_scene_row_delete_asks_then_deletes(qapp):
    from unittest.mock import patch

    controller, sidebar, ids = _sidebar(roll_names=["Tri-X"], active_name="Tri-X")
    sid = rolls.create_scene(controller.session.repo, ids["Tri-X"], "Beach", ["h1"])
    sidebar._refresh_scenes()
    delete = sidebar._scene_rows_layout.itemAt(0).widget().layout().itemAt(3).widget()

    with patch("negpy.desktop.view.confirm.confirm_delete_named", return_value=False):
        delete.click()
    controller.request_delete_scene.assert_not_called()

    with patch("negpy.desktop.view.confirm.confirm_delete_named", return_value=True) as ask:
        delete.click()
    assert ask.call_args.args[1:3] == ("Scene", "Beach")
    controller.request_delete_scene.assert_called_once_with(sid)
