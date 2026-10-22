import pytest

from conftest import FakeController as _Controller, FakeRepo as _Repo
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.sidebar.favourites import FavouritesSidebar, load_favourites
from negpy.desktop.view.slider_shortcut_groups import SLIDER_GROUP_BY_ID
from negpy.desktop.view.slider_targets import SLIDER_ATTRS, slider_widget_map
from negpy.desktop.view.widgets.favourites_dialog import FavouritesDialog
from negpy.desktop.view.widgets.sliders import CompactSlider, HueSlider, KelvinSlider, LogSlider, PowerWarpSlider, clone_slider


@pytest.fixture(scope="module")
def controls(qapp):
    controller = _Controller(_Repo())
    return ControlsPanel(controller)


def test_widget_map_covers_every_shortcut_group(controls):
    assert set(slider_widget_map(controls)) == set(SLIDER_GROUP_BY_ID)


def test_every_favouritable_slider_resolves_and_is_clonable(controls):
    """Guards drift in both directions: a renamed sidebar attribute breaks resolution, and a
    slider switched to an unhandled class (e.g. PowerWarpSlider) makes clone_slider raise."""
    widgets = slider_widget_map(controls)
    for slider_id in SLIDER_ATTRS:
        src = widgets[slider_id]()
        clone = clone_slider(src)
        assert clone.label.text() == src.label.text()


def test_clone_round_trips_compact_slider_construction(qapp):
    src = CompactSlider("Density", -2.0, 2.0, 0.25, step=0.05, precision=1000, has_neutral=True, unit=" EV", inverted=True)
    clone = clone_slider(src)
    assert (clone._min, clone._max, clone._default, clone._precision) == (-2.0, 2.0, 0.25, 1000)
    assert clone.spin.singleStep() == pytest.approx(0.05)
    assert clone.spin.suffix() == " EV"
    assert clone.slider.objectName() == "neutral_slider"
    assert clone.slider.invertedAppearance()


def test_clone_round_trips_hue_and_kelvin(qapp):
    hue = clone_slider(HueSlider("Shadow hue", 210.0))
    assert isinstance(hue, HueSlider)
    assert (hue._min, hue._max, hue._default) == (0.0, 360.0, 210.0)

    kelvin = clone_slider(KelvinSlider("Temperature"))
    assert isinstance(kelvin, KelvinSlider)
    assert (kelvin._min, kelvin._max) == (3000.0, 12000.0)


def test_clone_keeps_log_travel(qapp):
    src = LogSlider("Capture Delay", 0, 10, 0, log_floor=0.1, unit=" s")
    clone = clone_slider(src)
    assert isinstance(clone, LogSlider)
    assert clone._log_floor == 0.1
    assert clone._to_int(2.5) == src._to_int(2.5)


def test_every_right_panel_slider_is_clonable(qapp):
    """The Find box clones every slider in the right panel, the Scan tab's included."""
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = RightPanel(_Controller(_Repo()))
    for src in panel.findChildren(CompactSlider):
        assert type(clone_slider(src)) is type(src)


def test_clone_refuses_unhandled_class(qapp):
    """A PowerWarpSlider flattened into a CompactSlider would keep its range but lose its
    nonlinear travel — that must fail loudly, not ship a subtly wrong control."""
    with pytest.raises(TypeError, match="PowerWarpSlider"):
        clone_slider(PowerWarpSlider("Warp", 0.0, 4.0, 1.0, center=1.0))


def test_mirror_value_drives_the_original(qapp):
    src = CompactSlider("Chroma", 0.0, 2.0, 1.0)
    live, committed = [], []
    src.valueChanged.connect(live.append)
    src.valueCommitted.connect(committed.append)

    src.mirror_value(1.5, commit=False)
    assert live == [1.5] and committed == []

    src.mirror_value(1.75, commit=True)
    assert live == [1.5, 1.75] and committed == [1.75]


def test_mirror_commits_even_when_value_matches_a_previous_load(qapp):
    """_rebase_commit=False is load-bearing: setValue's rebase would move the commit baseline
    onto the new value and _on_committed would then see no change and stay silent."""
    src = CompactSlider("Chroma", 0.0, 2.0, 1.0)
    src.setValue(1.5)
    committed = []
    src.valueCommitted.connect(committed.append)
    src.mirror_value(1.5, commit=True)
    assert committed == []

    src.mirror_value(1.6, commit=True)
    assert committed == [1.6]


def _favourites(controls, repo):
    controls.controller.session.repo = repo
    return FavouritesSidebar(controls.controller, controls)


def test_panel_is_empty_by_default(controls, qapp):
    panel = _favourites(controls, _Repo())
    assert panel._mirrors == []
    assert panel.empty_hint.isVisible() or not panel.isVisible()


def test_panel_mirrors_stored_favourites_in_order(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["saturation", "density"]))
    labels = [clone.label.text() for clone, _ in panel._mirrors]
    assert labels == [controls.lab_sidebar.saturation_slider.label.text(), controls.tone_sidebar.density_slider.label.text()]


def test_moving_a_mirror_moves_the_original(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["saturation"]))
    clone, src = panel._mirrors[0]
    committed = []
    src.valueCommitted.connect(committed.append)

    clone.mirror_value(1.4, commit=True)
    assert committed == [1.4]
    assert src.value() == pytest.approx(1.4)


def test_sync_hides_a_mirror_only_when_the_original_is_explicitly_hidden(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["saturation"]))
    clone, src = panel._mirrors[0]

    src.setVisible(False)
    panel.sync_ui()
    assert clone.isHidden()

    src.setVisible(True)
    panel.sync_ui()
    assert not clone.isHidden()


def test_load_drops_unknown_ids(qapp):
    repo = _Repo(favourite_sliders=["density", "retired_slider", "saturation"])
    assert load_favourites(repo) == ["density", "saturation"]


def test_load_tolerates_missing_and_malformed_settings(qapp):
    assert load_favourites(_Repo()) == []
    assert load_favourites(_Repo(favourite_sliders="density")) == []


def _dialog(qapp, selected):
    choices = [("density", "Exposure", "Density"), ("grade", "Exposure", "Grade"), ("saturation", "Lab", "Chroma")]
    return FavouritesDialog(None, choices, selected)


def test_dialog_reads_the_dropped_order_back(qapp):
    """InternalMove takes the row out and re-inserts it, so the list is the truth after a drop."""
    dlg = _dialog(qapp, ["density", "grade", "saturation"])

    dlg.chosen_list.insertItem(0, dlg.chosen_list.takeItem(2))
    dlg.chosen_list.reordered.emit()

    assert dlg.selected_ids() == ["saturation", "density", "grade"]


def test_chosen_list_takes_internal_drags(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QAbstractItemView

    dlg = _dialog(qapp, ["density", "grade"])

    assert dlg.chosen_list.dragDropMode() == QAbstractItemView.DragDropMode.InternalMove
    assert dlg.chosen_list.defaultDropAction() == Qt.DropAction.MoveAction


def test_dialog_ticking_appends_and_unticking_removes(qapp):
    dlg = _dialog(qapp, ["grade"])
    assert dlg.selected_ids() == ["grade"]

    dlg.available_toggles["density"].setChecked(True)
    assert dlg.selected_ids() == ["grade", "density"]

    dlg.available_toggles["grade"].setChecked(False)
    assert dlg.selected_ids() == ["density"]


def test_dialog_drops_stored_ids_it_was_not_offered(qapp):
    dlg = _dialog(qapp, ["density", "retired_slider"])
    assert dlg.selected_ids() == ["density"]


def test_dialog_without_defaults_has_no_restore_button(qapp):
    from PyQt6.QtWidgets import QPushButton

    dlg = _dialog(qapp, ["density"])
    assert [b.text() for b in dlg.findChildren(QPushButton) if b.text() == "Restore Defaults"] == []


def test_dialog_restore_defaults_resets_the_chosen_list_and_the_ticks(qapp):
    """The toolbar editor passes defaults so the stock row is one click away."""
    choices = [("density", "Exposure", "Density"), ("grade", "Exposure", "Grade"), ("saturation", "Lab", "Chroma")]

    dlg = FavouritesDialog(None, choices, ["saturation"], defaults=["grade", "density"])

    dlg._restore_defaults()

    assert dlg.selected_ids() == ["grade", "density"]
    states = {slider_id: toggle.isChecked() for slider_id, toggle in dlg.available_toggles.items()}
    assert states == {"density": True, "grade": True, "saturation": False}
