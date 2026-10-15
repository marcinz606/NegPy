import pytest

from negpy.desktop.view.widgets.exposure_targets_dialog import ExposureTargetsDialog
from negpy.features.exposure.models import DEFAULT_TARGETS, TUNABLE_TARGETS


@pytest.fixture
def dialog(qapp):
    current = dict(DEFAULT_TARGETS, anchor_target_density=0.65, auto_grade_strength=0.55, shadow_hold_strength=0.0)
    dlg = ExposureTargetsDialog(current)
    yield dlg
    dlg.deleteLater()


def test_every_tunable_target_has_a_slider(dialog):
    assert set(dialog.values()) == set(TUNABLE_TARGETS)


def test_values_round_trip_percent_sliders(dialog):
    values = dialog.values()
    assert values["anchor_target_density"] == pytest.approx(0.65)
    assert values["auto_grade_strength"] == pytest.approx(0.55)
    assert dialog._sliders["auto_grade_strength"].value() == pytest.approx(55.0)


def test_reset_goes_to_the_factory_default_not_the_opened_value(dialog):
    slider = dialog._sliders["anchor_target_density"]
    slider.mouseDoubleClickEvent(None)
    assert dialog.values()["anchor_target_density"] == pytest.approx(DEFAULT_TARGETS["anchor_target_density"])


def test_restore_defaults_previews_the_factory_targets(dialog):
    seen = []
    dialog.targets_previewed.connect(seen.append)
    dialog._restore_defaults()
    assert seen[-1] == pytest.approx(DEFAULT_TARGETS)


def test_off_shows_at_the_minimum(dialog):
    assert dialog._sliders["shadow_hold_strength"].spin.text() == "Off"
    assert dialog._sliders["highlight_hold_density"].spin.text() != "Off"


def test_a_short_restored_height_never_clips_the_hints(dialog, qapp):
    from PyQt6.QtWidgets import QLabel

    dialog.show()
    dialog.resize(340, 200)
    qapp.processEvents()
    hints = [lbl for lbl in dialog.findChildren(QLabel) if lbl.wordWrap()]
    assert hints
    for lbl in hints:
        assert lbl.height() >= lbl.heightForWidth(lbl.width())


def test_shadow_hold_reads_as_a_decimal_like_highlight_hold(dialog):
    dialog._sliders["shadow_hold_strength"].setValue(0.4)
    assert dialog.values()["shadow_hold_strength"] == pytest.approx(0.4)
    assert "%" not in dialog._sliders["shadow_hold_strength"].spin.text()
