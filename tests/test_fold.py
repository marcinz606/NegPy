from unittest.mock import MagicMock

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from negpy.desktop.view.styles.templates import FieldLabel, field_row
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import hidden_by_gating, make_section
from negpy.desktop.view.widgets.fold import Fold, FoldButton, group_fold, rail_fold
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.widgets.step_combo import StepCombo


def _repo(saved=None):
    store = dict(saved or {})
    repo = MagicMock()
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = store.__setitem__
    return repo, store


def _card(repo, *widgets):
    body = QWidget()
    layout = QVBoxLayout(body)
    for w in widgets:
        layout.addWidget(w)
    return make_section(repo, "Tone", "tone", body, "fa5s.adjust", default_expanded=True)


def _grade_card(repo):
    grade = CompactSlider("ISO-R Grade", 50, 180, 115)
    shadows, highlights = CompactSlider("Shadows Grade", -50, 50, 0), CompactSlider("Highlights Grade", -50, 50, 0)
    fold = rail_fold("tone_grade", grade, shadows, highlights)
    return _card(repo, grade, fold), grade, shadows, fold


def test_a_rail_starts_folded_and_its_button_sits_in_the_parent_row():
    repo, _ = _repo()
    section, grade, shadows, fold = _grade_card(repo)
    section.show()

    assert not fold.is_open() and not shadows.isVisible()
    assert QWidget.layout(grade).indexOf(fold.button) == 0


def test_the_state_left_is_saved_and_comes_back():
    repo, store = _repo()
    _section, _grade, _shadows, fold = _grade_card(repo)
    fold.button.click()
    assert store["fold_tone_grade"] is True

    repo2, _ = _repo(store)
    _section, _grade, shadows, fold = _grade_card(repo2)
    assert fold.is_open() and shadows.isVisibleTo(fold.parentWidget())


def test_a_closed_fold_marks_an_edited_control_behind_it():
    repo, _ = _repo()
    _section, _grade, shadows, fold = _grade_card(repo)
    assert not fold.button.is_marked()

    shadows.setValue(-15)
    assert fold.button.is_marked()
    fold.set_open(True)
    assert not fold.button.is_marked()
    shadows.setValue(0)
    fold.set_open(False)
    assert not fold.button.is_marked()


def test_folded_is_not_gated_but_a_hidden_rail_is():
    repo, _ = _repo()
    section, _grade, shadows, fold = _grade_card(repo)
    section.show()
    assert not hidden_by_gating(shadows)

    fold.content.setVisible(False)
    assert hidden_by_gating(shadows)
    assert fold.button.isHidden()
    fold.content.setVisible(True)
    assert not fold.button.isHidden()


def test_the_parent_label_gives_the_gutter_up_so_the_tracks_line_up():
    repo, _ = _repo()
    section, grade, shadows, _fold = _grade_card(repo)
    section.resize(480, 300)
    section.show()

    assert grade.label.minimumWidth() + FoldButton.GUTTER + THEME.space_md == shadows.label.minimumWidth() + 8


def test_a_field_row_opens_with_the_chevron_and_stays_aligned():
    repo, _ = _repo()
    mode = StepCombo()
    row = field_row("Mode", mode)
    offset = CompactSlider("Crop Offset", -5, 100, 0)
    fold = rail_fold("crop_offset", row, offset)
    body = QWidget()
    layout = QVBoxLayout(body)
    layout.addLayout(row)
    layout.addWidget(fold)
    make_section(repo, "Crop", "autocrop", body, "fa5s.crop", default_expanded=True)

    assert row.itemAt(0).widget() is fold.button
    assert isinstance(row.itemAt(1).widget(), FieldLabel)
    assert row.itemAt(1).widget().property("fold_gutter") == FoldButton.GUTTER + THEME.space_md


def test_a_group_fold_header_folds_its_items():
    repo, store = _repo()
    glow, halation = CompactSlider("Glow", 0, 1, 0), CompactSlider("Halation", 0, 1, 0)
    header, fold = group_fold("lab_effects", "Effects", glow, halation)
    body = QWidget()
    layout = QVBoxLayout(body)
    layout.addLayout(header)
    layout.addWidget(fold)
    section = make_section(repo, "Lab", "lab", body, "fa5s.flask", default_expanded=True)
    section.show()

    assert fold.button.text() == "EFFECTS" and not glow.isVisible()
    fold.button.click()
    assert glow.isVisible() and store["fold_lab_effects"] is True
    assert "Glow" in fold.button.toolTip()


def test_reveal_opens_the_fold_that_holds_a_control():
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    repo, _ = _repo()
    _section, _grade, shadows, fold = _grade_card(repo)
    panel = MagicMock()
    panel._pages_holding.return_value = []
    RightPanel.reveal_widget(panel, shadows)

    assert fold.is_open()
    assert isinstance(fold, Fold)


def _mouse(widget, kind, x, buttons=None):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    left = Qt.MouseButton.LeftButton
    held = left if buttons is None else buttons
    pos = QPointF(x, 5)
    QApplication.sendEvent(widget, QMouseEvent(kind, pos, widget.mapToGlobal(pos), left, held, Qt.KeyboardModifier.NoModifier))


def test_a_click_on_the_parents_name_leaves_the_fold_and_a_drag_still_scrubs():
    """A click on the name is half of the reset double-click; only the chevron toggles."""
    from PyQt6.QtCore import QEvent

    repo, _ = _repo()
    section, grade, _shadows, fold = _grade_card(repo)
    section.show()
    before = grade.value()

    _mouse(grade.label, QEvent.Type.MouseButtonPress, 10)
    _mouse(grade.label, QEvent.Type.MouseMove, 12)
    _mouse(grade.label, QEvent.Type.MouseButtonRelease, 12)
    assert not fold.is_open() and grade.value() == before, "a click within the drag distance is a click"

    _mouse(grade.label, QEvent.Type.MouseButtonPress, 10)
    _mouse(grade.label, QEvent.Type.MouseMove, 60)
    _mouse(grade.label, QEvent.Type.MouseButtonRelease, 60)
    assert not fold.is_open() and grade.value() != before, "a drag scrubs and leaves the fold alone"


def test_a_click_on_a_field_rows_name_leaves_the_fold():
    from PyQt6.QtCore import QEvent

    repo, _ = _repo()
    row = field_row("Mode", StepCombo())
    fold = rail_fold("crop_offset", row, CompactSlider("Crop Offset", -5, 100, 0))
    body = QWidget()
    layout = QVBoxLayout(body)
    layout.addLayout(row)
    layout.addWidget(fold)
    make_section(repo, "Crop", "autocrop", body, "fa5s.crop", default_expanded=True)

    _mouse(row.itemAt(1).widget(), QEvent.Type.MouseButtonRelease, 5)
    assert not fold.is_open()


def test_a_fold_can_start_open_until_the_user_closes_it():
    repo, store = _repo()
    glow = CompactSlider("Glow", 0, 1, 0)
    header, fold = group_fold("paper_curve_shape", "Curve Shape", glow, default_open=True)
    body = QWidget()
    layout = QVBoxLayout(body)
    layout.addLayout(header)
    layout.addWidget(fold)
    make_section(repo, "Paper", "paper", body, "fa5s.scroll", default_expanded=True)
    assert fold.is_open()

    fold.button.click()
    repo2, _ = _repo(store)
    _header, again = group_fold("paper_curve_shape", "Curve Shape", CompactSlider("Glow", 0, 1, 0), default_open=True)
    again.bind(repo2)
    assert not again.is_open()


def test_a_warning_marks_a_closed_fold_amber_and_an_edit_outranks_it():
    repo, _ = _repo()
    _section, _grade, shadows, fold = _grade_card(repo)

    fold.set_warning(True)
    assert fold.button.is_marked() and fold.button._mark_color == THEME.warn_amber

    shadows.setValue(-15)
    assert fold.button._mark_color == THEME.mark_edited

    fold.set_open(True)
    assert not fold.button.is_marked()


def test_the_analysis_stats_report_a_warning(qapp):
    from negpy.desktop.view.widgets.stats import NegativeStatsWidget
    from negpy.features.exposure.stats import StatRow

    stats = NegativeStatsWidget()
    seen = []
    stats.warning_changed.connect(seen.append)
    stats.update_stats([StatRow("Clipping", "Sh 3.0%", warn=True)])
    stats.update_stats([StatRow("Clipping", "Sh 0.0%")])
    assert seen == [True, False]
