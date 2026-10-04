from negpy.desktop.view.widgets.choice_button import ChoiceButton

_CHOICES = (("fa5s.globe", "Global"), ("fa5s.moon", "Shadows"), ("fa5s.sun", "Highlights"))


def test_choosing_from_the_menu_changes_the_button_and_emits(qapp):
    btn = ChoiceButton(_CHOICES, "tip")
    seen = []
    btn.currentChanged.connect(seen.append)

    btn.choice_menu.actions()[2].trigger()

    assert btn.currentIndex() == 2
    assert btn.text().strip() == "Highlights"
    assert btn.choice_menu.actions()[2].isChecked()
    assert seen == [2]

    btn.setCurrentIndex(2)
    assert seen == [2]


def test_edited_marks_the_menu_item_and_the_button_for_the_current_choice(qapp):
    btn = ChoiceButton(_CHOICES, "tip")

    btn.set_edited(1, True)
    assert btn.choice_menu.actions()[1].text() == "Shadows\t•"
    assert btn.edited_dot.isHidden()

    btn.setCurrentIndex(1)
    assert not btn.edited_dot.isHidden()

    btn.set_edited(1, False)
    assert btn.choice_menu.actions()[1].text() == "Shadows"
    assert btn.edited_dot.isHidden()


def test_scroll_wheel_leaves_the_choice_and_scrolls_the_panel(qapp):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent

    btn = ChoiceButton(_CHOICES, "tip")
    event = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    btn.wheelEvent(event)
    assert btn.currentIndex() == 0
    assert not event.isAccepted()


def test_toggle_menu_button_is_checked_while_any_option_is_on(qapp):
    from negpy.desktop.view.widgets.choice_button import ToggleMenuButton

    btn = ToggleMenuButton("fa5s.magic", "Auto", "tip")
    a = btn.add_toggle("A", "a")
    b = btn.add_toggle("B", "b")
    assert not btn.isChecked()

    a.trigger()
    assert btn.isChecked()
    b.trigger()
    a.trigger()
    assert btn.isChecked()
    b.trigger()
    assert not btn.isChecked()

    btn.nextCheckState()  # what a click runs besides opening the menu
    assert not btn.isChecked()


def test_tool_toggle_centers_by_default_and_left_aligns_on_request(qapp):
    from negpy.desktop.view.styles.templates import tool_toggle

    assert "text-align" not in tool_toggle("fa5s.magic", "Label", "tip").styleSheet()
    assert "text-align: left" in tool_toggle("fa5s.magic", "Label", "tip", align_left=True).styleSheet()


def test_choices_can_be_replaced_and_carry_values(qapp):
    btn = ChoiceButton((), "tip")
    assert btn.count() == 0 and btn.currentData() is None

    btn.set_choices((("", "8-bit"), ("", "16-bit")), data=(8, 16))
    assert [a.text() for a in btn.choice_menu.actions()] == ["8-bit", "16-bit"]
    assert btn.currentData() == 8 and btn.findData(16) == 1 and btn.findData(14) == -1

    btn.setCurrentIndex(1)
    btn.set_choices((("", "14-bit"),), data=(14,))
    assert btn.count() == 1 and btn.currentData() == 14


def test_a_choice_can_be_disabled(qapp):
    btn = ChoiceButton(_CHOICES, "tip")
    btn.set_choice_enabled(1, False)
    assert not btn.is_choice_enabled(1)
    assert not btn.choice_menu.actions()[1].isEnabled()


def test_the_click_that_dismissed_the_menu_does_not_reopen_it(qapp, monkeypatch):
    btn = ChoiceButton(_CHOICES, "tip")
    opened = []
    monkeypatch.setattr(btn.choice_menu, "exec", lambda *_: opened.append(True))

    # A press on the button while the menu is open: Qt hides the menu, then emits clicked.
    monkeypatch.setattr(btn, "_dismissed_by_press_on_button", lambda: True)
    btn.choice_menu.aboutToHide.emit()
    btn.clicked.emit()
    assert opened == []

    btn.clicked.emit()
    assert opened == [True]


def test_a_hide_not_caused_by_the_button_leaves_the_next_click_live(qapp, monkeypatch):
    btn = ChoiceButton(_CHOICES, "tip")
    opened = []
    monkeypatch.setattr(btn.choice_menu, "exec", lambda *_: opened.append(True))

    monkeypatch.setattr(btn, "_dismissed_by_press_on_button", lambda: False)
    btn.choice_menu.aboutToHide.emit()
    btn.clicked.emit()
    assert opened == [True]


def test_a_release_off_the_button_disarms_the_swallow(qapp, monkeypatch):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    btn = ChoiceButton(_CHOICES, "tip")
    opened = []
    monkeypatch.setattr(btn.choice_menu, "exec", lambda *_: opened.append(True))

    # Armed by the dismissing press; the release lands off the button, so no clicked fires.
    monkeypatch.setattr(btn, "_dismissed_by_press_on_button", lambda: True)
    btn.choice_menu.aboutToHide.emit()
    release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(-10.0, -10.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    btn.mouseReleaseEvent(release)

    btn.clicked.emit()
    assert opened == [True]
