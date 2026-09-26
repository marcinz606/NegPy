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


def test_scroll_wheel_steps_through_enabled_choices_and_stops_at_the_ends(qapp):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent

    def wheel(btn, dy):
        btn.wheelEvent(
            QWheelEvent(
                QPointF(5, 5),
                QPointF(5, 5),
                QPoint(0, 0),
                QPoint(0, dy),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
        )

    btn = ChoiceButton(_CHOICES, "tip")
    btn.choice_menu.actions()[1].setEnabled(False)

    wheel(btn, -120)
    assert btn.currentIndex() == 2
    wheel(btn, -120)
    assert btn.currentIndex() == 2
    wheel(btn, 120)
    assert btn.currentIndex() == 0


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
