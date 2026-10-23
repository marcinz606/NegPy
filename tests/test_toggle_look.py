from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPalette
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QMenu,
    QProgressBar,
    QStyle,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QWidget,
)

from negpy.desktop.view.styles.templates import (
    header_row,
    icon_button,
    labeled_action,
    load_stylesheet,
    section_subheader,
    themed_palette,
    tool_toggle,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.segmented_choice import ToggleMenuButton
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.flat_rows import paint_fill
from negpy.desktop.view.widgets.step_combo import StepCombo


def _fill_at_rest_and_under_mouse(qapp, button) -> tuple[str, str]:
    """Paints the button through its style with and without State_MouseOver: a real pointer
    move is unreliable once other tests leave windows open."""
    host = QWidget()
    host.setStyleSheet(load_stylesheet())
    QHBoxLayout(host).addWidget(button)
    host.resize(240, 60)
    button.ensurePolished()
    fills = []
    for hovered in (False, True):
        option = QStyleOptionButton()
        button.initStyleOption(option)
        if hovered:
            option.state |= QStyle.StateFlag.State_MouseOver
        image = _paint(
            button.width(), button.height(), lambda p: button.style().drawControl(QStyle.ControlElement.CE_PushButton, option, p, button)
        )
        fills.append(image.pixelColor(3, button.height() // 2).name())
    return fills[0], fills[1]


def test_hover_on_a_toggle_that_is_off_draws_no_fill(qapp):
    """The fill means checked."""
    for button in (tool_toggle("fa5s.circle", "Oval", ""), ToggleMenuButton("fa5s.magic", "", "")):
        rest, hover = _fill_at_rest_and_under_mouse(qapp, button)
        assert rest == hover, type(button).__name__


def test_hover_on_an_action_fills(qapp):
    rest, hover = _fill_at_rest_and_under_mouse(qapp, labeled_action("fa5s.undo", "Reset", ""))
    assert rest != hover, (rest, hover)


def _paint(width: int, height: int, paint) -> QImage:
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(QColor(THEME.bg_panel))
    p = QPainter(image)
    paint(p)
    p.end()
    return image


def test_the_current_row_takes_the_chip_and_no_bar(qapp):
    """History, Library and preset rows: the chip of a checked toggle marks the current one."""
    image = _paint(200, 40, lambda p: paint_fill(p, QRect(0, 0, 200, 40), current=True, hover=False))
    names = {image.pixelColor(x, 20).name() for x in range(6)}

    assert image.pixelColor(100, 20).name() == QColor(THEME.bg_checked).name()
    assert QColor(THEME.border_checked).name() in names


def test_a_checked_menu_item_takes_the_chip(qapp):
    menu = QMenu()
    menu.setStyleSheet(load_stylesheet())
    menu.addAction("Off").setCheckable(True)
    on = menu.addAction("On")
    on.setCheckable(True)
    on.setChecked(True)
    menu.popup(QPoint(0, 0))
    QTest.qWait(20)
    image = menu.grab().toImage()
    row = menu.actionGeometry(on)
    colors = {image.pixelColor(x, row.center().y()).name() for x in range(row.left(), row.right())}
    menu.close()

    assert QColor(THEME.bg_checked).name() in colors


def test_the_current_dropdown_row_takes_the_chip_and_lightens_on_hover(qapp):
    combo = StepCombo()
    combo.addItems(["Digital", "Paper"])
    combo.setCurrentIndex(1)
    delegate = combo.view().itemDelegate()

    def fill(state) -> str:
        option = QStyleOptionViewItem()
        option.rect = QRect(0, 0, 200, 24)
        option.state = QStyle.StateFlag.State_Enabled | state
        return _paint(200, 24, lambda p: delegate.paint(p, option, combo.model().index(1, 0))).pixelColor(3, 12).name()

    assert fill(QStyle.StateFlag.State_None) == QColor(THEME.bg_checked).name()
    assert fill(QStyle.StateFlag.State_Selected) == QColor(THEME.bg_checked_hover).name()


def test_an_edited_card_takes_a_dim_scope_stripe(qapp):
    for scope, token in (("frame", THEME.border_scope_frame), ("roll", THEME.border_checked_hover)):
        section = CollapsibleSection("Calibration")
        section.setStyleSheet(load_stylesheet())
        section.set_modified(1)
        section.set_scope_buttons(visible=True, scope=scope)
        header = section.toggle_button.grab().toImage()

        assert header.pixelColor(1, header.height() // 2).name() == QColor(token).name(), scope


def test_a_primary_button_is_a_dark_red_fill(qapp):
    host = QWidget()
    host.setStyleSheet(load_stylesheet())
    button = labeled_action("fa5s.check", "Export", "", primary=True)
    QHBoxLayout(host).addWidget(button)
    host.resize(240, 60)
    image = host.grab().toImage()
    probe = button.geometry().topLeft() + QPoint(4, button.height() // 2)

    assert image.pixelColor(probe).name() == QColor(THEME.bg_primary).name()


def test_a_progress_bar_fills_with_the_chip(qapp):
    bar = QProgressBar()
    bar.setStyleSheet(load_stylesheet())
    bar.setTextVisible(False)
    bar.resize(200, 20)
    bar.setValue(50)
    image = bar.grab().toImage()

    assert image.pixelColor(50, 10).name() == QColor(THEME.bg_checked_hover).name()
    assert image.pixelColor(150, 10).name() == QColor(THEME.bg_input).name()


def test_a_button_sits_on_the_track_surface_in_a_subheader_too_and_a_flat_one_on_nothing(qapp):
    host = QWidget()
    host.setObjectName("host")
    host.setStyleSheet(load_stylesheet() + f"#host {{ background: {THEME.bg_card}; }}")
    boxed = labeled_action("fa5s.undo", "Reset", "")
    subheader = icon_button("fa5s.trash-alt", "Clear")
    header_row(section_subheader("Clone"), subheader)
    flat = icon_button("fa5s.undo", "Undo")
    flat.setFlat(True)
    layout = QHBoxLayout(host)
    for w in (boxed, subheader, flat):
        layout.addWidget(w)
    host.resize(400, 60)
    image = host.grab().toImage()

    def at(w, x):
        return image.pixelColor(w.geometry().left() + x, w.geometry().center().y()).name()

    track = (QColor(THEME.border_primary).name(), QColor(THEME.bg_input).name())
    assert (at(boxed, 0), at(boxed, 4)) == track
    assert (at(subheader, 0), at(subheader, 4)) == track
    assert at(flat, 0) == QColor(THEME.bg_card).name()


def test_selected_text_and_links_take_no_qt_blue(qapp):
    palette = themed_palette(QPalette())

    assert palette.color(QPalette.ColorRole.Highlight).name() == QColor(THEME.bg_checked_hover).name()
    for role in (QPalette.ColorRole.Link, QPalette.ColorRole.LinkVisited, QPalette.ColorRole.HighlightedText):
        assert palette.color(role).name() == QColor(THEME.text_primary).name()


def _on_card(widget: QWidget, width: int = 300) -> tuple[QWidget, QImage]:
    host = QWidget()
    host.setObjectName("host")
    host.setStyleSheet(load_stylesheet() + f"#host {{ background: {THEME.bg_card}; }}")
    QHBoxLayout(host).addWidget(widget)
    host.resize(width, 160)
    host.show()
    QTest.qWait(20)
    return host, host.grab().toImage()


def test_a_button_box_puts_cancel_before_ok(qapp):
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    _host, _image = _on_card(box)

    ok = box.button(QDialogButtonBox.StandardButton.Ok)
    assert box.button(QDialogButtonBox.StandardButton.Cancel).x() < ok.x()


def test_a_message_box_toggle_draws_as_a_labeled_toggle(qapp):
    from negpy.desktop.view.styles.templates import box_toggle

    toggle = box_toggle("Do Not Show Again")
    toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # the app's buttons take Tab focus only
    _host, image = _on_card(toggle)
    assert _edge_and_fill(image, toggle) == (QColor(THEME.border_primary).name(), QColor(THEME.bg_input).name())

    toggle.setChecked(True)
    host, _image = _on_card(toggle)
    assert _edge_and_fill(host.grab().toImage(), toggle)[1] == QColor(THEME.bg_checked).name()


def test_the_app_has_no_checkbox_but_the_message_box_toggle():
    """Every on/off choice is a toggle; Qt's message box alone needs a QCheckBox (templates.box_toggle)."""
    from pathlib import Path

    import negpy.desktop as desktop

    root = Path(desktop.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("QCheckBox(", "ItemIsUserCheckable"):
            if needle in text and not (path.name == "templates.py" and needle == "QCheckBox("):
                offenders.append(f"{path.relative_to(root)}: {needle}")
    assert offenders == []


def _edge_and_fill(image: QImage, widget: QWidget) -> tuple[str, str]:
    y = widget.geometry().center().y()
    return image.pixelColor(widget.geometry().left(), y).name(), image.pixelColor(widget.geometry().left() + 3, y).name()


def test_a_panel_tool_button_and_the_dropdown_faces_sit_on_the_track(qapp):
    from PyQt6.QtWidgets import QComboBox, QToolButton

    from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo

    boxed = QToolButton()
    boxed.setText("Import")
    boxed.setProperty("boxed", True)
    boxed.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # the app's buttons take Tab focus only
    gear = SearchableGearCombo(placeholder="Search cameras…")
    editable = QComboBox()
    editable.setEditable(True)
    editable.addItem("4000")
    editable.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    track = (QColor(THEME.border_primary).name(), QColor(THEME.bg_input).name())
    for widget in (boxed, editable):
        _host, image = _on_card(widget)
        assert _edge_and_fill(image, widget) == track, widget

    host, image = _on_card(gear)
    assert _edge_and_fill(image, gear)[0] == QColor(THEME.slider_disabled).name(), "the face shows the field's focus"
    gear._line.clearFocus()
    QTest.qWait(20)
    assert _edge_and_fill(host.grab().toImage(), gear) == track
    assert gear._arrow.text() == "" and not gear._arrow.icon().isNull()


def test_a_running_scan_takes_the_chip_of_a_toggle_that_is_on(qapp):
    from PyQt6.QtWidgets import QPushButton

    stop = QPushButton(" Stop")
    stop.setObjectName("scan_btn")
    stop.setProperty("scanning", "true")
    _host, image = _on_card(stop)

    assert image.pixelColor(stop.geometry().left() + 4, stop.geometry().center().y()).name() == QColor(THEME.bg_checked).name()


def test_keyboard_focus_is_a_neutral_gray_edge(qapp):
    from negpy.desktop.view.widgets.step_combo import StepCombo

    for widget in (labeled_action("fa5s.undo", "Reset", ""), StepCombo.of(["Print", "Flat"], "")):
        host, _image = _on_card(widget)
        host.activateWindow()
        widget.setFocus(Qt.FocusReason.TabFocusReason)
        QTest.qWait(20)
        image = host.grab().toImage()
        edge = {
            image.pixelColor(widget.geometry().left(), y).name() for y in range(widget.geometry().top() + 4, widget.geometry().bottom() - 4)
        }
        assert QColor(THEME.border_focus).name() in edge, type(widget).__name__


def test_a_tab_focused_slider_takes_the_edge_round_its_row_and_a_click_leaves_none(qapp):
    from negpy.desktop.view.widgets.sliders import CompactSlider

    row = CompactSlider("Print Density", 0.0, 2.0, 1.0)
    host, _image = _on_card(row)
    host.activateWindow()

    def edge() -> set:
        QTest.qWait(20)
        image = host.grab().toImage()
        g = row.geometry()
        return {image.pixelColor(g.left(), y).name() for y in range(g.top() + 4, g.bottom() - 4)}

    # Activation hands the first focusable widget its focus; tabbing arrives from elsewhere.
    row.slider.clearFocus()
    row.slider.setFocus(Qt.FocusReason.TabFocusReason)
    assert QColor(THEME.border_focus).name() in edge()
    row.slider.clearFocus()
    row.slider.setFocus(Qt.FocusReason.MouseFocusReason)
    assert QColor(THEME.border_focus).name() not in edge()


def test_no_theme_color_is_a_bright_red_but_errors_and_the_red_channel():
    """Selection, focus, hover and marks on the photo are muted or neutral; only an error and
    the red channel of RGB data stay bright red."""
    import dataclasses

    allowed = {"error", "error_text", "channel_red", "channel_red_text"}
    bright = []
    for field in dataclasses.fields(THEME):
        value = getattr(THEME, field.name)
        if not (isinstance(value, str) and value.startswith("#")) or field.name in allowed:
            continue
        color = QColor(value)
        hue = color.hslHueF()
        if (hue < 0.05 or hue > 0.95) and color.hslSaturationF() > 0.5 and color.lightnessF() > 0.3:
            bright.append(f"{field.name} {value}")
    assert bright == []
