import dataclasses
import html

import qtawesome as qta
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QLabel, QProgressBar, QPushButton, QStackedLayout, QWidget

from negpy.desktop.view.styles.fonts import ui_font_family
from negpy.desktop.view.styles.theme import THEME

_default_btn_height: int | None = None


def load_stylesheet() -> str:
    """modern_dark.qss with @theme tokens and icon placeholders resolved."""
    from negpy.kernel.system.paths import get_resource_path

    qss_path = get_resource_path("negpy/desktop/view/styles/modern_dark.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        qss = f.read()
    # Longest token name first so a shorter one can't clobber its prefix.
    for f_ in sorted(dataclasses.fields(THEME), key=lambda f_: -len(f_.name)):
        value = getattr(THEME, f_.name)
        if isinstance(value, str):
            qss = qss.replace(f"@{f_.name}", value)
    # QSS url() cannot resolve relative paths reliably across dev and frozen runs, so bake in
    # the absolute icon path, with forward slashes for Qt.
    check_icon = get_resource_path("media/icons/checkbox_check.svg").replace("\\", "/")
    qss = qss.replace("__CHECKBOX_CHECK_ICON__", check_icon)
    return qss.replace("__UI_FONT__", ui_font_family())


def default_button_height() -> int:
    """Height a default-styled button renders at under the live app font/QSS —
    measured from a reference button, not hardcoded (a wrong constant makes
    mixed button rows stair-step)."""
    global _default_btn_height
    if _default_btn_height is None:
        ref = QPushButton(" Ref")
        ref.setIcon(qta.icon("fa5s.circle"))
        _default_btn_height = ref.sizeHint().height()
    return _default_btn_height


def wrap_tooltip(text: str, footer: str = "") -> str:
    """Plain-text tooltips never word-wrap in Qt; rich text does. Wrap in <qt> so
    long tooltips break into lines instead of spanning the screen. Text that
    already carries markup (e.g. tooltip_with_shortcut's chips) must pass through
    unescaped or its tags render as literal text.

    `footer` is trusted markup appended inside the <qt> document, so callers adding
    a boilerplate line don't have to re-implement the escape/passthrough rule."""
    if not text.startswith("<qt>"):
        body = text if ("<" in text and ">" in text) else html.escape(text)
        text = f"<qt>{body}</qt>"
    if footer:
        text = text.removesuffix("</qt>") + footer + "</qt>"
    return text


def hint_label(text: str = "", kind: str = "muted") -> QLabel:
    """Small informational label under a control. kind: "muted" | "warning" |
    "error" — styled by the QLabel[hint=...] rules in modern_dark.qss. Change
    kind at runtime with set_hint_kind (a plain setProperty won't repolish)."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setProperty("hint", kind)
    return lbl


def set_hint_kind(lbl: QLabel, kind: str) -> None:
    lbl.setProperty("hint", kind)
    style = lbl.style()
    style.unpolish(lbl)
    style.polish(lbl)


def pane_header_qss() -> str:
    """Bold mini-header for dialog panes (preset list / gear library columns)."""
    return f"color: {THEME.text_muted}; font-size: 10px; font-weight: bold; letter-spacing: 1px;"


def dialog_pane_qss() -> str:
    """Left column pane in two-pane dialogs: panel fill + right divider."""
    return f"background: {THEME.bg_panel}; border-right: 1px solid {THEME.border_primary};"


def labeled_toggle_qss() -> str:
    """Segmented/selector toggle (channel rows, intent rows): base type, 8px padding."""
    return f"font-size: {THEME.font_size_base}px; padding: 8px;"


class EditedDot(QLabel):
    """Red dot marking an edited (non-default) control. Standalone for layouts;
    pass overlay_on to pin it to a widget's top-right corner instead."""

    def __init__(self, overlay_on: QWidget | None = None, margin: int = 4) -> None:
        super().__init__(overlay_on)
        self._margin = margin
        self.setFixedSize(8, 8)
        self.setStyleSheet(f"background-color: {THEME.channel_red}; border-radius: 4px;")
        self.hide()
        if overlay_on is not None:
            overlay_on.installEventFilter(self)

    def set_active(self, active: bool) -> None:
        self.setVisible(active)
        if self.parent() is not None:
            self._reposition()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.parent() and event.type() == QEvent.Type.Resize:
            self._reposition()
        return False

    def _reposition(self) -> None:
        parent = self.parent()
        self.move(parent.width() - self.width() - self._margin, self._margin)


class StatusStrip(QWidget):
    """One fixed-height row for a surface's transient state: the pass that is running, the
    message it left behind, or the resting summary — whichever is current, in that order.

    The height is reserved once, at construction. That is the whole point: a scan surface
    that shows each of these in its own appearing row moves the button underneath them.
    """

    def __init__(self, parent: QWidget | None = None, lines: int = 2) -> None:
        super().__init__(parent)
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;")

        self._message = QLabel("")
        self._message.setWordWrap(True)
        self._message.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;")

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)

        for w in (self._summary, self._message, self._bar):
            self._stack.addWidget(w)
        self._stack.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.setFixedHeight(self._summary.fontMetrics().lineSpacing() * lines + THEME.space_lg)
        self._running = False
        self._show_current()

    def set_summary(self, markup: str) -> None:
        """The resting line. Rich text: callers weight the parts that matter."""
        self._summary.setText(markup)
        self._show_current()

    def set_message(self, text: str) -> None:
        """A result, a warning or an error. Empty falls back to the summary."""
        self._message.setText(text)
        self._message.setToolTip(text)  # the strip clips rather than grows; nothing is lost
        self._show_current()

    def message(self) -> str:
        return self._message.text()

    def showing(self) -> str:
        """Which role the row currently carries: "progress" | "message" | "summary"."""
        return {self._bar: "progress", self._message: "message"}.get(self._stack.currentWidget(), "summary")

    def start_progress(self, fmt: str) -> None:
        self._bar.setRange(0, 100)
        self._bar.setFormat(fmt)
        self._bar.setValue(0)
        self._running = True
        self._show_current()

    def set_progress_indeterminate(self, fmt: str) -> None:
        self._bar.setRange(0, 0)
        self._bar.setFormat(fmt)
        self._running = True
        self._show_current()

    def set_progress(self, fmt: str, fraction: float) -> None:
        if self._bar.maximum() == 0:
            self._bar.setRange(0, 100)
        self._bar.setFormat(fmt)
        self._bar.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        self._running = True
        self._show_current()

    def stop_progress(self) -> None:
        self._running = False
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._show_current()

    def _show_current(self) -> None:
        if self._running:
            self._stack.setCurrentWidget(self._bar)
        elif self._message.text():
            self._stack.setCurrentWidget(self._message)
        else:
            self._stack.setCurrentWidget(self._summary)


def section_subheader(text: str) -> QLabel:
    """Small all-caps label for section grouping in sidebars."""
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"font-size: {THEME.font_size_xs}px; "
        f"color: {THEME.text_muted}; "
        f"font-weight: {THEME.weight_semibold}; "
        f"margin-top: {THEME.space_xl}px;"
    )
    return lbl


def field_label_qss() -> str:
    """Style for labels sitting next to a combo/entry field (muted, semibold)."""
    return f"font-size: {THEME.font_size_base}px; color: {THEME.text_secondary}; font-weight: {THEME.weight_semibold};"


def field_label(text: str) -> QLabel:
    """Muted semibold label for a combo/entry field."""
    lbl = QLabel(text)
    lbl.setStyleSheet(field_label_qss())
    return lbl


def tool_toggle_qss(icon_only: bool = False) -> str:
    """Icon-only padding; the checked look is the app-wide rule in modern_dark.qss."""
    return "QPushButton {padding: 6px;}" if icon_only else ""


def slider_label_qss(color: str) -> str:
    return f"font-size: {THEME.font_size_base}px; color: {color};"


def slider_handle_qss(color: str) -> str:
    """Recolors the handle only; geometry cascades from the app-wide QSlider style."""
    return f"QSlider::handle:horizontal {{background: {color};}}"
