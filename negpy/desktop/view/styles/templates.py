import html

import qtawesome as qta
from PyQt6.QtWidgets import QLabel, QPushButton

from negpy.desktop.view.styles.theme import THEME

_default_btn_height: int | None = None


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


def wrap_tooltip(text: str) -> str:
    """Plain-text tooltips never word-wrap in Qt; rich text does. Wrap in <qt> so
    long tooltips break into lines instead of spanning the screen. Text that
    already carries markup (e.g. tooltip_with_shortcut's chips) must pass through
    unescaped or its tags render as literal text."""
    if text.startswith("<qt>"):
        return text
    if "<" in text and ">" in text:
        return f"<qt>{text}</qt>"
    return f"<qt>{html.escape(text)}</qt>"


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


def labeled_toggle_qss(color: str | None = None) -> str:
    """Segmented/selector toggle (channel rows, intent rows): base type, 8px
    padding; optional text colour carries the edited-state tint."""
    suffix = f" color: {color};" if color else ""
    return f"font-size: {THEME.font_size_base}px; padding: 8px;{suffix}"


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
    """Padding tweaks for canvas-tool toggles. The armed (checked) look itself
    comes from the app-wide QPushButton:checked rule in modern_dark.qss."""
    return "QPushButton {padding: 6px;}" if icon_only else ""


def slider_label_qss(color: str, edited: bool) -> str:
    label_color = THEME.accent_edited if edited else color
    return f"font-size: {THEME.font_size_base}px; color: {label_color};"


def slider_handle_qss(color: str) -> str:
    """Recolors the handle only; geometry cascades from the app-wide QSlider style."""
    return f"QSlider::handle:horizontal {{background: {color};}}"


def swatch_qss(hex_col: str) -> str:
    return (
        f"QToolButton {{background-color: {hex_col}; border: 1px solid #444; border-radius: 3px;}}"
        f" QToolButton:checked {{border: 2px solid {THEME.text_muted};}}"
        f" QToolButton:hover {{border: 1px solid #888;}}"
    )
