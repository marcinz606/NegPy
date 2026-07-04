from PyQt6.QtWidgets import QLabel

from negpy.desktop.view.styles.theme import THEME


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
