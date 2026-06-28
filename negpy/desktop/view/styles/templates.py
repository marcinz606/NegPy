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


def slider_label_qss(color: str, edited: bool) -> str:
    label_color = THEME.accent_edited if edited else color
    return f"font-size: {THEME.font_size_base}px; color: {label_color};"


def hue_handle_qss(color: str) -> str:
    return (
        f"QSlider::groove:horizontal {{"
        f"background: {THEME.bg_header}; height: 6px; border-radius: 3px;}}"
        f" QSlider::handle:horizontal {{"
        f"background: {color}; width: 12px; height: 12px;"
        f"margin: -3px 0; border-radius: 6px; border: 2px solid rgba(0,0,0,0.5);}}"
    )


def swatch_qss(hex_col: str) -> str:
    return (
        f"QToolButton {{background-color: {hex_col}; border: 1px solid #444; border-radius: 3px;}}"
        f" QToolButton:checked {{border: 2px solid {THEME.text_muted};}}"
        f" QToolButton:hover {{border: 1px solid #888;}}"
    )
