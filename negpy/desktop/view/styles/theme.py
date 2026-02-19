from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ThemeConfig:
    """
    Centralized UI styling constants.
    """

    # Fonts
    font_family: str = "Inter, Segoe UI, Roboto, sans-serif"
    font_size_base: int = 12
    font_size_small: int = 12
    font_size_header: int = 13
    font_size_title: int = 16

    # Colors
    bg_dark: str = "#0D0D0D"
    bg_panel: str = "#0D0D0D"
    bg_header: str = "#161616"
    bg_status_bar: str = "#0a0a0a"
    border_primary: str = "#262626"
    border_color: str = "#333333"
    text_primary: str = "#D4D4D4"
    text_secondary: str = "#A0A0A0"
    text_muted: str = "#555555"
    text_unit: str = "#666666"
    accent_primary: str = "#B71C1C"
    accent_secondary: str = "#C62828"

    slider_height_compact: int = 18
    header_padding: int = 10

    sidebar_expanded_defaults: Dict[str, bool] = field(
        default_factory=lambda: {
            "analysis": True,
            "presets": False,
            "exposure": True,
            "geometry": True,
            "lab": True,
            "toning": False,
            "retouch": True,
            "icc": False,
            "export": True,
        }
    )


THEME = ThemeConfig()
