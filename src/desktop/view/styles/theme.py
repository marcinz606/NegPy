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
    bg_dark: str = "#0f0f0f"
    bg_panel: str = "#1a1a1a"
    bg_header: str = "#2a2a2a"
    bg_status_bar: str = "#0a0a0a"
    border_primary: str = "#222222"
    border_color: str = "#333333"
    text_primary: str = "#eeeeee"
    text_secondary: str = "#aaaaaa"
    text_muted: str = "#555555"
    accent_primary: str = "#b10000"
    accent_secondary: str = "#d10000"

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
