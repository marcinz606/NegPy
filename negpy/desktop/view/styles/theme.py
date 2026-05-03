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
    accent_edited: str = "#C79A3A"

    slider_height_compact: int = 18
    header_padding: int = 10

    # Spacing scale (px)
    space_xs: int = 2
    space_sm: int = 4
    space_md: int = 6
    space_lg: int = 8
    space_xl: int = 12
    space_2xl: int = 16

    # Radius scale (px)
    radius_sm: int = 3
    radius_md: int = 4
    radius_lg: int = 6

    # Surface overlay tokens (rgba)
    surface_overlay: str = "rgba(13,13,13,0.88)"
    surface_overlay_strong: str = "rgba(26,26,26,0.82)"
    surface_overlay_hover: str = "rgba(34,34,34,0.88)"

    # Font weight scale
    weight_regular: int = 400
    weight_medium: int = 500
    weight_semibold: int = 600
    weight_bold: int = 700

    # Font sizes (font_size_xs for status/captions, font_size_lg for toast)
    font_size_xs: int = 11
    font_size_lg: int = 15

    # Channel colors (histogram)
    channel_red: str = "#D32F2F"
    channel_green: str = "#388E3C"
    channel_blue: str = "#1976D2"

    # Canvas background swatches
    canvas_bg_black: str = "#050505"
    canvas_bg_dark_grey: str = "#1C1C1C"
    canvas_bg_mid_grey: str = "#404040"

    # Status semantic
    status_success: str = "#558B2F"

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
