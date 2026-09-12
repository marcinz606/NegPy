from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ThemeConfig:
    """
    Centralized UI styling constants.
    """

    # Fonts (families are resolved at runtime in styles/fonts.py). Four roles, in px:
    # small = caption/hint, base = body (also the QSS global), header = section, title = dialog.
    # display is the sidebar wordmark alone.
    font_size_small: int = 12
    font_size_base: int = 13
    font_size_header: int = 14
    font_size_title: int = 16
    font_size_display: int = 24
    # Chart axis and swatch captions only: the one size below the reading scale.
    font_size_micro: int = 9

    # Colors. Surfaces, darkest first.
    bg_dark: str = "#0D0D0D"
    bg_panel: str = "#0D0D0D"
    bg_input: str = "#0A0A0A"
    bg_card: str = "#121212"
    bg_header: str = "#161616"
    bg_status_bar: str = "#0a0a0a"
    bg_selected: str = "#222222"
    bg_hover: str = "#262626"
    bg_menu_selected: str = "#2E2E2E"
    # The checked-toggle tint: warm, so an engaged tool reads apart from a hovered one.
    bg_checked: str = "#2A1D1D"
    bg_checked_hover: str = "#351F1F"
    bg_accent_muted: str = "#1A1414"
    border_input: str = "#1A1A1A"
    border_primary: str = "#262626"
    border_color: str = "#333333"
    border_hover: str = "#3A3A3A"
    border_indicator: str = "#5A5A5A"
    border_indicator_hover: str = "#8A8A8A"
    bg_indicator: str = "#1A1A1A"
    bg_indicator_hover: str = "#202020"
    bg_indicator_disabled: str = "#141414"
    slider_disabled: str = "#555555"
    slider_fill_disabled: str = "#2A2A2A"
    text_on_accent: str = "#FFFFFF"
    text_primary: str = "#E4E4E4"
    text_secondary: str = "#B4B4B4"
    # hint = readable secondary copy (6.4:1 on bg_panel); muted = disabled widgets only, and
    # too dim (3.6:1) for text a user must read.
    text_hint: str = "#949494"
    text_muted: str = "#6B6B6B"
    text_unit: str = "#7C7C7C"
    warn_amber: str = "#C79A3A"
    accent_primary: str = "#B71C1C"
    accent_secondary: str = "#C62828"
    accent_pressed: str = "#8E1414"
    # Errors and invalid input. Not the brand accent: red already means "selected" and "armed".
    error: str = "#D32F2F"

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

    # Surface overlay tokens (rgba). Integer alpha where a widget sheet reads it.
    surface_scrim: str = "rgba(5, 5, 5, 115)"
    surface_pill: str = "rgba(0, 0, 0, 140)"
    surface_hover_faint: str = "rgba(255, 255, 255, 12)"
    surface_overlay: str = "rgba(13,13,13,0.88)"
    surface_overlay_strong: str = "rgba(26,26,26,0.82)"
    surface_overlay_hover: str = "rgba(34,34,34,0.88)"
    # Canvas toast / loading chip plate. Integer alpha: QSS float alpha is not reliably parsed.
    surface_toast: str = "rgba(10, 10, 10, 225)"
    border_toast: str = "rgba(255, 255, 255, 55)"

    # Font weight scale
    weight_regular: int = 400
    weight_medium: int = 500
    weight_semibold: int = 600
    weight_bold: int = 700

    # Channel colors: fills (histogram, chart bands) and a lighter tier for text, dots and
    # slider handles, where the fill tier falls under 4.5:1 on the panel.
    channel_red: str = "#D32F2F"
    channel_green: str = "#388E3C"
    channel_blue: str = "#1976D2"
    channel_red_text: str = "#FF5A5A"
    channel_green_text: str = "#5ADC78"
    channel_blue_text: str = "#5F96FF"
    # Subtractive filtration sliders.
    filter_cyan: str = "#00B1B1"
    filter_magenta: str = "#B100B1"
    filter_yellow: str = "#B1B100"
    # Dodge & Burn: the same amber and blue on the overlay, the mask list and the printing notes.
    dodge: str = "#E8C84A"
    burn: str = "#4A8FE8"
    # Clipping and paper-limit warnings drawn on charts and the zone overlay.
    clip_warning: str = "#DC5050"
    # Film-mode chips: orange mask, silver grey, slide blue.
    mode_c41: str = "#E08A3C"
    mode_bw: str = "#8C8C8C"
    mode_e6: str = "#4FB0D8"

    # Canvas background swatches
    canvas_bg_black: str = "#050505"
    canvas_bg_dark_grey: str = "#1C1C1C"
    canvas_bg_mid_grey: str = "#404040"
    canvas_bg_white: str = "#FFFFFF"

    # Status semantic
    status_success: str = "#1D9E75"

    sidebar_expanded_defaults: Dict[str, bool] = field(
        default_factory=lambda: {
            "analysis": True,
            "process": True,
            "color": True,
            "tone": True,
            "geometry": True,
            "lab": True,
            "retouch": True,
        }
    )


THEME = ThemeConfig()
