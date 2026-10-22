from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ThemeConfig:
    """
    Centralized UI styling constants.
    """

    # Fonts (families are resolved at runtime in styles/fonts.py). Four roles, in px:
    # small = caption/hint, base = body (also the QSS global), header = section, title = dialog.
    font_size_small: int = 12
    font_size_base: int = 13
    font_size_header: int = 14
    font_size_title: int = 16
    # Chart axis and swatch captions only: the one size below the reading scale.
    font_size_micro: int = 9

    # Colors. Surfaces, darkest first.
    bg_dark: str = "#0D0D0D"
    bg_panel: str = "#0D0D0D"
    bg_input: str = "#0A0A0A"
    bg_card: str = "#121212"
    bg_header: str = "#161616"
    bg_selected: str = "#222222"
    bg_hover: str = "#262626"
    bg_menu_selected: str = "#2E2E2E"
    # A checked toggle and a segmented control's current choice; also the Scan button's tint
    # while it cannot start or while it reads Stop.
    bg_checked: str = "#2A1D1D"
    bg_checked_hover: str = "#351F1F"
    border_checked: str = "#4A2424"
    border_checked_hover: str = "#5E2A2A"
    # The primary action: the strongest button on a panel.
    bg_primary: str = "#4A1C1C"
    bg_primary_hover: str = "#5A2121"
    # An edited card's frame-scope stripe and lit Frame icon; a roll-scope card takes border_checked_hover.
    border_scope_frame: str = "#5D4B24"
    # The dot on an edited control: red, a step below the R channel's red.
    mark_edited: str = "#9C3B3B"
    bg_accent_muted: str = "#1A1414"
    border_input: str = "#1A1A1A"
    border_primary: str = "#262626"
    border_color: str = "#333333"
    border_hover: str = "#3A3A3A"
    # Keyboard focus: a neutral edge, as hover and the arrow-key row are neutral.
    border_focus: str = "#949494"
    border_indicator: str = "#5A5A5A"
    slider_disabled: str = "#555555"
    text_on_accent: str = "#FFFFFF"
    text_primary: str = "#E4E4E4"
    text_secondary: str = "#B4B4B4"
    # hint = readable secondary copy (6.4:1 on bg_panel); muted = disabled widgets only, and
    # too dim (3.6:1) for text a user must read.
    text_hint: str = "#949494"
    text_muted: str = "#6B6B6B"
    text_unit: str = "#7C7C7C"
    warn_amber: str = "#C79A3A"
    # Marks drawn over the photo: white on a dark edge or plate, which reads on any picture.
    mark_photo: str = "#FFFFFF"
    # The dark edge or plate under a photo mark; each site sets its own alpha (templates.with_alpha).
    mark_shadow: str = "#000000"
    # Errors and invalid input.
    error: str = "#D32F2F"

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
    surface_reveal_flash: str = "rgba(94, 42, 42, 0.35)"  # border_checked_hover, faint
    surface_hover_faint: str = "#0CFFFFFF"  # #AARRGGBB: QColor reads it, not rgba()
    surface_overlay: str = "rgba(13,13,13,0.88)"
    # A card header over its chart: light enough for the chart to read through.
    header_veil: str = "rgba(26,26,26,0.55)"
    header_veil_hover: str = "rgba(34,34,34,0.62)"
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
    # Contrast Mask band on the Analysis chart.
    chart_mask: str = "#AF8CFF"
    # Film-mode chips: orange mask, silver grey, slide blue.
    mode_c41: str = "#E08A3C"
    mode_bw: str = "#8C8C8C"
    mode_e6: str = "#4FB0D8"

    # Canvas background swatches
    canvas_bg_black: str = "#050505"
    canvas_bg_dark_grey: str = "#1C1C1C"
    canvas_bg_mid_grey: str = "#404040"
    canvas_bg_light_grey: str = "#A0A0A0"
    canvas_bg_white: str = "#E6E6E6"

    # Status semantic
    status_success: str = "#1D9E75"

    sidebar_expanded_defaults: Dict[str, bool] = field(
        default_factory=lambda: {
            "analysis": True,
            "baseline": True,
            "process": False,
            "color": True,
            "tone": True,
            "paper": True,
            "geometry": True,
            "autocrop": True,
            "optics": False,
            "assembly": False,
            "lab": True,
            "retouch": True,
        }
    )


THEME = ThemeConfig()
# Scene marks cycle through existing tokens; red is left out because it reads as an error or the selection mat.
SCENE_COLORS = (THEME.channel_blue, THEME.status_success, THEME.warn_amber, THEME.filter_magenta, THEME.filter_cyan, THEME.mode_c41)


def scene_color(ordinal: int) -> str:
    return SCENE_COLORS[(ordinal - 1) % len(SCENE_COLORS)]
