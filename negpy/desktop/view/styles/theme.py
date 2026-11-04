import dataclasses
from dataclasses import dataclass, field
from typing import Dict

from negpy.kernel.system.i18n import mark


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
    # bg_dark is also the macOS window background (view/mac_window.py): an alternate theme updates that fix.
    bg_dark: str = "#323232"
    bg_panel: str = "#323232"
    bg_input: str = "#2F2F2F"
    bg_card: str = "#373737"
    bg_header: str = "#3B3B3B"
    bg_selected: str = "#484848"
    bg_hover: str = "#4C4C4C"
    bg_menu_selected: str = "#545454"
    # A checked toggle and a segmented control's current choice; also the Scan button's tint
    # while it cannot start or while it reads Stop.
    bg_checked: str = "#504242"
    bg_checked_hover: str = "#5B4544"
    border_checked: str = "#714A4A"
    border_checked_hover: str = "#865050"
    # The primary action: the strongest button on a panel.
    bg_primary: str = "#724241"
    bg_primary_hover: str = "#824747"
    # An edited card's frame-scope stripe and lit Frame icon; a roll-scope card takes border_checked_hover.
    border_scope_frame: str = "#5D4B24"
    # The dot on an edited control. A warning takes a sign in warn_amber, never a dot.
    mark_edited: str = "#C79A3A"
    # The canvas strip's job bar: the one warm red bright enough to read as progress.
    progress_fill: str = "#9C3B3B"
    bg_accent_muted: str = "#3F3939"
    border_input: str = "#3F3F3F"
    border_primary: str = "#4C4C4C"
    border_color: str = "#595959"
    border_hover: str = "#606060"
    # Keyboard focus: a neutral edge, as hover and the arrow-key row are neutral.
    border_focus: str = "#898989"
    border_indicator: str = "#696969"
    slider_disabled: str = "#7C7C7C"
    text_on_accent: str = "#FFFFFF"
    text_primary: str = "#E4E4E4"
    text_secondary: str = "#BCBCBC"
    # hint = readable secondary copy (5.1:1 on bg_panel); muted = disabled widgets only, and
    # too dim (3.0:1) for text a user must read.
    text_hint: str = "#A4A4A4"
    text_muted: str = "#7A7A7A"
    text_unit: str = "#8B8B8B"
    warn_amber: str = "#C79A3A"
    # Marks drawn over the photo: white on a dark edge or plate, which reads on any picture.
    mark_photo: str = "#FFFFFF"
    # The dark edge or plate under a photo mark; each site sets its own alpha (templates.with_alpha).
    mark_shadow: str = "#000000"
    # Errors and invalid input. Error copy takes the lighter tier, as the channel colors do: error
    # falls under 4.5:1 on a card, and a lighter error would dim the white mark on a failed-frame badge.
    error: str = "#D32F2F"
    error_text: str = "#FF6868"

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
    surface_reveal_flash: str = "rgba(134, 80, 80, 0.35)"  # border_checked_hover, faint
    surface_hover_faint: str = "#0CFFFFFF"  # #AARRGGBB: QColor reads it, not rgba()
    surface_overlay: str = "rgba(50,50,50,0.88)"
    # A card header over its chart: light enough for the chart to read through.
    header_veil: str = "rgba(63,63,63,0.55)"
    header_veil_hover: str = "rgba(72,72,72,0.62)"
    surface_overlay_hover: str = "rgba(72,72,72,0.88)"

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
    channel_red_text: str = "#FF6868"
    channel_green_text: str = "#5ADC78"
    channel_blue_text: str = "#6398FF"
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

    # Canvas background swatches, equal CIE L* steps from black to white.
    canvas_bg_black: str = "#070707"
    canvas_bg_dark_grey: str = "#3A3A3A"
    canvas_bg_mid_grey: str = "#6F6F6F"
    canvas_bg_light_grey: str = "#AAAAAA"
    canvas_bg_white: str = "#E8E8E8"

    # Status semantic
    status_success: str = "#33AD83"

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


# Preferences > Interface > UI Theme. Every "surfaces, darkest first" background and border
# lightened by the same offset that turns bg_dark into a photographic middle gray (Light Gray), so the
# depth ordering between panel, card and header is unchanged. The grays used for secondary
# and hint copy go white, since they read too close to the new light-gray surfaces otherwise;
# text_muted stays put, since it is documented as deliberately low-contrast (disabled only).
# header_veil/_hover and surface_overlay/_hover are translucent washes over a card's chart
# and over the canvas, not flat token reads, so each needs its own override beside bg_header.
_LIGHT_GRAY_OVERRIDES: Dict[str, str] = {
    "bg_dark": "#808080",
    "bg_panel": "#808080",
    "bg_input": "#7D7D7D",
    "bg_card": "#858585",
    "bg_header": "#898989",
    "bg_selected": "#959595",
    "bg_hover": "#999999",
    "bg_menu_selected": "#A1A1A1",
    "border_input": "#8D8D8D",
    "border_primary": "#999999",
    "border_color": "#A6A6A6",
    "border_hover": "#ADADAD",
    "border_indicator": "#B5B5B5",
    "slider_disabled": "#C8C8C8",
    "surface_overlay": "rgba(128,128,128,0.88)",
    "surface_overlay_hover": "rgba(149,149,149,0.88)",
    "header_veil": "rgba(141,141,141,0.55)",
    "header_veil_hover": "rgba(149,149,149,0.62)",
    "text_secondary": "#FFFFFF",
    "text_hint": "#FFFFFF",
    "text_unit": "#FFFFFF",
}

# Same lightening, anchored to canvas_bg_mid_grey (the "Mid Gray" canvas swatch). Still dark
# enough that the existing text grays stay readable, so none of them need the white override above.
_MID_GRAY_OVERRIDES: Dict[str, str] = {
    "bg_dark": "#404040",
    "bg_panel": "#404040",
    "bg_input": "#3D3D3D",
    "bg_card": "#454545",
    "bg_header": "#494949",
    "bg_selected": "#555555",
    "bg_hover": "#595959",
    "bg_menu_selected": "#616161",
    "border_input": "#4D4D4D",
    "border_primary": "#595959",
    "border_color": "#666666",
    "border_hover": "#6D6D6D",
    "border_indicator": "#757575",
    "slider_disabled": "#888888",
    "surface_overlay": "rgba(64,64,64,0.88)",
    "surface_overlay_hover": "rgba(85,85,85,0.88)",
    "header_veil": "rgba(77,77,77,0.55)",
    "header_veil_hover": "rgba(85,85,85,0.62)",
}

UI_THEMES: tuple[tuple[str, str], ...] = (
    ("black", mark("Black (Default)")),
    ("mid_gray", mark("Mid Gray")),
    ("light_gray", mark("Light Gray")),
)
_THEME_OVERRIDES: Dict[str, Dict[str, str]] = {
    "mid_gray": _MID_GRAY_OVERRIDES,
    "light_gray": _LIGHT_GRAY_OVERRIDES,
}
_DEFAULT_THEME = ThemeConfig()


def apply_theme(key: str) -> None:
    """Patches the shared THEME instance in place so every module that already holds a
    reference to it (imported at their own load time) picks up the chosen palette. Read once
    at startup; switching requires a restart to repaint every baked stylesheet and icon."""
    overrides = _THEME_OVERRIDES.get(key, {})
    for f in dataclasses.fields(THEME):
        object.__setattr__(THEME, f.name, overrides.get(f.name, getattr(_DEFAULT_THEME, f.name)))
