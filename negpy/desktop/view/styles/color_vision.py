"""Color vision: one Preferences choice, read by every view that tells things apart by hue alone.

A palette names roles, not hues, so a view asks for its role and gets colors that vision keeps.
"""

from dataclasses import dataclass

from negpy.desktop.view.styles.theme import THEME


@dataclass(frozen=True)
class VisionPalette:
    key: str
    label: str
    pair: tuple[str, str]  # two kinds of mark that must read apart over any image
    dodge_burn: tuple[str, str]  # Dodge & Burn masks: warm for dodge, cool for burn
    zone_ramp: tuple[str, ...]  # False Color zones I..IX: cool shadows, neutral V, warm highlights


# Standard is neon to read over any film; the others are Okabe-Ito colors on an axis that vision
# keeps. A pair keeps clear of the amber exclusion band, and every color reads as text on the panel.
# Zone ramps step in hue on standard vision and in lightness along a kept axis otherwise.
PALETTES: tuple[VisionPalette, ...] = (
    VisionPalette(
        "standard",
        "Standard",
        ("#39FF14", "#FF00FF"),
        (THEME.dodge, THEME.burn),
        ("#2D2DAF", "#1E69D7", "#00A5C3", "#28AF6E", "#8C8C8C", "#AFCD3C", "#F5D728", "#FA961E", "#F05F82"),
    ),
    VisionPalette(
        "protan_deutan",
        "Protanopia / deuteranopia (red-green)",
        ("#56B4E9", "#D55E00"),
        ("#E69F00", "#56B4E9"),
        ("#1B3A6B", "#0072B2", "#3F98D0", "#A6CEE8", "#8C8C8C", "#F7E8A0", "#F0E442", "#E69F00", "#A86400"),
    ),
    VisionPalette(
        "tritan",
        "Tritanopia (blue-yellow)",
        ("#D55E00", "#009E73"),
        ("#D55E00", "#009E73"),
        ("#004D40", "#009E73", "#4DB89A", "#A8DCC9", "#8C8C8C", "#F6C1B0", "#E88B6B", "#D55E00", "#8F3A00"),
    ),
    VisionPalette(
        "achromat",
        "Achromatopsia (no color)",
        ("#FFFFFF", "#8C8C8C"),
        ("#FFFFFF", "#8C8C8C"),
        ("#1E1E1E", "#3C3C3C", "#5A5A5A", "#737373", "#8C8C8C", "#A5A5A5", "#BEBEBE", "#D7D7D7", "#EBEBEB"),
    ),
)


def palette_for(key: str) -> VisionPalette:
    return next((p for p in PALETTES if p.key == key), PALETTES[0])
