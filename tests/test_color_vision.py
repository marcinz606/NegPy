from negpy.desktop.view.styles.color_vision import PALETTES
from negpy.desktop.view.styles.theme import THEME


def _luminance(hex_color: str) -> float:
    def channel(c: int) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def test_every_dodge_and_burn_color_reads_as_text_on_the_panel():
    """The mask list and the Printing Notes badges draw these as text on dark fills."""
    bg = _luminance(THEME.bg_panel)
    for palette in PALETTES:
        for color in palette.dodge_burn:
            assert (_luminance(color) + 0.05) / (bg + 0.05) >= 3.0, (palette.key, color)
