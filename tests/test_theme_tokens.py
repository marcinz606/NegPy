"""Every colour in the view layer is a theme token, so the palette lives in one file."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "negpy" / "desktop" / "view"
HEX = re.compile(r'#[0-9a-fA-F]{6}\b|"#[0-9a-fA-F]{3}"')
# The colour picker's own defaults are user data (the sheet's paper and ink), not chrome.
ALLOWED = {"theme.py", "contact_sheet_colors_dialog.py"}


def _offenders(pattern: re.Pattern) -> list[str]:
    found = []
    for path in VIEW.rglob("*.py"):
        if path.name in ALLOWED:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                found.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()[:90]}")
    return found


def test_no_literal_hex_colour_outside_the_theme():
    assert _offenders(HEX) == []


def test_the_sheet_reads_every_colour_from_a_token():
    qss = (VIEW / "styles" / "modern_dark.qss").read_text()
    assert re.findall(r"#[0-9a-fA-F]{3,6}\b", qss) == []


def test_no_literal_font_size_in_widgets():
    assert _offenders(re.compile(r"setPixelSize\(\d+\)|setPointSize\(\d+\)|font-size:\s*\d+px")) == []
