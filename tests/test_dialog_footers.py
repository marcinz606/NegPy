"""Every dialog footer routes through the one Enter-target helper.

Qt hands "default" to whichever autoDefault button was clicked last (issue #997), so a footer
that pins nothing repeats the last click on Enter. The static walk keeps a new dialog from
shipping without the helper; the behaviour itself is covered by the templates tests.
"""

import re
from pathlib import Path

WIDGETS = Path(__file__).resolve().parents[1] / "negpy" / "desktop" / "view" / "widgets"


def _dialog_sources() -> list[Path]:
    return [p for p in WIDGETS.glob("*.py") if re.search(r"class \w+\(QDialog\)", p.read_text())]


def test_every_dialog_with_buttons_pins_its_default():
    missing = []
    for path in _dialog_sources():
        src = path.read_text()
        has_buttons = "QPushButton(" in src or "QDialogButtonBox(" in src
        pinned = "pin_dialog_default(" in src or "pin_button_box(" in src
        if has_buttons and not pinned:
            missing.append(path.name)
    assert not missing, f"footers without an Enter target: {missing}"


def test_no_dialog_hand_rolls_the_primary_look():
    """The filled button is the [primary] QSS rule; a local accent stylesheet drifts from it."""
    offenders = []
    for path in _dialog_sources():
        for line in path.read_text().splitlines():
            if "setStyleSheet(" in line and "accent_primary" in line and "background" in line:
                offenders.append(f"{path.name}: {line.strip()[:80]}")
    assert not offenders, offenders
