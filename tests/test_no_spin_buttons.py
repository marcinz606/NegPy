import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "negpy"


def test_every_spin_box_hides_its_buttons():
    """A number field is typed into; Qt's up/down arrows do not match the app's look."""
    offenders = []
    for p in ROOT.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        made = len(re.findall(r"\b(?:Q\w*SpinBox|_NoScrollSpinBox)\(\)", text))
        if made > text.count("ButtonSymbols.NoButtons"):
            offenders.append(str(p.relative_to(ROOT)))
    assert offenders == []
