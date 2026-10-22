from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "negpy"


def test_no_radio_buttons():
    """Exclusive choices are SegmentedChoice or StepCombo, in panels and dialogs alike."""
    offenders = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py") if "QRadioButton" in p.read_text(encoding="utf-8")]
    assert offenders == []
