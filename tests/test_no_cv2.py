"""negpy never imports cv2: negpy_core holds every kernel."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1] / "negpy"


def test_negpy_imports_no_cv2():
    pattern = re.compile(r"^\s*(import cv2|from cv2\b)", re.MULTILINE)
    offenders = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"cv2 imported in {offenders}; port the call to negpy_core"
