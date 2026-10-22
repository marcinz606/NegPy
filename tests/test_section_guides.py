"""Every sidebar section has a guide slice, and every guide marker names a live section."""

import ast
from pathlib import Path

from negpy.desktop.view.widgets.section_help_dialog import _guides

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "negpy" / "desktop" / "view"

# Keys built with an f-string (metadata cards), and the help buttons of the strip-preview dialog and the Roll row.
DYNAMIC_KEYS = {
    "metadata_gear",
    "metadata_capture",
    "metadata_process",
    "metadata_scanning",
    "metadata_exposure",
    "metadata_preview",
    "scan_strip",
    "frames",
}


def _section_keys() -> set[str]:
    keys: set[str] = set()
    for path in DESKTOP.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            index = {"make_section": 2, "_make_section": 1}.get(name)
            if index is not None and len(node.args) > index:
                key = node.args[index]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys | DYNAMIC_KEYS


def test_every_section_has_a_guide_and_every_marker_a_section():
    guides = _guides()
    sections = _section_keys()
    assert sections, "no make_section calls found"
    assert sections - set(guides) == set(), "sections whose ⓘ would be missing"
    assert set(guides) - sections == set(), "markers no section header uses"


def test_slices_carry_no_marker_comments():
    for key, body in _guides().items():
        assert "<!-- panel:" not in body, key
