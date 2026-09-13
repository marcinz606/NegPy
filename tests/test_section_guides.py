"""Every sidebar section has a guide slice, and every guide marker names a live section."""

import re
from pathlib import Path

from negpy.desktop.view.widgets.section_help_dialog import _guides

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "negpy" / "desktop" / "view"

# Keys built with an f-string (metadata cards) and the strip-preview dialog's help button.
DYNAMIC_KEYS = {
    "metadata_presets",
    "metadata_gear",
    "metadata_capture",
    "metadata_process",
    "metadata_scanning",
    "metadata_exposure",
    "metadata_preview",
    "scan_strip",
}


def _section_keys() -> set[str]:
    keys: set[str] = set()
    pattern = re.compile(r'make_section\(\s*(?:self\.controller\.session\.repo|self\.session\.repo|repo)?,?\s*"[^"]+",\s*"([a-z_]+)"', re.S)
    for path in DESKTOP.rglob("*.py"):
        keys.update(pattern.findall(path.read_text()))
    # controls_panel goes through its own thin wrapper; a comment may sit before the title.
    wrapper = re.compile(r'self\._make_section\(\s*(?:#[^\n]*\n\s*)*"[^"]+",\s*"([a-z_]+)"', re.S)
    keys.update(wrapper.findall((DESKTOP / "sidebar" / "controls_panel.py").read_text()))
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
