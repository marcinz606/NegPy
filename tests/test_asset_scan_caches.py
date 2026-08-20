"""Mtime-stamped caches on the gear library and contact-sheet template scans.

Both run on the debounced sidebar resync; the caches must serve unchanged files
without disk reads and must pick up an on-disk change immediately.
"""

import json
import os

from negpy.services.assets.gear import GearProfiles
from negpy.services.export.contact_sheet_templates import ContactSheetTemplates


def test_gear_library_cache_hits_and_invalidates(tmp_path, monkeypatch):
    monkeypatch.setattr(GearProfiles, "_gear_dir", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(GearProfiles, "_library_cache", None)

    first = GearProfiles.load_library()
    assert GearProfiles.load_library() is first

    cam_file = tmp_path / "cameras.json"
    cam_file.write_text(json.dumps([{"id": "c1", "make": "Nikon", "model": "F3"}]))
    os.utime(cam_file, ns=(2, 2))

    reloaded = GearProfiles.load_library()
    assert reloaded is not first
    assert any(c.model == "F3" for c in reloaded.cameras)
    assert GearProfiles.load_library() is reloaded


def test_template_scan_cache_hits_and_invalidates(tmp_path, monkeypatch):
    monkeypatch.setattr(ContactSheetTemplates, "_templates_dir", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(ContactSheetTemplates, "_scan_cache", None)

    assert ContactSheetTemplates.list_templates() == [ContactSheetTemplates.DEFAULT_NAME]

    (tmp_path / "grid.toml").write_text('name = "Grid"\n[layout]\ncell_px = 400\n')
    names = ContactSheetTemplates.list_templates()
    assert "Grid" in names

    cached = ContactSheetTemplates._scan_cache
    ContactSheetTemplates.list_templates()
    assert ContactSheetTemplates._scan_cache is cached
