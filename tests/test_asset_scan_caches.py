"""Mtime-stamped cache on the gear library.

It runs on the debounced sidebar resync; the cache must serve unchanged files
without disk reads and must pick up an on-disk change immediately.
"""

import json
import os

from negpy.services.assets.gear import GearProfiles


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
