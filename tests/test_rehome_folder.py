"""Renaming a roll's folder on disk carries every path-keyed record with it."""

import os

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import composites, rolls
from negpy.services.assets.rehome import rehome_path_prefix


def test_every_path_keyed_store_follows_the_folder(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    old, new = str(tmp_path / "roll1"), str(tmp_path / "roll1-portra")
    a, b = os.path.join(old, "a.tif"), os.path.join(old, "b.tif")
    elsewhere = str(tmp_path / "roll10" / "c.tif")
    repo.save_file_settings("ha", WorkspaceConfig(), file_path=a)
    repo.save_file_settings("hc", WorkspaceConfig(), file_path=elsewhere)
    repo.save_file_mark("ha", "keeper", file_path=a)
    vroll = rolls.create_virtual_roll(repo, "picks", [a, elsewhere])
    repo.save_global_setting(composites.COMPOSITES_KEY, {a: {"kind": "stitch", "paths": [a, b], "hash": "hs"}})

    rehome_path_prefix(repo, old, new)

    assert repo.load_file_settings_by_path(os.path.join(new, "a.tif")) is not None
    assert repo.load_file_settings_by_path(elsewhere) is not None  # a sibling sharing the prefix stays
    assert os.path.join(new, "a.tif") in repo.load_file_marks_by_path()
    assert rolls.roll_for_id(repo, vroll)["member_paths"] == [os.path.join(new, "a.tif"), elsewhere]
    saved = composites.saved_composites(repo)
    assert list(saved) == [os.path.join(new, "a.tif")]
    assert saved[os.path.join(new, "a.tif")]["paths"] == [os.path.join(new, "a.tif"), os.path.join(new, "b.tif")]
