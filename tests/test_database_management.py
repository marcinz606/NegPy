from dataclasses import replace

from negpy.domain.models import ExportPreset, WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository


def _repo(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


def _seed(repo):
    """Populate every category the stats/clear methods touch."""
    cfg = WorkspaceConfig()
    repo.save_file_settings("hash-a", cfg, file_path="/photos/a.raw")
    repo.save_file_settings("hash-b", cfg, file_path="/photos/b.raw")
    repo.save_history_step("hash-a", 0, cfg)
    repo.save_history_step("hash-a", 1, replace(cfg))
    repo.save_file_mark("hash-a", "keeper")
    repo.save_normalization_roll("roll-1", (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
    repo.save_flatfield_profile("rig-1", "/refs/flat.dng", k1=-0.05)
    repo.save_export_presets([ExportPreset(name="mine")])
    repo.save_global_setting("window_geometry", [0, 0, 800, 600])
    repo.save_global_setting("last_open_folder", "/photos")


def test_database_stats_counts_each_category(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    stats = repo.database_stats()

    assert stats["file_settings"] == 2
    assert stats["edit_history"] == 2
    assert stats["file_marks"] == 1
    assert stats["normalization_rolls"] == 1
    assert stats["flatfield_profiles"] == 1
    assert stats["export_presets"] == 1
    # export_presets is one row inside global_settings; it must not inflate the
    # preferences count, which here is the two real settings we saved.
    assert stats["app_preferences"] == 2
    assert stats["edits_db_bytes"] > 0
    assert stats["settings_db_bytes"] > 0


def test_stats_on_empty_db_are_all_zero(tmp_path):
    repo = _repo(tmp_path)
    stats = repo.database_stats()
    for key in (
        "file_settings",
        "edit_history",
        "file_marks",
        "normalization_rolls",
        "flatfield_profiles",
        "export_presets",
        "app_preferences",
    ):
        assert stats[key] == 0


def test_clear_saved_edits_drops_only_per_image_data(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)

    repo.clear_saved_edits()
    stats = repo.database_stats()

    # Per-image data gone.
    assert stats["file_settings"] == 0
    assert stats["edit_history"] == 0
    assert stats["file_marks"] == 0
    # Tooling kept.
    assert stats["normalization_rolls"] == 1
    assert stats["flatfield_profiles"] == 1
    assert stats["export_presets"] == 1
    assert stats["app_preferences"] == 2


def test_clear_saved_edits_makes_reloaded_image_start_fresh(tmp_path):
    """The user's actual goal: after clearing, loading the same hash returns nothing,
    so hydration falls back to defaults instead of the previous look."""
    repo = _repo(tmp_path)
    edited = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, narrowband_scan=True))
    repo.save_file_settings("hash-x", edited, file_path="/photos/x.raw")
    assert repo.load_file_settings("hash-x") is not None

    repo.clear_saved_edits()

    assert repo.load_file_settings("hash-x") is None
    # Path-based fallback must not resurrect it either.
    assert repo.load_file_settings_by_path("/photos/x.raw") is None


def test_reset_everything_wipes_both_databases(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)

    repo.reset_everything()
    stats = repo.database_stats()

    for key in (
        "file_settings",
        "edit_history",
        "file_marks",
        "normalization_rolls",
        "flatfield_profiles",
        "export_presets",
        "app_preferences",
    ):
        assert stats[key] == 0
    # A global setting written before the reset is gone.
    assert repo.get_global_setting("window_geometry") is None


def test_repository_still_usable_after_reset(tmp_path):
    """Reset empties rows but preserves schema — the app must keep working."""
    repo = _repo(tmp_path)
    _seed(repo)
    repo.reset_everything()

    repo.save_file_settings("hash-new", WorkspaceConfig(), file_path="/photos/new.raw")
    assert repo.load_file_settings("hash-new") is not None
    assert repo.database_stats()["file_settings"] == 1


def test_clear_on_empty_db_is_a_noop(tmp_path):
    repo = _repo(tmp_path)
    repo.clear_saved_edits()  # must not raise
    repo.reset_everything()  # must not raise
    assert repo.database_stats()["file_settings"] == 0
