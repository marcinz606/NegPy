from dataclasses import replace

import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls
from negpy.services.assets.migrations.roll_fields import migrate_new_roll_field_locks


@pytest.fixture
def repo(tmp_path):
    r = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    r.initialize()
    return r


def _config(**process):
    cfg = WorkspaceConfig()
    return replace(cfg, process=replace(cfg.process, **process))


def test_locks_only_the_cards_whose_new_fields_a_frame_changed(repo):
    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/a.tif", "/r/b.tif", "/r/c.tif"])
    repo.save_file_settings("a", _config(white_point_trim_red=0.02), file_path="/r/a.tif")
    repo.save_file_settings("b", _config(highlight_reconstruction=2), file_path="/r/b.tif")
    repo.save_file_settings("c", _config(analysis_buffer=0.1), file_path="/r/c.tif")

    migrate_new_roll_field_locks(repo)

    assert rolls.frame_override_cards(repo, roll, "a") == {"process"}
    assert rolls.frame_override_cards(repo, roll, "b") == {"demosaic"}
    assert rolls.frame_override_cards(repo, roll, "c") == set()


def test_locks_in_every_roll_that_holds_the_path(repo):
    first = rolls.create_virtual_roll(repo, "One", ["/r/a.tif"])
    second = rolls.create_virtual_roll(repo, "Two", ["/r/a.tif"])
    repo.save_file_settings("a", _config(black_point_offset=-0.03), file_path="/r/a.tif")

    migrate_new_roll_field_locks(repo)

    assert rolls.frame_override_cards(repo, first, "a") == {"process"}
    assert rolls.frame_override_cards(repo, second, "a") == {"process"}


def test_a_fork_locks_its_own_roll_under_the_unforked_hash(repo):
    own = rolls.create_virtual_roll(repo, "Own", ["/r/a.tif"])
    other = rolls.create_virtual_roll(repo, "Other", ["/r/a.tif"])
    repo.save_file_settings(rolls.roll_edit_hash("a", own), _config(white_point_offset=0.05), file_path="/r/a.tif")

    migrate_new_roll_field_locks(repo)

    assert rolls.frame_override_cards(repo, own, "a") == {"process"}
    assert rolls.frame_override_cards(repo, other, "a") == set()


def test_runs_once(repo):
    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/a.tif"])
    migrate_new_roll_field_locks(repo)
    repo.save_file_settings("a", _config(white_point_offset=0.05), file_path="/r/a.tif")

    migrate_new_roll_field_locks(repo)

    assert rolls.frame_override_cards(repo, roll, "a") == set()


def test_a_normalization_lock_becomes_both_halves_of_the_split(repo):
    from negpy.services.assets.migrations.roll_fields import migrate_baseline_card_split

    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/a.tif", "/r/b.tif"])
    rolls.set_frame_override(repo, roll, "a", "process", True)
    rolls.set_frame_override(repo, roll, "b", "lens", True)

    migrate_baseline_card_split(repo)
    migrate_baseline_card_split(repo)

    assert rolls.frame_override_cards(repo, roll, "a") == {"process", "baseline"}
    assert rolls.frame_override_cards(repo, roll, "b") == {"lens"}


def test_cast_removal_locks_only_a_strength_off_its_modes_default(repo):
    from negpy.features.process.models import ProcessMode
    from negpy.services.assets.migrations.roll_fields import migrate_cast_removal_roll_locks

    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/a.tif", "/r/b.tif", "/r/c.tif"])

    def cfg(mode, strength):
        base = WorkspaceConfig()
        return replace(
            base,
            process=replace(base.process, process_mode=mode),
            exposure=replace(base.exposure, cast_removal_strength=strength),
        )

    repo.save_file_settings("a", cfg(ProcessMode.C41, 0.4), file_path="/r/a.tif")
    repo.save_file_settings("b", cfg(ProcessMode.C41, 1.0), file_path="/r/b.tif")
    repo.save_file_settings("c", cfg(ProcessMode.E6, 0.0), file_path="/r/c.tif")

    migrate_cast_removal_roll_locks(repo)
    repo.save_file_settings("b", cfg(ProcessMode.C41, 0.2), file_path="/r/b.tif")
    migrate_cast_removal_roll_locks(repo)

    assert rolls.frame_override_cards(repo, roll, "a") == {"cast_removal"}
    assert rolls.frame_override_cards(repo, roll, "b") == set(), "runs once"
    assert rolls.frame_override_cards(repo, roll, "c") == set()
