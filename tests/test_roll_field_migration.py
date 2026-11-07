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


def _set_push(repo, roll_id, key, values):
    """A whole-roll apply record as older versions stored it."""
    store = rolls._read(repo)
    store[roll_id].setdefault("section_pushes", {})[key] = values
    rolls._write(repo, store)


def _push(repo, roll_id, key):
    return rolls.roll_for_id(repo, roll_id).get("section_pushes", {}).get(key, {})


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


def test_a_frame_that_followed_its_roll_keeps_the_rolls_cast_removal(repo):
    from negpy.services.assets.migrations.roll_fields import migrate_cast_removal_to_frames

    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/a.tif", "/r/b.tif", "/r/c.tif"])
    for name in ("a", "b"):
        repo.save_file_settings(name, WorkspaceConfig(), file_path=f"/r/{name}.tif")
    rolls.set_roll_defaults(repo, roll, cast_removal_strength=0.4)
    rolls.set_frame_override(repo, roll, "b", "cast_removal", True)
    rolls.set_frame_override(repo, roll, "b", "sensor", True)

    migrate_cast_removal_to_frames(repo)
    rolls.set_roll_defaults(repo, roll, cast_removal_strength=0.7)
    migrate_cast_removal_to_frames(repo)

    assert repo.load_file_settings("a").exposure.cast_removal_strength == 0.4
    assert repo.load_file_settings("b").exposure.cast_removal_strength == WorkspaceConfig().exposure.cast_removal_strength
    assert rolls.frame_override_cards(repo, roll, "b") == {"sensor"}
    assert rolls.roll_defaults(repo, roll) == {"cast_removal_strength": 0.7}, "runs once"


def test_section_pushes_are_dropped_once(repo):
    from negpy.services.assets.migrations.roll_fields import migrate_drop_section_pushes

    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/a.tif"])
    _set_push(repo, roll, "tone", {"density": 1.2})

    migrate_drop_section_pushes(repo)
    _set_push(repo, roll, "tone", {"density": 1.2})
    migrate_drop_section_pushes(repo)

    assert _push(repo, roll, "tone") == {"density": 1.2}, "runs once"
    repo.save_global_setting("section_pushes_dropped_v1", False)
    migrate_drop_section_pushes(repo)
    assert "section_pushes" not in rolls.roll_for_id(repo, roll)


def test_auto_crop_locks_only_a_frame_saved_with_it_on(repo):
    from negpy.services.assets.migrations.roll_fields import migrate_auto_crop_roll_locks

    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/a.tif", "/r/b.tif", "/r/c.tif"])

    def cfg(rect, auto):
        base = WorkspaceConfig()
        return replace(base, geometry=replace(base.geometry, crop_rect=rect, crop_from_auto=auto))

    repo.save_file_settings("a", cfg((0.1, 0.1, 0.9, 0.9), True), file_path="/r/a.tif")
    repo.save_file_settings("b", cfg((0.1, 0.1, 0.9, 0.9), False), file_path="/r/b.tif")
    repo.save_file_settings("c", cfg(None, False), file_path="/r/c.tif")

    migrate_auto_crop_roll_locks(repo)

    assert rolls.frame_override_cards(repo, roll, "a") == {"auto_crop"}
    assert rolls.frame_override_cards(repo, roll, "b") == set()
    assert rolls.frame_override_cards(repo, roll, "c") == set()


def _crop(rect=None, auto=False, **geometry):
    base = WorkspaceConfig()
    return replace(base, geometry=replace(base.geometry, crop_rect=rect, crop_from_auto=auto, **geometry))


def test_each_frame_keeps_the_crop_its_roll_gave_it(repo):
    from negpy.services.assets.migrations.roll_fields import migrate_crop_to_frames

    rect = (0.1, 0.1, 0.9, 0.9)
    roll = rolls.create_virtual_roll(repo, "Roll", [f"/r/{n}.tif" for n in "abcde"])
    repo.save_file_settings("a", _crop(), file_path="/r/a.tif")
    repo.save_file_settings("b", _crop(rect), file_path="/r/b.tif")
    repo.save_file_settings("c", _crop(autocrop_ratio="5:4"), file_path="/r/c.tif")
    repo.save_file_settings("d", _crop(rect, True), file_path="/r/d.tif")
    rolls.set_roll_defaults(repo, roll, crop_from_auto=True, autocrop_ratio="3:2", autocrop_offset=8, hue_trim=2.0)
    rolls.set_frame_override(repo, roll, "c", "autocrop", True)
    rolls.set_frame_override(repo, roll, "d", "auto_crop", True)
    rolls.set_frame_override(repo, roll, "d", "sensor", True)

    migrate_crop_to_frames(repo)

    a, b, c, d = (repo.load_file_settings(n).geometry for n in "abcd")
    assert (a.crop_from_auto, a.crop_rect, a.autocrop_ratio, a.autocrop_offset) == (True, None, "3:2", 8)
    assert (b.crop_from_auto, b.crop_rect, b.autocrop_ratio) == (False, rect, "3:2"), "a hand-drawn crop stays"
    assert (c.crop_from_auto, c.autocrop_ratio, c.autocrop_offset) == (True, "5:4", 0), "a locked shape stays"
    assert (d.crop_from_auto, d.crop_rect) == (True, rect)
    assert repo.load_file_settings("e") is None
    assert rolls.roll_defaults(repo, roll) == {"hue_trim": 2.0}
    assert rolls.frame_override_cards(repo, roll, "c") == set()
    assert rolls.frame_override_cards(repo, roll, "d") == {"sensor"}


def test_the_crop_migration_runs_once(repo):
    from negpy.services.assets.migrations.roll_fields import migrate_crop_to_frames

    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/a.tif"])
    repo.save_file_settings("a", _crop(), file_path="/r/a.tif")
    migrate_crop_to_frames(repo)
    rolls.set_roll_defaults(repo, roll, autocrop_ratio="4:3")

    migrate_crop_to_frames(repo)

    assert repo.load_file_settings("a").geometry.autocrop_ratio == "Free"


def test_a_frame_with_no_saved_edit_keeps_its_rolls_crop_and_cast_removal(repo, tmp_path):
    from negpy.desktop.session import DesktopSessionManager
    from negpy.services.assets.migrations.roll_fields import migrate_cast_removal_to_frames, migrate_crop_to_frames

    path = str(tmp_path / "e.tif")
    roll = rolls.create_virtual_roll(repo, "Roll", [path, str(tmp_path / "f.tif")])
    repo.save_file_settings("f", _crop(), file_path=str(tmp_path / "f.tif"))
    rolls.set_roll_defaults(repo, roll, crop_from_auto=True, autocrop_ratio="6:7", cast_removal_strength=0.3)

    migrate_cast_removal_to_frames(repo)
    migrate_crop_to_frames(repo)
    session = DesktopSessionManager(repo)
    session.state.active_roll_id = roll
    config = session.config_for_asset({"name": "e.tif", "path": path, "hash": "e"})

    assert (config.geometry.crop_from_auto, config.geometry.autocrop_ratio) == (True, "6:7")
    assert config.exposure.cast_removal_strength == 0.3
    assert repo.load_file_settings("e") is None
    assert rolls.roll_defaults(repo, roll) == {}


def test_a_reset_frame_takes_its_rolls_frame_seed(repo, tmp_path):
    from negpy.desktop.session import DesktopSessionManager

    path = str(tmp_path / "a.tif")
    roll = rolls.create_virtual_roll(repo, "Roll", [path])
    rolls.add_frame_seed(repo, roll, autocrop_ratio="6:7")
    session = DesktopSessionManager(repo)
    session.state.active_roll_id = roll

    assert session._reset_frame({"name": "a.tif", "path": path, "hash": "a"}).geometry.autocrop_ratio == "6:7"
