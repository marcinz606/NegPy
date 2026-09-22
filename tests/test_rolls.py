"""A Roll is a navigation layer over a folder or a hand-built set of paths; it does not
scope or duplicate the edits themselves, with one exception: roll-wide defaults for a
handful of film, rig and scanning facts (see TestRollDefaults below)."""

from unittest.mock import MagicMock

from negpy.domain.models import ProcessConfig, WorkspaceConfig
from negpy.features.process.models import DemosaicMode, ProcessMode
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.rolls import (
    add_extra_member,
    all_rolls_sorted,
    create_virtual_roll,
    delete_roll,
    fork_edit,
    folder_roll_id_for_path,
    frame_override_cards,
    import_subfolders_as_rolls,
    is_forked,
    recognize_folder,
    rename_folder_roll_disk,
    rename_roll,
    resolve_roll_config,
    roll_defaults,
    section_push,
    set_section_push,
    roll_edit_hash,
    roll_for_id,
    roll_normalization,
    rolls_containing_path,
    saved_rolls,
    set_frame_override,
    set_roll_defaults,
    set_roll_normalization,
    unfork_edit,
    unforked_hash,
    virtual_rolls,
)


def _repo() -> MagicMock:
    """A repository whose global settings live in a dict, so a write is readable back."""
    repo = MagicMock(spec=StorageRepository)
    store: dict = {}
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    repo.settings = store
    return repo


def test_recognize_folder_names_it_from_the_path():
    repo = _repo()
    roll_id = recognize_folder(repo, "/scans/2024-10-portra")
    entry = roll_for_id(repo, roll_id)
    assert entry["kind"] == "folder"
    assert entry["folder_path"] == "/scans/2024-10-portra"
    assert entry["name"] == "2024-10-portra"
    assert entry["extra_paths"] == []


def test_recognize_folder_is_idempotent():
    repo = _repo()
    first = recognize_folder(repo, "/scans/roll_a")
    second = recognize_folder(repo, "/scans/roll_a")
    assert first == second
    assert len(saved_rolls(repo)) == 1


def test_folder_roll_id_for_path_finds_a_recognized_folder():
    repo = _repo()
    assert folder_roll_id_for_path(repo, "/scans/roll_a") is None
    roll_id = recognize_folder(repo, "/scans/roll_a")
    assert folder_roll_id_for_path(repo, "/scans/roll_a") == roll_id


def test_create_virtual_roll_stores_the_exact_member_list():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef", "/b.nef"])
    entry = roll_for_id(repo, roll_id)
    assert entry["kind"] == "virtual"
    assert entry["name"] == "Portra"
    assert entry["member_paths"] == ["/a.nef", "/b.nef"]


def test_add_extra_member_extends_a_folder_rolls_extra_paths():
    repo = _repo()
    roll_id = recognize_folder(repo, "/scans/roll_a")
    add_extra_member(repo, roll_id, "/elsewhere/c.nef")
    assert roll_for_id(repo, roll_id)["extra_paths"] == ["/elsewhere/c.nef"]


def test_add_extra_member_extends_a_virtual_rolls_member_paths():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    add_extra_member(repo, roll_id, "/b.nef")
    assert roll_for_id(repo, roll_id)["member_paths"] == ["/a.nef", "/b.nef"]


def test_add_extra_member_does_not_duplicate():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    add_extra_member(repo, roll_id, "/a.nef")
    assert roll_for_id(repo, roll_id)["member_paths"] == ["/a.nef"]


def test_add_extra_member_on_unknown_roll_is_a_noop():
    repo = _repo()
    add_extra_member(repo, "not-a-real-id", "/a.nef")
    assert saved_rolls(repo) == {}


def test_rename_and_delete_roll():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", [])
    rename_roll(repo, roll_id, "Portra 400")
    assert roll_for_id(repo, roll_id)["name"] == "Portra 400"

    delete_roll(repo, roll_id)
    assert roll_for_id(repo, roll_id) is None
    assert saved_rolls(repo) == {}


def test_rename_folder_roll_disk_renames_and_updates_folder_path(tmp_path):
    repo = _repo()
    old = tmp_path / "roll_a"
    old.mkdir()
    (old / "frame001.tif").write_bytes(b"x")
    roll_id = recognize_folder(repo, str(old))

    new_path = rename_folder_roll_disk(repo, roll_id, "roll_b")

    assert new_path == str(tmp_path / "roll_b")
    assert not old.exists()
    assert (tmp_path / "roll_b" / "frame001.tif").exists()
    assert roll_for_id(repo, roll_id)["folder_path"] == new_path


def test_rename_folder_roll_disk_refuses_a_sibling_collision(tmp_path):
    repo = _repo()
    old = tmp_path / "roll_a"
    old.mkdir()
    (tmp_path / "roll_b").mkdir()
    roll_id = recognize_folder(repo, str(old))

    assert rename_folder_roll_disk(repo, roll_id, "roll_b") is None
    assert old.exists()
    assert roll_for_id(repo, roll_id)["folder_path"] == str(old)


def test_rename_folder_roll_disk_same_name_is_a_noop_success(tmp_path):
    repo = _repo()
    old = tmp_path / "roll_a"
    old.mkdir()
    roll_id = recognize_folder(repo, str(old))

    assert rename_folder_roll_disk(repo, roll_id, "roll_a") == str(old)
    assert old.exists()


def test_rename_folder_roll_disk_on_a_virtual_roll_is_a_noop(tmp_path):
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", [])
    assert rename_folder_roll_disk(repo, roll_id, "anything") is None


def test_rename_folder_roll_disk_missing_folder_is_a_noop(tmp_path):
    repo = _repo()
    roll_id = recognize_folder(repo, str(tmp_path / "gone"))
    assert rename_folder_roll_disk(repo, roll_id, "roll_b") is None


def test_virtual_rolls_lists_only_virtual_ones_sorted_by_name():
    repo = _repo()
    recognize_folder(repo, "/scans/roll_a")
    create_virtual_roll(repo, "Zebra", [])
    create_virtual_roll(repo, "apple", [])

    names = [entry["name"] for _id, entry in virtual_rolls(repo)]
    assert names == ["apple", "Zebra"]


def test_all_rolls_sorted_lists_every_kind_by_name():
    repo = _repo()
    recognize_folder(repo, "/scans/roll_a", name="Zebra")
    create_virtual_roll(repo, "apple", [])

    names = [entry["name"] for _id, entry in all_rolls_sorted(repo)]
    assert names == ["apple", "Zebra"]


def test_import_subfolders_as_rolls_recognizes_each_immediate_subfolder(tmp_path):
    (tmp_path / "roll_a").mkdir()
    (tmp_path / "roll_b").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "roll_a" / "nested").mkdir()
    repo = _repo()

    roll_ids = import_subfolders_as_rolls(repo, str(tmp_path))

    names = sorted(roll_for_id(repo, rid)["name"] for rid in roll_ids)
    assert names == ["roll_a", "roll_b"]
    # One level only: "nested" inside roll_a is not recognized on its own.
    assert folder_roll_id_for_path(repo, str(tmp_path / "roll_a" / "nested")) is None


def test_import_subfolders_as_rolls_is_idempotent_per_subfolder(tmp_path):
    (tmp_path / "roll_a").mkdir()
    repo = _repo()

    first = import_subfolders_as_rolls(repo, str(tmp_path))
    second = import_subfolders_as_rolls(repo, str(tmp_path))

    assert first == second
    assert len(saved_rolls(repo)) == 1


def test_import_subfolders_as_rolls_on_a_missing_parent_returns_nothing():
    repo = _repo()
    assert import_subfolders_as_rolls(repo, "/does/not/exist") == []


def test_roll_edit_hash_suffixes_the_roll_id():
    assert roll_edit_hash("abc123", "r1") == "abc123#roll:r1"


def test_roll_edit_hash_preserves_a_half_frame_suffix():
    """A half-frame asset's own suffix (``#1``/``#2``) survives, so each half forks to
    its own identity rather than collapsing together."""
    assert roll_edit_hash("abc123#1", "r1") == "abc123#1#roll:r1"


def test_unforked_hash_strips_only_the_roll_suffix():
    assert unforked_hash("abc123#roll:r1") == "abc123"
    assert unforked_hash("abc123#1#roll:r1") == "abc123#1"
    assert unforked_hash("abc123#1") == "abc123#1"
    assert unforked_hash("abc123") == "abc123"


def test_fork_edit_seeds_the_forked_hash_and_marks_it_forked():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    config = object()

    forked = fork_edit(repo, roll_id, "abc123", "/a.nef", config)

    assert forked == "abc123#roll:" + roll_id
    repo.save_file_settings.assert_called_once_with(forked, config, file_path="/a.nef")
    assert is_forked(repo, roll_id, "abc123")


def test_fork_edit_on_an_unknown_roll_is_a_noop():
    repo = _repo()
    forked = fork_edit(repo, "not-a-real-id", "abc123", "/a.nef", object())
    assert forked == "abc123"
    repo.save_file_settings.assert_not_called()


def test_is_forked_is_false_before_forking():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    assert not is_forked(repo, roll_id, "abc123")


def test_unfork_edit_reverses_fork_edit():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    forked = fork_edit(repo, roll_id, "abc123", "/a.nef", object())

    unfork_edit(repo, roll_id, "abc123")

    assert not is_forked(repo, roll_id, "abc123")
    repo.delete_file_settings.assert_called_once_with(forked)


def test_rolls_containing_path_finds_a_folder_roll_by_prefix(tmp_path):
    repo = _repo()
    folder = tmp_path / "roll_a"
    folder.mkdir()
    roll_id = recognize_folder(repo, str(folder))
    assert rolls_containing_path(repo, str(folder / "frame001.tif")) == [roll_id]


def test_rolls_containing_path_finds_a_folder_rolls_extra_member():
    repo = _repo()
    roll_id = recognize_folder(repo, "/scans/roll_a")
    add_extra_member(repo, roll_id, "/elsewhere/c.nef")
    assert rolls_containing_path(repo, "/elsewhere/c.nef") == [roll_id]


def test_rolls_containing_path_finds_a_virtual_rolls_member():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    assert rolls_containing_path(repo, "/a.nef") == [roll_id]


def test_rolls_containing_path_lists_every_matching_roll(tmp_path):
    repo = _repo()
    folder = tmp_path / "roll_a"
    folder.mkdir()
    frame = folder / "frame001.tif"
    folder_roll = recognize_folder(repo, str(folder))
    virtual_roll = create_virtual_roll(repo, "Picks", [str(frame)])
    assert sorted(rolls_containing_path(repo, str(frame))) == sorted([folder_roll, virtual_roll])


def test_rolls_containing_path_is_empty_for_an_unshared_path():
    repo = _repo()
    create_virtual_roll(repo, "Portra", ["/a.nef"])
    assert rolls_containing_path(repo, "/unrelated.nef") == []


class TestRollDefaults:
    """Film, rig and scanning facts a roll shares across every member frame, unless a
    frame has locked the owning card to its own value."""

    def test_a_field_with_no_roll_default_leaves_the_frame_alone(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        config = WorkspaceConfig(process=ProcessConfig(linear_raw=True))

        resolved = resolve_roll_config(repo, roll_id, "h1", config)

        assert resolved is config

    def test_a_roll_default_overrides_the_frames_own_value(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, linear_raw=True, narrowband_scan=True)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(linear_raw=False)))

        assert resolved.process.linear_raw is True
        assert resolved.process.narrowband_scan is True

    def test_no_roll_id_leaves_the_frame_alone(self):
        repo = _repo()
        assert resolve_roll_config(repo, None, "h1", WorkspaceConfig()) == WorkspaceConfig()

    def test_locking_a_card_keeps_that_frames_own_value(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, linear_raw=True, demosaic_preview=DemosaicMode.VNG)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=True)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(linear_raw=False)))

        # sensor (Calibration) is locked, so linear_raw keeps the frame's own value...
        assert resolved.process.linear_raw is False
        # ...but demosaic (a different card) is not locked, so it still takes the roll default.
        assert resolved.process.demosaic_preview == DemosaicMode.VNG

    def test_locking_does_not_affect_a_different_frame_in_the_same_roll(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, linear_raw=True)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=True)

        resolved = resolve_roll_config(repo, roll_id, "h2", WorkspaceConfig(process=ProcessConfig(linear_raw=False)))

        assert resolved.process.linear_raw is True

    def test_unlocking_a_card_reverts_to_the_roll_default(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, linear_raw=True)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=True)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=False)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(linear_raw=False)))

        assert resolved.process.linear_raw is True
        assert frame_override_cards(repo, roll_id, "h1") == set()

    def test_frame_override_cards_is_empty_for_an_unknown_roll(self):
        repo = _repo()
        assert frame_override_cards(repo, "not-a-real-id", "h1") == set()

    def test_set_roll_defaults_on_unknown_roll_is_a_noop(self):
        repo = _repo()
        set_roll_defaults(repo, "not-a-real-id", linear_raw=True)
        assert saved_rolls(repo) == {}

    def test_roll_defaults_reads_back_what_was_set(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, linear_raw=True)
        set_roll_defaults(repo, roll_id, hue_trim=2.5)

        assert roll_defaults(repo, roll_id) == {"linear_raw": True, "hue_trim": 2.5}

    def test_process_mode_follows_the_roll_unless_the_film_card_is_locked(self):
        """process_mode is a roll default on the "film" card, like everything else --
        locking a different card leaves it following the roll."""
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, process_mode=ProcessMode.BW)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=True)
        set_frame_override(repo, roll_id, "h1", "demosaic", locked=True)
        set_frame_override(repo, roll_id, "h1", "process", locked=True)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(process_mode=ProcessMode.C41)))

        assert resolved.process.process_mode == ProcessMode.BW

    def test_process_mode_can_be_locked_away_on_the_film_card(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, process_mode=ProcessMode.BW)
        set_frame_override(repo, roll_id, "h1", "film", locked=True)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(process_mode=ProcessMode.C41)))

        assert resolved.process.process_mode == ProcessMode.C41

    def test_positive_source_follows_the_roll_unless_the_film_card_is_locked(self):
        """Positive is a roll default on the same "film" card as process_mode."""
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, positive_source=True)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=True)
        set_frame_override(repo, roll_id, "h1", "demosaic", locked=True)
        set_frame_override(repo, roll_id, "h1", "process", locked=True)

        resolved = resolve_roll_config(
            repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=False))
        )

        assert resolved.process.positive_source is True

    def test_a_geometry_field_follows_the_roll_unless_its_card_is_locked(self):
        """Roll defaults span config sections: the Auto Crop card's fields live on
        GeometryConfig, not ProcessConfig."""
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, autocrop_rebate_trim=0.5, distortion_k1=0.02)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig())

        assert resolved.geometry.autocrop_rebate_trim == 0.5
        assert resolved.geometry.distortion_k1 == 0.02

    def test_locking_auto_crop_leaves_lens_following_the_roll(self):
        """Both cards write GeometryConfig, so one lock must not take the other with it."""
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, autocrop_rebate_trim=0.5, distortion_k1=0.02)
        set_frame_override(repo, roll_id, "h1", "autocrop", locked=True)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig())

        assert resolved.geometry.autocrop_rebate_trim == 1.0
        assert resolved.geometry.distortion_k1 == 0.02

    def test_flat_field_follows_the_roll_unless_its_card_is_locked(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, apply=True, profile_id="rig-1")

        assert resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig()).flatfield.profile_id == "rig-1"

        set_frame_override(repo, roll_id, "h1", "flatfield", locked=True)
        assert resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig()).flatfield.profile_id == ""

    def test_positive_source_can_be_locked_away_on_the_film_card(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, positive_source=True)
        set_frame_override(repo, roll_id, "h1", "film", locked=True)

        resolved = resolve_roll_config(
            repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=False))
        )

        assert resolved.process.positive_source is False


class TestSectionPush:
    """What a frame-level card last pushed to the roll: a record, not a binding -- it
    never overlays onto another frame, it only lets the card say whether the frame in
    front of you still agrees with the roll."""

    def test_a_roll_with_no_push_reads_empty(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        assert section_push(repo, roll_id, "tone") == {}

    def test_it_reads_back_what_was_pushed(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_section_push(repo, roll_id, "tone", {"dye_separation": 0.4})
        assert section_push(repo, roll_id, "tone") == {"dye_separation": 0.4}

    def test_a_second_push_merges_rather_than_replaces(self):
        """Applying two of a card's settings in two goes leaves both at the roll."""
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_section_push(repo, roll_id, "tone", {"dye_separation": 0.4})
        set_section_push(repo, roll_id, "tone", {"toe": 0.2})
        assert section_push(repo, roll_id, "tone") == {"dye_separation": 0.4, "toe": 0.2}

    def test_cards_do_not_share_a_record(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_section_push(repo, roll_id, "tone", {"dye_separation": 0.4})
        assert section_push(repo, roll_id, "finish") == {}

    def test_pushing_to_an_unknown_roll_is_a_noop(self):
        repo = _repo()
        set_section_push(repo, "not-a-real-id", "tone", {"dye_separation": 0.4})
        assert saved_rolls(repo) == {}


class TestRollNormalization:
    """A roll's own Batch Analysis baseline: written only by Batch Analysis itself, read
    by any frame's Use Luma/Color Average axes -- unlike ROLL_DEFAULT_FIELDS, this has no
    lock/override of its own."""

    def test_unanalyzed_roll_has_no_baseline(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        assert roll_normalization(repo, roll_id) is None

    def test_reads_back_what_was_set(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9), (0.01, 0.0, -0.01))

        data = roll_normalization(repo, roll_id)

        assert data == {"floors": (0.1, 0.1, 0.1), "ceils": (0.9, 0.9, 0.9), "cast": (0.01, 0.0, -0.01)}

    def test_defaults_cast_to_zero(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))

        assert roll_normalization(repo, roll_id)["cast"] == (0.0, 0.0, 0.0)

    def test_overwrites_a_previous_baseline(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
        set_roll_normalization(repo, roll_id, (0.2, 0.2, 0.2), (0.8, 0.8, 0.8))

        assert roll_normalization(repo, roll_id)["floors"] == (0.2, 0.2, 0.2)

    def test_set_on_unknown_roll_is_a_noop(self):
        repo = _repo()
        set_roll_normalization(repo, "not-a-real-id", (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
        assert saved_rolls(repo) == {}

    def test_normalization_is_isolated_per_roll(self):
        repo = _repo()
        roll_a = create_virtual_roll(repo, "Portra", [])
        roll_b = create_virtual_roll(repo, "Tri-X", [])
        set_roll_normalization(repo, roll_a, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))

        assert roll_normalization(repo, roll_a) is not None
        assert roll_normalization(repo, roll_b) is None
