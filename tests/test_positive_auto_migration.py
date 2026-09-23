import sqlite3
from dataclasses import replace

import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls
from negpy.services.assets.migrations.positive_auto import migrate_auto_meter_for_positive_frames

_DONE_FLAG = "auto_meter_positive_migrated_v1"


@pytest.fixture
def repo(tmp_path):
    r = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    r.initialize()
    return r


def _config(**process_overrides):
    """A Slide frame, since only Slide can carry Positive."""
    cfg = WorkspaceConfig()
    return replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, **process_overrides))


def _load(repo, file_hash):
    return repo.load_file_settings(file_hash)


def test_turns_off_both_toggles_on_an_untouched_positive_frame(repo):
    repo.save_file_settings("h1", _config(positive_source=True), file_path="/a.tif")

    migrate_auto_meter_for_positive_frames(repo)

    loaded = _load(repo, "h1")
    assert loaded.exposure.auto_exposure is False
    assert loaded.exposure.auto_normalize_contrast is False


def test_leaves_a_negative_frame_alone(repo):
    repo.save_file_settings("h1", _config(positive_source=False), file_path="/a.tif")

    migrate_auto_meter_for_positive_frames(repo)

    loaded = _load(repo, "h1")
    assert loaded.exposure.auto_exposure is True
    assert loaded.exposure.auto_normalize_contrast is True


def test_leaves_a_deliberate_choice_on_a_positive_frame_alone(repo):
    cfg = _config(positive_source=True)
    cfg = replace(cfg, exposure=replace(cfg.exposure, auto_exposure=False, auto_normalize_contrast=True))
    repo.save_file_settings("h1", cfg, file_path="/a.tif")

    migrate_auto_meter_for_positive_frames(repo)

    loaded = _load(repo, "h1")
    assert loaded.exposure.auto_exposure is False
    # Only the one still at the negative default moves; the other was already off.
    assert loaded.exposure.auto_normalize_contrast is False


def test_corrects_a_frame_that_only_inherits_positive_from_its_roll(repo):
    """positive_source predates the "film" card's lock, so no saved row has ever
    overridden it away from the roll -- the roll default is the complete answer."""
    roll_id = rolls.recognize_folder(repo, "/roll")
    rolls.set_roll_defaults(repo, roll_id, positive_source=True)
    repo.save_file_settings("h1", _config(positive_source=False), file_path="/roll/a.tif")

    migrate_auto_meter_for_positive_frames(repo)

    loaded = _load(repo, "h1")
    assert loaded.exposure.auto_exposure is False
    assert loaded.exposure.auto_normalize_contrast is False


def test_leaves_a_frame_outside_any_positive_roll_alone(repo):
    roll_id = rolls.recognize_folder(repo, "/roll")
    rolls.set_roll_defaults(repo, roll_id, positive_source=True)
    repo.save_file_settings("h1", _config(positive_source=False), file_path="/elsewhere/a.tif")

    migrate_auto_meter_for_positive_frames(repo)

    loaded = _load(repo, "h1")
    assert loaded.exposure.auto_exposure is True
    assert loaded.exposure.auto_normalize_contrast is True


def test_sets_the_done_flag(repo):
    migrate_auto_meter_for_positive_frames(repo)
    assert repo.get_global_setting(_DONE_FLAG) is True


def test_no_rows_just_sets_the_flag(repo):
    migrate_auto_meter_for_positive_frames(repo)
    assert repo.get_global_setting(_DONE_FLAG) is True


def test_second_run_never_touches_a_choice_made_after_the_first(repo):
    """The done flag is what makes this safe to ship: once it has run, a user turning
    Auto Density back on for a Positive frame must survive every later restart."""
    repo.save_file_settings("h1", _config(positive_source=True), file_path="/a.tif")

    migrate_auto_meter_for_positive_frames(repo)
    # The user deliberately re-enables it after the migration already ran once.
    repo.save_file_settings("h1", _config(positive_source=True), file_path="/a.tif")
    turned_on = _load(repo, "h1")
    turned_on = replace(turned_on, exposure=replace(turned_on.exposure, auto_exposure=True))
    repo.save_file_settings("h1", turned_on, file_path="/a.tif")

    migrate_auto_meter_for_positive_frames(repo)

    assert _load(repo, "h1").exposure.auto_exposure is True


def test_closes_the_connection_it_opens(repo, monkeypatch):
    """Mirrors test_normalization_roll_migration's leak guard: sqlite3.connect() used
    as a bare context manager commits on exit but never closes."""
    repo.save_file_settings("h1", _config(positive_source=True), file_path="/a.tif")
    edits_db = repo.edits_db_path
    real_connect = sqlite3.connect
    opened = []

    def tracking_connect(database, *args, **kwargs):
        conn = real_connect(database, *args, **kwargs)
        if str(database) == str(edits_db):
            opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    migrate_auto_meter_for_positive_frames(repo)
    monkeypatch.undo()

    assert opened, "migration opened no connection to the edits DB"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
