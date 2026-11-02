"""The Film Strip marks a frame that keeps its own value on a Roll-tab card."""

from unittest.mock import MagicMock

from negpy.desktop.session import DesktopSessionManager
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls


def _session(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/h1.tif", "/r/h2.tif"])
    session.state.active_roll_id = roll
    session.state.uploaded_files = [{"name": f"{h}.tif", "path": f"/r/{h}.tif", "hash": h} for h in ("h1", "h2")]
    session.asset_model = MagicMock()
    return session, repo, roll


def test_a_locked_frame_is_marked_and_unmarked(tmp_path):
    session, repo, roll = _session(tmp_path)
    rolls.set_frame_override(repo, roll, "h2", "process", True)

    session.refresh_own_marks()

    assert [bool(f.get("own_cards")) for f in session.state.uploaded_files] == [False, True]

    rolls.set_frame_override(repo, roll, "h2", "process", False)
    session.refresh_own_marks()

    assert not any(f.get("own_cards") for f in session.state.uploaded_files)


def test_a_forked_frame_reads_its_physical_scans_lock(tmp_path):
    session, repo, roll = _session(tmp_path)
    session.state.uploaded_files[1]["hash"] = rolls.roll_edit_hash("h2", roll)
    rolls.set_frame_override(repo, roll, "h2", "film", True)

    session.refresh_own_marks()

    assert session.state.uploaded_files[1].get("own_cards")
