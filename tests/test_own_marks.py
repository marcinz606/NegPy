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


def test_the_strip_tooltip_names_each_mark(tmp_path, qapp):
    from PyQt6.QtCore import Qt

    from negpy.desktop.session import AssetListModel

    session, repo, roll = _session(tmp_path)
    rolls.set_frame_override(repo, roll, "h2", "demosaic", True)
    rolls.set_frame_override(repo, roll, "h2", "process", True)
    session.refresh_own_marks()
    session.state.uploaded_files[1]["average"] = "outlier"
    session.state.uploaded_files[0]["average"] = "on"
    model = AssetListModel(session.state)
    model.refresh()

    tips = [model.data(model.index(row, 0), Qt.ItemDataRole.ToolTipRole) for row in range(2)]

    assert "Prints on the roll average." in tips[0] and "own value" not in tips[0]
    assert "far from the other frames" in tips[1]
    assert "Keeps its own value on: Raw Decode, Metering" in tips[1]


def test_a_saved_frame_that_follows_the_roll_average_is_marked(tmp_path):
    """Hydration lays the roll's Use Average over a saved edit, so the strip reads it the same way."""
    from negpy.domain.models import WorkspaceConfig

    session, repo, roll = _session(tmp_path)
    rolls.set_roll_normalization(repo, roll, (0.1,) * 3, (0.9,) * 3, frames=("h1", "h2"))
    rolls.set_roll_defaults(repo, roll, use_luma_average=True, use_color_average=True)
    repo.save_file_settings("h2", WorkspaceConfig(), file_path="/r/h2.tif")

    session.refresh_average_marks()

    assert [f.get("average") for f in session.state.uploaded_files] == ["on", "on"]

    rolls.set_frame_override(repo, roll, "h2", "baseline", True)
    session.refresh_average_marks()

    assert session.state.uploaded_files[1].get("average") is None
