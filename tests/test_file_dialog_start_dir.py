import os
from unittest.mock import MagicMock

from negpy.desktop.view.widgets.file_dialogs import last_open_folder, pick_start_dir
from negpy.infrastructure.storage.repository import StorageRepository


def test_file_candidate_gives_its_folder(tmp_path):
    frame = tmp_path / "red.dng"
    frame.write_bytes(b"")
    assert pick_start_dir(str(frame)) == str(tmp_path)


def test_folder_candidate_is_kept(tmp_path):
    assert pick_start_dir(str(tmp_path)) == str(tmp_path)


def test_empty_and_missing_candidates_are_skipped(tmp_path):
    frame = tmp_path / "green.dng"
    frame.write_bytes(b"")
    gone = tmp_path / "unmounted" / "blue.dng"
    assert pick_start_dir("", str(gone), str(frame)) == str(tmp_path)


def test_home_is_the_last_resort():
    # Never "" — Qt reads that as the working directory, which is / for a bundled app.
    assert pick_start_dir("", "/no/such/place") == os.path.expanduser("~")


def test_last_open_folder_reads_the_global_setting():
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = "/scans/roll-12"
    assert last_open_folder(repo) == "/scans/roll-12"


def test_last_open_folder_unset_is_empty():
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    assert last_open_folder(repo) == ""


def test_empty_triplet_row_browses_beside_the_red_frame(qapp, tmp_path, monkeypatch):
    from negpy.desktop.view.sidebar import files as files_mod

    red = tmp_path / "roll12_r.dng"
    red.write_bytes(b"")
    dlg = files_mod._RgbTripletDialog(None, str(red), "", "", start_dir="")

    seen = {}

    def fake_open(_parent, _caption, start, _filter):
        seen["start"] = start
        return ("", "")

    monkeypatch.setattr(files_mod.QFileDialog, "getOpenFileName", fake_open)
    dlg._browse(dlg._edits["Blue"])
    assert seen["start"] == str(tmp_path)
