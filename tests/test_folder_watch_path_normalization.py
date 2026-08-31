"""FolderWatchService candidate/existing-path comparison.

Candidates come from os.path.abspath (OS-native separators); existing_paths can carry
Qt's forward-slash convention instead. Windows paths are also case-insensitive. The
comparison must treat both spellings of the same file as the same path.
"""

from negpy.infrastructure.filesystem.watcher import FolderWatchService


def test_forward_slash_existing_path_is_not_reported_new(tmp_path):
    (tmp_path / "roll12.nef").write_bytes(b"stub")
    forward_slash_path = str(tmp_path / "roll12.nef").replace("\\", "/")

    new_files = FolderWatchService.scan_for_new_files(str(tmp_path), {forward_slash_path})

    assert new_files == []


def test_genuinely_new_file_is_still_reported(tmp_path):
    (tmp_path / "roll12.nef").write_bytes(b"stub")

    new_files = FolderWatchService.scan_for_new_files(str(tmp_path), set())

    assert len(new_files) == 1
    assert new_files[0].endswith("roll12.nef")
