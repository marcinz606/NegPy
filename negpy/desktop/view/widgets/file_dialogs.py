"""Start-directory resolution for the file pickers."""

import os

_LAST_OPEN_FOLDER = "last_open_folder"


def pick_start_dir(*candidates: str) -> str:
    """The first candidate that resolves to a directory that exists, else the home folder.

    A candidate is a file or a folder path; a file gives its parent folder. Give an empty
    string for a candidate that does not apply. Qt reads an empty start directory as the
    process working directory, which is `/` for a bundled app, so never return one.
    """
    for candidate in candidates:
        if not candidate:
            continue
        folder = candidate if os.path.isdir(candidate) else os.path.dirname(candidate)
        if folder and os.path.isdir(folder):
            return folder
    return os.path.expanduser("~")


def last_open_folder(repo) -> str:
    """The folder of the most recent Add Files / Add Folder, as a picker fallback."""
    return repo.get_global_setting(_LAST_OPEN_FOLDER, "") or ""
