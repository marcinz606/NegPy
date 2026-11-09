"""Path-keyed records follow a folder renamed on disk."""

import os
from typing import Any, Optional

from negpy.services.assets import composites, rolls


def rehome_path_prefix(repo: Any, old_prefix: str, new_prefix: str) -> None:
    """Repoint the edits DB, the rolls and the composites from *old_prefix* to *new_prefix*.
    Content hashes do not move, but these stores find a frame by its path."""
    old_prefix, new_prefix = old_prefix.rstrip("/\\"), new_prefix.rstrip("/\\")

    def move(path: str) -> str:
        if path == old_prefix or path.startswith(old_prefix + os.sep):
            return new_prefix + path[len(old_prefix) :]
        return path

    repo.rehome_path_prefix(old_prefix, new_prefix)
    rolls.rehome_paths(repo, move)
    composites.rehome_paths(repo, move)


def entry_folder_missing(entry: Optional[dict]) -> bool:
    """True when *entry* is a folder roll whose folder is not on disk."""
    return bool(entry) and entry.get("kind") == "folder" and not os.path.isdir(entry.get("folder_path") or "")


def folder_roll_missing(repo: Any, roll_id: str) -> bool:
    """True when *roll_id* is a folder roll whose folder is not on disk."""
    return entry_folder_missing(rolls.roll_for_id(repo, roll_id))


def folder_fits_roll(repo: Any, roll_id: str, path: str) -> bool:
    """False when *path* holds none of the edited files the roll saved under its old folder.
    A roll with no saved edits accepts any folder."""
    entry = rolls.roll_for_id(repo, roll_id) or {}
    known = {name.casefold() for name in repo.edited_file_names_under(entry.get("folder_path") or "")}
    if not known:
        return True
    try:
        present = {name.casefold() for name in os.listdir(path)}
    except OSError:
        return False
    return bool(known & present)


RELINK_NOT_A_FOLDER_ROLL = "not_a_folder_roll"
RELINK_NOT_A_FOLDER = "not_a_folder"
RELINK_FOLDER_TAKEN = "folder_taken"


def relink_folder_roll(repo: Any, roll_id: str, new_path: str) -> tuple[Optional[str], str]:
    """Point a folder roll at *new_path*, its folder after a rename or move outside NegPy.
    Returns (old path, "") or (None, reason). A name still equal to the old folder's default
    name takes the new folder's; a name the user set stays."""
    entry = rolls.roll_for_id(repo, roll_id)
    if not entry or entry.get("kind") != "folder":
        return None, RELINK_NOT_A_FOLDER_ROLL
    if not os.path.isdir(new_path):
        return None, RELINK_NOT_A_FOLDER
    other = rolls.folder_roll_id_for_path(repo, new_path)
    if other and other != roll_id:
        return None, RELINK_FOLDER_TAKEN
    old_path = entry.get("folder_path") or ""
    default_name = bool(old_path) and entry.get("name") == rolls.folder_roll_name(repo, old_path)
    if old_path:
        rehome_path_prefix(repo, old_path, new_path)
    if default_name:
        rolls.rename_roll(repo, roll_id, rolls.folder_roll_name(repo, new_path))
    return old_path, ""
