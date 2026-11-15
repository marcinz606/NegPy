"""Path-keyed records follow a folder renamed on disk."""

import os
import sys
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


def member_folders(entry: Optional[dict]) -> list[str]:
    """The folders that hold a virtual roll's frames, sorted."""
    if not entry or entry.get("kind") == "folder":
        return []
    return sorted({os.path.dirname(p) for p in entry.get("member_paths") or () if p})


def missing_member_folders(entry: Optional[dict]) -> list[str]:
    """The folders of a virtual roll's frames that are not on disk."""
    return [folder for folder in member_folders(entry) if not os.path.isdir(folder)]


def entry_missing(entry: Optional[dict]) -> bool:
    """True when a folder roll's folder, or a folder of a virtual roll's frames, is not on disk."""
    return entry_folder_missing(entry) or bool(missing_member_folders(entry))


def roll_missing(repo: Any, roll_id: str) -> bool:
    return entry_missing(rolls.roll_for_id(repo, roll_id))


def gone_member_paths(entry: Optional[dict]) -> list[str]:
    """The virtual roll's frames whose files are not on disk."""
    if not entry or entry.get("kind") == "folder":
        return []
    return [p for p in entry.get("member_paths") or () if not os.path.exists(p)]


def gone_folders_for(entry: Optional[dict], gone: list[str], path: str) -> list[str]:
    """The folders in *gone* that *path* fits best: the largest share of each folder's own frames
    by file name, then a folder of the same name. Empty when it fits none; more than one on a tie."""
    try:
        names = os.listdir(path)
    except OSError:
        return []
    present = set(names)
    # Names match without case only where the file system does.
    folded = {name.casefold() for name in names} if sys.platform in ("darwin", "win32") else set()
    members = (entry or {}).get("member_paths") or ()
    scores = {}
    for folder in gone:
        own = [os.path.basename(p) for p in members if os.path.dirname(p) == folder]
        hits = sum(n in present or n.casefold() in folded for n in own)
        if hits:
            scores[folder] = (hits / len(own), hits)
    if not scores:
        return []
    top = max(scores.values())
    best = [f for f, score in scores.items() if score == top]
    same_name = [f for f in best if os.path.basename(f).casefold() == os.path.basename(path.rstrip("/\\")).casefold()]
    return same_name if len(same_name) == 1 else best


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
