"""Roll membership: a Roll is a named, navigable group of frames -- either a real
library folder recognized as a roll, or a virtual roll built by hand from whatever the
Film Strip currently holds (a search result, a hand-picked selection, extras added to a
folder roll that are not physically in that folder).

Edits are not stored here and are not scoped by roll: they stay in the edits DB under
each frame's own content hash, exactly as if no Roll existed. A Roll only decides which
files show up when you open it.
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional

ROLLS_KEY = "rolls_by_id"


def _read(repo: Any) -> Dict[str, dict]:
    saved = repo.get_global_setting(ROLLS_KEY, default=None)
    return dict(saved) if isinstance(saved, dict) else {}


def _write(repo: Any, store: Dict[str, dict]) -> None:
    repo.save_global_setting(ROLLS_KEY, store)


def saved_rolls(repo: Any) -> Dict[str, dict]:
    """Every remembered roll, keyed by id."""
    return _read(repo)


def roll_for_id(repo: Any, roll_id: str) -> Optional[dict]:
    return _read(repo).get(roll_id)


def folder_roll_id_for_path(repo: Any, path: str) -> Optional[str]:
    """The id of the roll recognizing *path*, or None if not yet recognized."""
    for roll_id, entry in _read(repo).items():
        if entry.get("kind") == "folder" and entry.get("folder_path") == path:
            return roll_id
    return None


def recognize_folder(repo: Any, path: str, name: str = "") -> str:
    """Mark *path* as a recognized folder roll. Idempotent: returns the existing id
    when the folder is already recognized, without touching its stored name."""
    existing = folder_roll_id_for_path(repo, path)
    if existing:
        return existing
    store = _read(repo)
    roll_id = uuid.uuid4().hex
    store[roll_id] = {
        "kind": "folder",
        "name": name or path.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1] or path,
        "folder_path": path,
        "extra_paths": [],
        "created_at": time.time(),
    }
    _write(repo, store)
    return roll_id


def import_subfolders_as_rolls(repo: Any, parent_path: str) -> List[str]:
    """Recognize every immediate subfolder of *parent_path* as its own folder roll.

    One level only: a subfolder's own children are not walked. Idempotent per
    subfolder, so re-running over a parent that already has some rolls recognized
    only creates the missing ones.
    """
    try:
        entries = sorted(e.path for e in os.scandir(parent_path) if e.is_dir() and not e.name.startswith("."))
    except OSError:
        return []
    return [recognize_folder(repo, path) for path in entries]


def create_virtual_roll(repo: Any, name: str, member_paths: List[str]) -> str:
    store = _read(repo)
    roll_id = uuid.uuid4().hex
    store[roll_id] = {
        "kind": "virtual",
        "name": name,
        "member_paths": list(member_paths),
        "created_at": time.time(),
    }
    _write(repo, store)
    return roll_id


def add_extra_member(repo: Any, roll_id: str, path: str) -> None:
    """Extend a roll's membership by one path: a folder roll's extra_paths, or a virtual
    roll's member_paths. No-op for an unknown roll id or an already-member path."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return
    key = "extra_paths" if entry["kind"] == "folder" else "member_paths"
    if path not in entry[key]:
        entry[key] = [*entry[key], path]
        _write(repo, store)


def rename_roll(repo: Any, roll_id: str, name: str) -> None:
    store = _read(repo)
    if roll_id in store:
        store[roll_id]["name"] = name
        _write(repo, store)


def rename_folder_roll_disk(repo: Any, roll_id: str, new_name: str) -> Optional[str]:
    """Rename a folder roll's actual folder on disk to *new_name*, in its current
    parent directory, and update the roll's own folder_path to match. Returns the
    new path, or None (and nothing is touched) when the roll is not a folder roll,
    its folder is missing, a sibling is already named that, or the OS rename fails
    (no permission, a mount that refuses it). *new_name* equal to the folder's
    current basename is a no-op success, not a collision.
    """
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None or entry.get("kind") != "folder":
        return None
    old_path = entry.get("folder_path", "")
    if not old_path or not os.path.isdir(old_path):
        return None
    parent = os.path.dirname(old_path.rstrip("/\\"))
    new_path = os.path.join(parent, new_name)
    if os.path.normcase(os.path.abspath(new_path)) == os.path.normcase(os.path.abspath(old_path)):
        return old_path
    if os.path.exists(new_path):
        return None
    try:
        os.rename(old_path, new_path)
    except OSError:
        return None
    entry["folder_path"] = new_path
    _write(repo, store)
    return new_path


def delete_roll(repo: Any, roll_id: str) -> None:
    store = _read(repo)
    if roll_id in store:
        del store[roll_id]
        _write(repo, store)


def all_rolls_sorted(repo: Any) -> List[tuple]:
    """(roll_id, entry) pairs for every roll, folder and virtual alike, name-sorted."""
    return sorted(_read(repo).items(), key=lambda pair: pair[1].get("name", "").casefold())


def virtual_rolls(repo: Any) -> List[tuple]:
    """(roll_id, entry) pairs for every virtual roll, name-sorted."""
    return [pair for pair in all_rolls_sorted(repo) if pair[1].get("kind") == "virtual"]


def rolls_containing_path(repo: Any, path: str) -> List[str]:
    """Every roll *path* belongs to: under a folder roll's own folder or in its
    extra_paths, or listed in a virtual roll's member_paths. A cheap path comparison
    against the small rolls store, not a disk walk."""
    norm = os.path.normcase(os.path.abspath(path))
    out = []
    for roll_id, entry in _read(repo).items():
        if entry.get("kind") == "folder":
            folder = entry.get("folder_path", "")
            folder_norm = os.path.normcase(os.path.abspath(folder)) if folder else ""
            in_folder = bool(folder_norm) and norm.startswith(folder_norm + os.sep)
            if in_folder or path in entry.get("extra_paths", []):
                out.append(roll_id)
        elif path in entry.get("member_paths", []):
            out.append(roll_id)
    return out
