"""Path-keyed records follow a folder renamed on disk."""

import os
from typing import Any

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
