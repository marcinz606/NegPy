"""The library is the filesystem: roots are folders on disk, and a search is a walk.

There is no index database. An index would have to be kept true against a tree the
user reorganizes outside NegPy, and it would buy less than it looks: the metadata
worth searching lives in the edits (joined by path, see
``StorageRepository.load_settings_by_path``), not in the files, so a walk only has to
supply name, path and date — which ``os.stat`` already gives it. A walk cannot go
stale, needs no reindex action and no cache to clear.

ponytail: the per-session walk cache below is the whole optimization. If a real
archive ever measures slow on the first search, the upgrade is a path-keyed table of
these same stat rows — not a different search.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterator, Optional

from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS, is_ir_sidecar_path
from negpy.services.assets.search import Term, facts_for, match


def _is_image(name: str) -> bool:
    return name.lower().endswith(tuple(SUPPORTED_RAW_EXTENSIONS))


def folder_counts(path: str) -> tuple[int, int]:
    """(images, subfolders) directly inside a folder — one readdir, nothing opened.

    What a file manager reads to label a row, and the reason browsing a library is
    free: the expensive pass (hashing, thumbnails) only runs on an accepted load.
    """
    images = subfolders = 0
    try:
        for entry in os.scandir(path):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                subfolders += 1
            elif _is_image(entry.name) and not is_ir_sidecar_path(entry.path):
                images += 1
    except OSError:
        return 0, 0
    return images, subfolders


def summarize_counts(images: int, subfolders: int) -> str:
    parts = []
    if images:
        parts.append(f"{images} photo{'s' if images != 1 else ''}")
    if subfolders:
        parts.append(f"{subfolders} folder{'s' if subfolders != 1 else ''}")
    return " · ".join(parts) or "empty"


def iter_library_files(roots: list[str]) -> Iterator[dict[str, Any]]:
    """Walk the roots, yielding one asset-shaped dict per supported image.

    Shaped like a session asset (``name``/``path``/``mtime``) so the same
    ``facts_for`` builds facts for a walked file and a loaded frame alike.
    """
    extensions = tuple(SUPPORTED_RAW_EXTENSIONS)
    seen_roots: set[str] = set()
    for root in roots:
        real = os.path.realpath(root)
        # Nested or repeated roots would otherwise yield their overlap twice.
        if any(real == s or real.startswith(s + os.sep) for s in seen_roots):
            continue
        seen_roots.add(real)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if not filename.lower().endswith(extensions):
                    continue
                path = os.path.join(dirpath, filename)
                if is_ir_sidecar_path(path):
                    continue
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                yield {"name": filename, "path": path, "mtime": stat.st_mtime, "size": stat.st_size}


def search_library(
    files: list[dict[str, Any]],
    terms: list[Term],
    configs_by_path: dict[str, Any],
    marks_by_path: Optional[dict[str, str]] = None,
) -> list[str]:
    """Paths of the walked files matching every term, in walk order.

    An empty query returns nothing rather than the whole archive: a library search is
    an explicit action, and answering it with 50 000 frames is never what was meant.
    """
    if not terms:
        return []
    marks = marks_by_path or {}
    hits = []
    for entry in files:
        path = entry["path"]
        mark = marks.get(path)
        asset = {**entry, "keeper": mark == "keeper", "excluded": mark == "excluded"}
        if match(terms, facts_for(asset, configs_by_path.get(path))):
            hits.append(path)
    return hits


class LibraryWalkCache:
    """One walk per session unless the user asks for a refresh.

    Held rather than re-walked so a run of searches costs one traversal; dropped
    whenever the roots change, so it can never describe a tree that is no longer
    the library.
    """

    def __init__(self) -> None:
        self._roots: tuple[str, ...] = ()
        self._files: Optional[list[dict[str, Any]]] = None

    def invalidate(self) -> None:
        self._files = None

    def files(self, roots: list[str], progress: Optional[Callable[[int], None]] = None) -> list[dict[str, Any]]:
        if self._files is not None and self._roots == tuple(roots):
            return self._files
        walked: list[dict[str, Any]] = []
        for entry in iter_library_files(roots):
            walked.append(entry)
            if progress and len(walked) % 500 == 0:
                progress(len(walked))
        self._roots = tuple(roots)
        self._files = walked
        return walked
