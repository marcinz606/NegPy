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
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS, is_ir_sidecar_path
from negpy.services.assets.search import Term, facts_for, match


@dataclass(frozen=True)
class FolderEntry:
    """A subfolder as the contact sheet shows it: a name and what is inside."""

    path: str
    name: str
    image_count: int
    subfolder_count: int

    def summary(self) -> str:
        parts = []
        if self.image_count:
            parts.append(f"{self.image_count} photo{'s' if self.image_count != 1 else ''}")
        if self.subfolder_count:
            parts.append(f"{self.subfolder_count} folder{'s' if self.subfolder_count != 1 else ''}")
        return " · ".join(parts) or "empty"


@dataclass(frozen=True)
class FolderContents:
    """One directory listing: its subfolders (with counts) and its own images.

    Costs one readdir for the folder plus one per subfolder for the counts —
    what a file manager does to draw the same view, and nothing is opened or hashed.
    """

    path: str
    folders: tuple[FolderEntry, ...]
    image_paths: tuple[str, ...]

    @property
    def image_count(self) -> int:
        return len(self.image_paths)


def _is_image(name: str) -> bool:
    return name.lower().endswith(tuple(SUPPORTED_RAW_EXTENSIONS))


def _counts(path: str) -> tuple[int, int]:
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


def scan_folder(path: str) -> FolderContents:
    """List one folder for browsing: subfolders with their counts, and its own images."""
    folders: list[FolderEntry] = []
    images: list[str] = []
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name.lower())
    except OSError:
        return FolderContents(path=path, folders=(), image_paths=())

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            image_count, subfolder_count = _counts(entry.path)
            folders.append(FolderEntry(entry.path, entry.name, image_count, subfolder_count))
        elif _is_image(entry.name) and not is_ir_sidecar_path(entry.path):
            images.append(entry.path)

    return FolderContents(path=path, folders=tuple(folders), image_paths=tuple(images))


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
