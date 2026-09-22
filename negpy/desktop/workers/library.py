from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.kernel.image.logic import calculate_file_hash
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.assets.library import LibraryWalkCache, search_library
from negpy.services.assets.search import parse_query

logger = get_logger(__name__)

# Seek-bound, like AssetDiscoveryWorker's own hashing pass: more concurrent readers
# than this just thrashes a spinning disk instead of finishing faster.
_HASH_WORKERS = min(8, APP_CONFIG.max_workers)


@dataclass(frozen=True)
class LibrarySearchTask:
    """Request to find library files matching a query, across the roots on disk."""

    roots: list[str]
    query: str
    configs_by_path: dict[str, Any] = field(default_factory=dict)
    marks_by_path: dict[str, str] = field(default_factory=dict)
    rewalk: bool = False  # drop the cached traversal first (folders changed on disk)


class LibrarySearchWorker(QObject):
    """Walks the library roots off the UI thread and matches the query against them.

    Holds the walk cache, so a run of searches in one session costs one traversal.
    It never hashes: identity is the loader's job, and this only has to answer
    "which paths".
    """

    progress = pyqtSignal(int)  # files walked so far
    finished = pyqtSignal(list)  # matching paths
    error = pyqtSignal(str)
    # Every walked file, hashed, unfiltered -- the caller checks which are already
    # embedded (it holds the repo, this worker never has), for search by meaning
    # across the whole library.
    indexing_scanned = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self._cache = LibraryWalkCache()

    @pyqtSlot()
    def invalidate(self) -> None:
        self._cache.invalidate()

    @pyqtSlot(LibrarySearchTask)
    def search(self, task: LibrarySearchTask) -> None:
        try:
            if task.rewalk:
                self._cache.invalidate()
            files = self._cache.files(list(task.roots), progress=self.progress.emit)
            self.finished.emit(search_library(files, parse_query(task.query), task.configs_by_path, task.marks_by_path))
        except Exception as exc:
            logger.exception("Library search failed")
            self.error.emit(str(exc))

    @pyqtSlot(list)
    def scan_for_indexing(self, roots: list) -> None:
        """Walks `roots` (the same cached traversal a keyword search already paid
        for) and hashes every file with the same bounded, no-RAW-decode fingerprint
        AssetDiscoveryWorker uses -- cheap enough to run over a whole library, unlike
        the decode+embed pass this only prepares the file list for."""
        try:
            files = self._cache.files(list(roots), progress=self.progress.emit)
            if not files:
                self.indexing_scanned.emit([])
                return
            paths = [f["path"] for f in files]
            if len(paths) < 2:
                hashes = [calculate_file_hash(p) for p in paths]
            else:
                hashes = [""] * len(paths)
                with ThreadPoolExecutor(max_workers=min(_HASH_WORKERS, len(paths))) as ex:
                    futures = {ex.submit(calculate_file_hash, p): i for i, p in enumerate(paths)}
                    for fut in as_completed(futures):
                        hashes[futures[fut]] = fut.result()
            self.indexing_scanned.emit([{**f, "hash": h} for f, h in zip(files, hashes)])
        except Exception as exc:
            logger.exception("Library indexing scan failed")
            self.error.emit(str(exc))
