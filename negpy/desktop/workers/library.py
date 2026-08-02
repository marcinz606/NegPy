from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.kernel.system.logging import get_logger
from negpy.services.assets.library import LibraryWalkCache, search_library
from negpy.services.assets.search import parse_query

logger = get_logger(__name__)


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
