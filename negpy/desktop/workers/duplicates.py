import threading
from typing import Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.kernel.system.logging import get_logger
from negpy.services.assets.duplicates import FINGERPRINT_EDGE, find_groups, fingerprint
from negpy.services.assets.thumbnails import decode_bounded_source_preview

logger = get_logger(__name__)


class DuplicateWorker(QObject):
    """Fingerprints frames and groups the ones scanned more than once, off the UI thread."""

    progress = pyqtSignal(int, int, str)  # current, total, label
    found = pyqtSignal(object)  # List[DuplicateGroup]
    cancelled = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._cancel = threading.Event()

    @pyqtSlot()
    def cancel(self) -> None:
        self._cancel.set()

    @pyqtSlot(object)
    def run(self, paths: Tuple[str, ...]) -> None:
        self._cancel.clear()
        prints = {}
        for i, path in enumerate(paths):
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            self.progress.emit(i, len(paths), path.rsplit("/", 1)[-1])
            try:
                # Twice the fingerprint edge, so the fingerprint is an area average.
                img = decode_bounded_source_preview(path, max_edge=2 * FINGERPRINT_EDGE, should_cancel=self._cancel.is_set)
            except InterruptedError:
                self.cancelled.emit()
                return
            except Exception as e:
                logger.warning("Find Duplicates could not read %s: %s", path, e)
                continue
            if img is not None:
                prints[path] = fingerprint(np.asarray(img))
        self.progress.emit(len(paths), len(paths), "")
        self.found.emit(find_groups(prints))
