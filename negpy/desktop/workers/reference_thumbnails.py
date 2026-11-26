"""Thumbnails of reference photos, made off the UI thread."""

from __future__ import annotations

from PyQt6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QImage, QImageReader

# Two threads: a HEIC decode is slow, and the dialog only needs the cards in view.
_THREADS = 2


def reference_thumbnail(path: str, edge: int) -> QImage:
    """An upright image that fits in edge x edge. A null image when Qt cannot read the file:
    HEIC needs the macOS image plugin."""
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    if not reader.canRead():
        return QImage()
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > edge:
        reader.setScaledSize(size.scaled(QSize(edge, edge), Qt.AspectRatioMode.KeepAspectRatio))
    return reader.read()


class _Job(QRunnable):
    def __init__(self, path: str, edge: int, owner: "ReferenceThumbnails") -> None:
        super().__init__()
        self._path = path
        self._edge = edge
        self._owner = owner

    def run(self) -> None:
        self._owner.loaded.emit(self._path, reference_thumbnail(self._path, self._edge))


class ReferenceThumbnails(QObject):
    """Emits `loaded(path, image)` for each requested path. A QImage is safe across threads."""

    loaded = pyqtSignal(str, QImage)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(_THREADS)

    def request(self, path: str, edge: int) -> None:
        self._pool.start(_Job(path, edge, self))

    def stop(self) -> None:
        self._pool.clear()
        self._pool.waitForDone()
