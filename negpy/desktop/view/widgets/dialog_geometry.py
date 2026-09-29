"""Remembered size and position for dialogs."""

from PyQt6.QtCore import QByteArray, QEvent, QObject
from PyQt6.QtWidgets import QDialog

from negpy.domain.interfaces import IRepository
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = "dialog_geometry_"


def remember_dialog_geometry(dialog: QDialog, repo: IRepository | None, name: str) -> None:
    """Open the dialog at the size and position it had when it last closed, and keep them current.

    Call last in __init__, after the default size and the window flags and before the first
    show(). A stored value that does not parse keeps the default. repo=None keeps the dialog
    but not its geometry.
    """
    if repo is None:
        return
    key = _KEY_PREFIX + name
    encoded = repo.get_global_setting(key)
    if isinstance(encoded, str) and encoded:
        dialog.restoreGeometry(QByteArray.fromBase64(encoded.encode("utf-8")))
    dialog.installEventFilter(_GeometryKeeper(dialog, repo, key))


class _GeometryKeeper(QObject):
    """Saves the dialog's geometry each time it hides.

    It uses the watched object, not a stored reference: a window still open at quit hides
    from its C++ destructor, after its Python wrapper is gone.
    """

    def __init__(self, dialog: QDialog, repo: IRepository, key: str) -> None:
        super().__init__(dialog)
        self._repo = repo
        self._key = key

    def eventFilter(self, obj, event) -> bool:
        try:
            # A window-system minimize is a spontaneous hide; the widget still counts as visible.
            if event.type() == QEvent.Type.Hide and not event.spontaneous() and not obj.isMinimized():
                self._repo.save_global_setting(self._key, bytes(obj.saveGeometry().toBase64()).decode("ascii"))
        except Exception:
            # An exception inside a Qt virtual aborts the process.
            logger.exception("Failed to persist dialog geometry")
        return False
