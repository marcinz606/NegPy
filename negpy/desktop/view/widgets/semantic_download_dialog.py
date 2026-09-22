"""First-run CLIP model download for search by meaning: opt-in, cancellable, and
built on update_dialog.py's DownloadWorker idiom rather than the release-specific
code around it.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from negpy.desktop.view.styles.templates import pin_dialog_default, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.update_dialog import _own
from negpy.kernel.system.logging import get_logger
from negpy.services.assets.semantic_model import MODEL_DOWNLOAD_SIZE, ClipDownloadError, download_clip_model

logger = get_logger(__name__)


def _mb(value: int) -> str:
    return f"{value / 1_048_576:.0f} MB"


class ClipDownloadWorker(QThread):
    """Streams the CLIP model files, reporting bytes as they land."""

    progress = pyqtSignal(int, int)
    ready = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            download_clip_model(
                on_progress=lambda done, total: self.progress.emit(done, total),
                is_cancelled=lambda: self._cancelled,
            )
        except ClipDownloadError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            logger.exception("CLIP model download failed")
            self.failed.emit(f"Download failed: {exc}")
        else:
            self.ready.emit()


class ClipDownloadDialog(QDialog):
    """accept() once the model is on disk; reject() on Cancel or a closed window --
    the caller only turns the preference on after accept()."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._worker: Optional[ClipDownloadWorker] = None

        self.setWindowTitle("Search by Meaning")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(420, 180)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(THEME.space_xl)

        heading = QLabel("Download the Model")
        heading.setStyleSheet(f"color: {THEME.text_primary}; font-size: {THEME.font_size_title}px; font-weight: bold;")
        root.addWidget(heading)

        self.subtitle = QLabel(f"A one-time download ({MODEL_DOWNLOAD_SIZE}) lets the Film Strip search by plain-language description.")
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_base}px;")
        root.addWidget(self.subtitle)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;")
        self.status.setVisible(False)
        root.addWidget(self.status)

        self.bar = QProgressBar()
        self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 0)
        self.bar.setVisible(False)
        self.bar.setStyleSheet(f"""
            QProgressBar {{ background-color: {THEME.border_primary}; border: none; border-radius: {THEME.radius_sm}px; }}
            QProgressBar::chunk {{ background-color: {THEME.status_success}; border-radius: {THEME.radius_sm}px; }}
        """)
        root.addWidget(self.bar)

        actions = QHBoxLayout()
        actions.addStretch()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip(wrap_tooltip("Stop the download and close. Search by meaning stays off."))
        self.cancel_button.clicked.connect(self.reject)
        actions.addWidget(self.cancel_button)
        self.download_button = QPushButton("Download")
        self.download_button.setToolTip(wrap_tooltip("Fetch the model once; it is cached for every later search"))
        self.download_button.clicked.connect(self._on_download)
        actions.addWidget(self.download_button)
        pin_dialog_default(self.download_button, self.cancel_button)
        root.addLayout(actions)

    def _on_download(self) -> None:
        self.download_button.setEnabled(False)
        self.bar.setVisible(True)
        self._set_status("Downloading…")

        self._worker = ClipDownloadWorker()
        self._worker.progress.connect(self._on_progress)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        _own(self._worker)
        self._worker.start()

    def _set_status(self, text: str) -> None:
        self.status.setText(text)
        self.status.setVisible(True)

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            if self.bar.maximum() != total:
                self.bar.setRange(0, total)
            self.bar.setValue(done)
            self._set_status(f"Downloading — {_mb(done)} of {_mb(total)}")

    def _on_ready(self) -> None:
        self.accept()

    def _on_failed(self, message: str) -> None:
        self.bar.setVisible(False)
        self._set_status(message)
        self.download_button.setEnabled(True)

    def reject(self) -> None:
        # The worker outlives this window (see _own in update_dialog.py), so closing
        # need not block on a socket read. It stops at the next chunk and drops the
        # part file.
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        super().reject()
