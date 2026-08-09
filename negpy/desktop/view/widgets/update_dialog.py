"""The update window behind the sidebar's "Update Available" notice.

Shows the release notes, downloads the asset that matches this install, and hands
it to `updater.apply_update`. NegPy then closes: the swap only happens once this
process is gone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.updater import (
    UpdateError,
    UpdateInfo,
    apply_update,
    download_asset,
    find_update,
    staging_dir,
)
from negpy.kernel.system.version import get_app_version

logger = get_logger(__name__)


class UpdateCheckWorker(QThread):
    """Background release check. `checked` always fires — with None when the running
    version is current or the check could not reach GitHub."""

    checked = pyqtSignal(object)

    def run(self) -> None:
        try:
            info = find_update()
        except Exception:
            logger.exception("Update check failed")
            info = None
        self.checked.emit(info)


# Every network thread this module starts, kept here for its whole life. A QThread
# destroyed while it still runs takes the process down with it, and a stalled socket
# outlives the panel or dialog that asked for it — so neither owns one.
_RUNNING: list[QThread] = []
_QUIT_HOOKED = False


def _wait_for_checks() -> None:
    for worker in _RUNNING:
        worker.wait(6000)


def _own(worker: QThread) -> None:
    """Take ownership of `worker` and make sure the app waits for it on the way out."""
    global _QUIT_HOOKED

    _RUNNING.append(worker)
    app = QApplication.instance()
    if app is not None and not _QUIT_HOOKED:
        _QUIT_HOOKED = True
        app.aboutToQuit.connect(_wait_for_checks)


def start_update_check(on_checked: Callable[[Optional[UpdateInfo]], None]) -> None:
    """Run the release check off the UI thread, then hand `on_checked` the result.

    Qt drops the connection by itself once the receiver is gone, so a check that
    outlives the panel that started it simply reports to nobody.
    """
    worker = UpdateCheckWorker()
    worker.checked.connect(on_checked)
    _own(worker)
    worker.start()


class DownloadWorker(QThread):
    """Streams the release asset, reporting bytes as they land."""

    progress = pyqtSignal(int, int)
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, info: UpdateInfo) -> None:
        super().__init__()
        self._info = info
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            path = download_asset(
                self._info,
                staging_dir(),
                on_progress=lambda done, total: self.progress.emit(done, total),
                is_cancelled=lambda: self._cancelled,
            )
        except UpdateError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            logger.exception("Update download failed")
            self.failed.emit(f"Download failed: {exc}")
        else:
            self.ready.emit(path)


def _mb(value: int) -> str:
    return f"{value / 1_048_576:.0f} MB"


class UpdateDialog(QDialog):
    """Release notes, one button, and a progress bar for the swap."""

    def __init__(self, info: UpdateInfo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.info = info
        self._worker: Optional[DownloadWorker] = None

        self.setWindowTitle(f"NegPy {info.version} is available")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(640, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(THEME.space_xl)

        heading = QLabel(f"NegPy {info.version}")
        heading.setStyleSheet(f"color: {THEME.text_primary}; font-size: {THEME.font_size_title}px; font-weight: bold;")
        root.addWidget(heading)

        subtitle = f"You are on {get_app_version()}."
        if info.can_self_install:
            subtitle += f" NegPy can install this for you ({_mb(info.size)} download)."
        else:
            subtitle += " Download it from the releases page to update."
        self.subtitle = QLabel(subtitle)
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_base}px;")
        root.addWidget(self.subtitle)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {THEME.border_color};")
        root.addWidget(divider)

        self.notes = QTextBrowser()
        self.notes.setReadOnly(True)
        self.notes.setOpenLinks(False)
        self.notes.setFrameShape(QFrame.Shape.NoFrame)
        self.notes.setStyleSheet(
            f"QTextBrowser {{ background: transparent; border: none; color: {THEME.text_secondary}; font-size: {THEME.font_size_base}px; }}"
        )
        self.notes.setMarkdown(info.notes or "_No release notes._")
        root.addWidget(self.notes, stretch=1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px;")
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
        self.page_button = QPushButton("Release Notes on GitHub")
        self.page_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.info.page_url)))
        actions.addWidget(self.page_button)
        actions.addStretch()

        self.later_button = QPushButton("Later")
        self.later_button.clicked.connect(self.reject)
        actions.addWidget(self.later_button)

        self.install_button = QPushButton("Install Update" if info.can_self_install else "Open Releases Page")
        self.install_button.setProperty("primary", True)
        self.install_button.setDefault(True)
        self.install_button.clicked.connect(self._on_install)
        actions.addWidget(self.install_button)
        root.addLayout(actions)

    def _on_install(self) -> None:
        if not self.info.can_self_install:
            QDesktopServices.openUrl(QUrl(self.info.page_url))
            self.accept()
            return

        self.install_button.setEnabled(False)
        self.later_button.setText("Cancel")
        self.bar.setVisible(True)
        self._set_status(f"Downloading {self.info.asset_name}…")

        self._worker = DownloadWorker(self.info)
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
            self._set_status(f"Downloading {self.info.asset_name} — {_mb(done)} of {_mb(total)}")

    def _on_ready(self, path: Path) -> None:
        self.later_button.setEnabled(False)
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        self._set_status("Installing — NegPy will close and reopen on the new version.")

        try:
            apply_update(path, self.info)
        except UpdateError as exc:
            self._on_failed(str(exc))
            return
        except Exception as exc:
            logger.exception("Failed to start the installer")
            self._on_failed(f"Could not start the installer: {exc}")
            return

        self.accept()
        app = QApplication.instance()
        if app is not None:
            app.closeAllWindows()
            app.quit()

    def _on_failed(self, message: str) -> None:
        self.bar.setVisible(False)
        self._set_status(f"{message}\nYou can still download it from the releases page.")
        self.install_button.setEnabled(True)
        self.later_button.setEnabled(True)
        self.later_button.setText("Close")

    def reject(self) -> None:
        # The worker outlives this window (see `_own`), so closing need not block on
        # a socket read: it stops at the next chunk and drops the part file.
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        super().reject()
