import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from src.kernel.system.config import APP_CONFIG, BASE_USER_DIR
from src.kernel.system.paths import get_resource_path
from src.infrastructure.storage.repository import StorageRepository
from src.desktop.session import DesktopSessionManager
from src.desktop.controller import AppController
from src.desktop.view.main_window import MainWindow


def _bootstrap_environment() -> None:
    """Ensure user directories exist."""
    dirs = [
        BASE_USER_DIR,
        APP_CONFIG.presets_dir,
        APP_CONFIG.cache_dir,
        APP_CONFIG.user_icc_dir,
        APP_CONFIG.default_export_dir,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main() -> None:
    """
    Desktop entry point.
    """
    _bootstrap_environment()

    app = QApplication(sys.argv)
    app.setApplicationName("NegPy")
    app.setStyle("Fusion")

    # Window Icon
    icon_path = get_resource_path("media/icons/icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Apply Theme
    qss_path = get_resource_path("src/desktop/view/styles/modern_dark.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            app.setStyleSheet(f.read())

    # Initialize Core Services
    repo = StorageRepository(APP_CONFIG.edits_db_path, APP_CONFIG.settings_db_path)
    repo.initialize()

    session_manager = DesktopSessionManager(repo)
    controller = AppController(session_manager)

    # Setup UI
    window = MainWindow(controller)
    window.show()

    # Handle graceful exit
    exit_code = app.exec()
    controller.cleanup()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
