import sys
import os
from PyQt6.QtWidgets import QApplication
from src.kernel.system.config import APP_CONFIG
from src.infrastructure.storage.repository import StorageRepository
from src.desktop.session import DesktopSessionManager
from src.desktop.controller import AppController
from src.desktop.view.main_window import MainWindow


def main() -> None:
    """
    Desktop entry point.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("Darkroom-Py")
    app.setStyle("Fusion")

    # Apply Theme
    qss_path = os.path.join(
        os.path.dirname(__file__), "view", "styles", "modern_dark.qss"
    )
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
