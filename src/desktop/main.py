import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from src.kernel.system.config import APP_CONFIG, BASE_USER_DIR
from src.kernel.system.paths import get_resource_path
from src.infrastructure.storage.repository import StorageRepository
from src.desktop.session import DesktopSessionManager
from src.desktop.controller import AppController
from src.desktop.view.main_window import MainWindow
from src.kernel.system.logging import setup_logging, get_logger

logger = get_logger(__name__)


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
    setup_logging()

    if getattr(sys, "frozen", False):
        log_path = os.path.join(os.path.expanduser("~"), "negpy_boot.log")
        with open(log_path, "a") as f:
            f.write("\n--- Booting NegPy ---\n")

    try:
        os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

        # Platform-specific safeguards for display and GPU stability
        if sys.platform == "linux":
            if "QT_X11_NO_MITSHM" not in os.environ:
                os.environ["QT_X11_NO_MITSHM"] = "1"
            if "WGPU_BACKEND_TYPE" not in os.environ:
                os.environ["WGPU_BACKEND_TYPE"] = "Vulkan"

        _bootstrap_environment()

        if hasattr(Qt.HighDpiScaleFactorRoundingPolicy, "PassThrough"):
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )

        app = QApplication(sys.argv)
        app.setApplicationName("NegPy")
        app.setStyle("Fusion")

        icon_path = get_resource_path("media/icons/icon.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))

        qss_path = get_resource_path("src/desktop/view/styles/modern_dark.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                app.setStyleSheet(f.read())

        repo = StorageRepository(APP_CONFIG.edits_db_path, APP_CONFIG.settings_db_path)
        repo.initialize()

        session_manager = DesktopSessionManager(repo)
        controller = AppController(session_manager)

        window = MainWindow(controller)
        window.show()

        exit_code = app.exec()
        controller.cleanup()
        sys.exit(exit_code)
    except Exception as e:
        if getattr(sys, "frozen", False):
            import traceback

            log_path = os.path.join(os.path.expanduser("~"), "negpy_boot.log")
            with open(log_path, "a") as f:
                f.write(f"CRASH: {str(e)}\n")
                f.write(traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()
