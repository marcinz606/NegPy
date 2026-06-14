import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.main_window import MainWindow
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.crosstalk import CrosstalkProfiles
from negpy.kernel.system.config import APP_CONFIG, BASE_USER_DIR
from negpy.kernel.system.logging import get_logger, setup_logging
from negpy.kernel.system.override import apply as apply_override
from negpy.kernel.system.override import load_or_create as load_override
from negpy.kernel.system.paths import get_resource_path

logger = get_logger(__name__)


def _bootstrap_environment() -> None:
    """Ensure user directories exist."""
    dirs = [
        BASE_USER_DIR,
        APP_CONFIG.presets_dir,
        APP_CONFIG.cache_dir,
        APP_CONFIG.user_icc_dir,
        APP_CONFIG.crosstalk_dir,
        APP_CONFIG.default_export_dir,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    CrosstalkProfiles.seed_example()


def main() -> None:
    """
    Desktop entry point.
    """
    override_cfg = load_override(APP_CONFIG.override_toml_path)
    setup_logging(level=override_cfg.log_level_int)

    if getattr(sys, "frozen", False):
        log_path = os.path.join(os.path.expanduser("~"), "negpy_boot.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n--- Booting NegPy ---\n")

    try:
        os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

        apply_override(override_cfg, APP_CONFIG)

        _bootstrap_environment()

        # Storage (sqlite, no Qt dependency) — created before QApplication so the saved
        # UI scale can be applied via QT_SCALE_FACTOR, which Qt only reads at startup.
        repo = StorageRepository(APP_CONFIG.edits_db_path, APP_CONFIG.settings_db_path)
        repo.initialize()

        scale = float(repo.get_global_setting("ui_scale", 1.0) or 1.0)
        scale = max(0.8, min(1.2, scale))
        if scale != 1.0 and "QT_SCALE_FACTOR" not in os.environ:
            os.environ["QT_SCALE_FACTOR"] = f"{scale:.2f}"

        # Global attributes for Windows stability
        if sys.platform == "win32":
            QCoreApplication = getattr(sys.modules["PyQt6.QtCore"], "QCoreApplication")
            QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings)

        app = QApplication(sys.argv)
        app.setApplicationName("NegPy")
        app.setStyle("Fusion")

        icon_path = get_resource_path("media/icons/icon.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))

        qss_path = get_resource_path("negpy/desktop/view/styles/modern_dark.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())

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
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"CRASH: {str(e)}\n")
                f.write(traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()
