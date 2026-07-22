import os
import sys
import tempfile
from typing import Callable

from PyQt6.QtCore import QEvent, QObject, Qt, qInstallMessageHandler
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog, QProxyStyle, QStyle

from negpy.desktop.frozen_entry import (
    dispatch_capture_helper as _dispatch_capture_helper,
    dispatch_packaging_smoke as _dispatch_packaging_smoke,
)
from negpy.kernel.system.config import APP_CONFIG, BASE_USER_DIR
from negpy.kernel.system.logging import get_logger, setup_logging
from negpy.kernel.system.override import apply as apply_override
from negpy.kernel.system.override import load_or_create as load_override
from negpy.kernel.system.parallel import configure_cpu_parallel
from negpy.kernel.system.paths import get_resource_path
from negpy.services.assets.crosstalk import CrosstalkProfiles
from negpy.services.assets.gear import GearProfiles
from negpy.services.repair.hybrid_runtime_manifest import (
    HybridRuntimeManifestError,
    load_default_hybrid_runtime_manifest,
)

logger = get_logger(__name__)

# A Finder-launched frozen macOS app can be stopped by the first access to
# ~/Documents before Qt has created a foreground window to host the TCC prompt.
# The handoff process below has no user-data side effects beyond a temporary,
# empty access probe; after consent it re-execs so the real process retains the
# existing pre-QApplication override/RHI/UI-scale ordering.
_MACOS_DOCUMENTS_READY_ENV = "NEGPY_MACOS_DOCUMENTS_READY"
_MACOS_ACCESS_PROBE_PREFIX = ".negpy-access-"

# qtawesome paints toolbar icons into a null pixmap when a button is asked to
# render before its first layout has given it valid geometry (e.g. while the
# startup "Restore Session" dialog spins a modal loop). The paint is harmless
# but Qt emits a fixed cascade of QPainter warnings. Drop exactly that cascade;
# forward every other Qt message to stderr unchanged.
_PAINTER_NOISE = (
    "QPainter::begin: Paint device returned engine == 0",
    "QPainter::save: Painter not active",
    "QPainter::setPen: Painter not active",
    "QPainter::setWorldTransform: Painter not active",
    "QPainter::setOpacity: Painter not active",
    "QPainter::setFont: Painter not active",
    "QPainter::setBrush: Painter not active",
    "QPainter::setClipRect: Painter not active",
    "QPainter::restore: Unbalanced save/restore",
)


def _filter_qt_messages(mode, context, message: str) -> None:
    if message.startswith(_PAINTER_NOISE):
        return
    sys.stderr.write(message + "\n")


class _AppStyle(QProxyStyle):
    """Fusion with a longer tooltip hover delay — the default 700 ms pops tooltips
    the moment the cursor crosses a toolbar, which reads as noise."""

    _TOOLTIP_WAKEUP_MS = 1400

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
            return self._TOOLTIP_WAKEUP_MS
        return super().styleHint(hint, option, widget, returnData)


class _ApplicationShutdownGate(QObject):
    """Consume every Qt quit request until scanner teardown is proven."""

    def __init__(self, allow_shutdown: Callable[[], bool]) -> None:
        super().__init__()
        self._allow_shutdown = allow_shutdown

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Quit and not self._allow_shutdown():
            return True
        return super().eventFilter(watched, event)


class _MacOSDocumentsHandoff(QProgressDialog):
    """A startup-only progress window that cannot dismiss the active TCC request."""

    def closeEvent(self, event) -> None:
        event.ignore()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            event.accept()
            return
        super().keyPressEvent(event)


def _probe_user_data_access(user_dir: str) -> None:
    """Prove that the configured user-data directory is writable without leaving a file behind."""

    os.makedirs(user_dir, exist_ok=True)
    fd: int | None = None
    probe_path: str | None = None
    try:
        fd, probe_path = tempfile.mkstemp(prefix=_MACOS_ACCESS_PROBE_PREFIX, dir=user_dir)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            finally:
                if probe_path is not None:
                    os.unlink(probe_path)


def _show_macos_documents_handoff() -> QProgressDialog:
    """Show the foreground, non-cancellable UI that makes a TCC prompt visible."""

    handoff = _MacOSDocumentsHandoff("Opening your existing NegPy workspace…", None, 0, 0)
    handoff.setWindowTitle("NegPy")
    handoff.setCancelButton(None)
    handoff.setAutoClose(False)
    handoff.setAutoReset(False)
    handoff.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
    handoff.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
    handoff.setWindowModality(Qt.WindowModality.ApplicationModal)
    handoff.show()
    return handoff


def _macos_frozen_documents_handoff() -> bool:
    """Handle the visible first-process permission check for a frozen macOS bundle.

    Returns True only when access was denied/unavailable and ``main`` must stop.
    On success this function replaces the process and does not return. The child
    consumes the one-process environment sentinel and follows normal startup.
    """

    if not (sys.platform == "darwin" and getattr(sys, "frozen", False)):
        return False
    if os.environ.pop(_MACOS_DOCUMENTS_READY_ENV, None) == "1":
        return False

    app = QApplication.instance() or QApplication(sys.argv)
    _handoff = _show_macos_documents_handoff()
    app.processEvents()

    try:
        _probe_user_data_access(BASE_USER_DIR)
    except OSError:
        QMessageBox.critical(
            None,
            "NegPy needs Documents access",
            "NegPy could not open your existing Documents/NegPy workspace. "
            "Your existing files were not changed. Grant NegPy access to the Documents folder "
            "in macOS Privacy & Security, then reopen the app.",
        )
        return True

    env = os.environ.copy()
    env[_MACOS_DOCUMENTS_READY_ENV] = "1"
    executable = os.path.abspath(sys.executable)
    os.execve(executable, [executable, *sys.argv[1:]], env)
    raise AssertionError("os.execve unexpectedly returned")


def _install_exception_hook() -> None:
    """Log every unhandled exception — especially ones raised inside a Qt slot — to the file log and
    show a non-fatal notice, instead of letting PyQt call qFatal() and abort with a native crash
    report that hides the Python traceback. This is what surfaces user-side bugs we can't reproduce
    (e.g. the Big Scanlight calibration crash): the traceback lands in negpy.log for them to attach."""

    def _hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
        try:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(
                None,
                "NegPy hit an error",
                f"Something went wrong and was logged:\n\n{exc_type.__name__}: {exc_value}\n\n"
                f"The app kept running. If it keeps happening, please attach the log file "
                f"({os.path.join(BASE_USER_DIR, 'negpy.log')}) to a bug report on GitHub.",
            )
        except Exception:
            logger.warning("could not show the error dialog", exc_info=True)

    sys.excepthook = _hook


def _bootstrap_environment() -> None:
    """Ensure user directories exist."""
    dirs = [
        BASE_USER_DIR,
        APP_CONFIG.presets_dir,
        APP_CONFIG.cache_dir,
        APP_CONFIG.user_icc_dir,
        APP_CONFIG.crosstalk_dir,
        APP_CONFIG.gear_dir,
        APP_CONFIG.contact_sheet_templates_dir,
        APP_CONFIG.default_export_dir,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    CrosstalkProfiles.ensure_user_dir()
    GearProfiles.ensure_user_dir()


def _load_desktop_hybrid_runtime():
    """Load the optional external companion without risking desktop startup."""

    try:
        runtime = load_default_hybrid_runtime_manifest()
    except HybridRuntimeManifestError as error:
        logger.error("Hybrid runtime disabled: %s", error)
        return None
    if runtime is None:
        logger.info("Hybrid runtime is not installed; exact Digital ICE remains available")
    else:
        try:
            runtime.validate_files()
        except ValueError as error:
            logger.error("Hybrid runtime disabled: %s", error)
            return None
        logger.info("Loaded pinned external hybrid runtime")
    return runtime


def main() -> None:
    """
    Desktop entry point.
    """
    if _dispatch_packaging_smoke(sys.argv) or _dispatch_capture_helper(sys.argv):
        return

    # A successful macOS handoff replaces this process. A denied handoff has
    # already shown a visible explanation and must not initialize a silent,
    # empty replacement workspace.
    if _macos_frozen_documents_handoff():
        return

    # Keep desktop-only imports behind the helper dispatch. A UI regression
    # must never prevent the frozen capture worker or packaging smoke from
    # starting in their isolated process.
    from negpy.desktop.controller import AppController
    from negpy.desktop.session import DesktopSessionManager
    from negpy.desktop.view.main_window import MainWindow
    from negpy.infrastructure.storage.repository import StorageRepository

    override_cfg = load_override(APP_CONFIG.override_toml_path)
    setup_logging(level=override_cfg.log_level_int)
    _install_exception_hook()  # log unhandled slot exceptions to negpy.log instead of aborting

    if getattr(sys, "frozen", False):
        log_path = os.path.join(os.path.expanduser("~"), "negpy_boot.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n--- Booting NegPy ---\n")

    try:
        os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

        apply_override(override_cfg, APP_CONFIG)
        # Multi-core Numba kernels: platform default (off on macOS) unless overridden.
        configure_cpu_parallel(APP_CONFIG.cpu_parallel)

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

        qInstallMessageHandler(_filter_qt_messages)
        app = QApplication(sys.argv)
        app.setApplicationName("NegPy")
        app.setStyle(_AppStyle("Fusion"))

        icon_path = get_resource_path("media/icons/icon.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))

        if os.path.exists(get_resource_path("negpy/desktop/view/styles/modern_dark.qss")):
            from negpy.desktop.view.styles.templates import load_stylesheet

            app.setStyleSheet(load_stylesheet())

        session_manager = DesktopSessionManager(repo)
        controller = AppController(
            session_manager,
            hybrid_runtime=_load_desktop_hybrid_runtime(),
        )

        window = MainWindow(controller)
        shutdown_gate = _ApplicationShutdownGate(
            window.request_shutdown_for_exit,
        )
        app.installEventFilter(shutdown_gate)
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
