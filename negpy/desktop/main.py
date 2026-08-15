import os
import sys

from PyQt6.QtCore import Qt, qInstallMessageHandler
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QProxyStyle, QStyle

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.main_window import MainWindow
from negpy.features.flatfield.logic import set_gain_provider
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.crosstalk import CrosstalkProfiles
from negpy.services.assets.flatfield import FlatFieldProfiles
from negpy.services.assets.flatfield_migration import migrate_legacy_flatfield_profiles
from negpy.services.assets.gear import GearProfiles
from negpy.kernel.system.config import APP_CONFIG, BASE_USER_DIR
from negpy.kernel.system.logging import get_logger, setup_logging
from negpy.kernel.system.override import apply as apply_override
from negpy.kernel.system.override import load_or_create as load_override
from negpy.kernel.system.parallel import configure_cpu_parallel, parallel_enabled, resolve_cpu_parallel, set_parallel_enabled
from negpy.kernel.system.paths import get_resource_path

logger = get_logger(__name__)

# qtawesome paints toolbar icons into a null pixmap when a button is asked to render
# before its first layout has given it valid geometry, for example while the startup
# "Restore Session" dialog spins a modal loop. The paint is harmless, but Qt emits a
# fixed cascade of QPainter warnings. Drop exactly that cascade and forward every other
# Qt message to stderr unchanged.
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
    the moment the cursor crosses a toolbar, which reads as noise — and no mnemonic
    underlines on macOS, where they mark a key that does nothing."""

    _TOOLTIP_WAKEUP_MS = 1400

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
            return self._TOOLTIP_WAKEUP_MS
        if hint == QStyle.StyleHint.SH_UnderlineShortcut and sys.platform == "darwin":
            # Qt's own standard-button text carries the mnemonic ("&Yes"), but macOS has no mnemonic
            # convention, so QKeySequence::mnemonic() returns empty there and nothing is ever bound.
            # The native style answers this hint false and draws no underline, while Fusion, which we
            # force for the dark theme, answers true. The result is an underlined Y on a button no
            # key press can reach. Windows and Linux keep theirs, where Alt+letter does work.
            return 0
        return super().styleHint(hint, option, widget, returnData)


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
            from PyQt6.QtWidgets import QApplication, QMessageBox

            # Startup work runs before QApplication exists, and constructing a widget
            # without one makes Qt call qFatal() — an abort at the C level, with no Python
            # exception to catch, that buries the real failure under a complaint about
            # widgets. Print the traceback instead; there is no UI to notify yet.
            if QApplication.instance() is None:
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return

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


class UserDirectoryError(Exception):
    """NegPy has nowhere to keep its databases, caches and presets.

    Raised instead of the bare OSError so the startup path can say which directory it
    could not create and what to do about it, rather than aborting on a makedirs
    traceback the user cannot act on (issue #651).
    """

    def __init__(self, failed_dir: str, cause: OSError) -> None:
        self.failed_dir = failed_dir
        env_override = os.environ.get("NEGPY_USER_DIR")
        lines = [
            "NegPy could not create the directory it keeps its data in:",
            f"    {failed_dir}",
            f"    {cause.strerror or cause} (errno {cause.errno})",
        ]
        if env_override:
            lines += [
                "",
                f"NEGPY_USER_DIR is set to {env_override!r}, which is where that path comes from.",
                "It must be an absolute path on this machine that you can write to. If it came",
                "from .env.local, note that make does not expand ~ or $HOME — use $(HOME).",
            ]
        else:
            lines += [
                "",
                "Set NEGPY_USER_DIR to a directory you can write to, to start somewhere else.",
            ]
        super().__init__("\n".join(lines))


def _bootstrap_environment() -> None:
    """Ensure user directories exist."""
    dirs = [
        BASE_USER_DIR,
        APP_CONFIG.presets_dir,
        APP_CONFIG.cache_dir,
        APP_CONFIG.user_icc_dir,
        APP_CONFIG.crosstalk_dir,
        APP_CONFIG.sensor_dir,
        APP_CONFIG.gear_dir,
        APP_CONFIG.contact_sheet_templates_dir,
        APP_CONFIG.default_export_dir,
    ]
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            raise UserDirectoryError(d, e) from e
    CrosstalkProfiles.ensure_user_dir()
    GearProfiles.ensure_user_dir()


def _offer_to_disable_cpu_parallel(repo, parent) -> None:
    """The previous run had multi-core kernels on and did not exit cleanly. Say so.

    Numba's workqueue threading layer terminates the process on concurrent entry, with no
    Python exception to catch and nothing written to the log, so the app cannot report the
    crash as it happens — only notice afterwards that the last run never finished. Without
    this the setting is un-diagnosable: a user turns it on, hits an abort weeks later, and
    reports a crash nobody can connect to it.
    """
    from PyQt6.QtWidgets import QMessageBox

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("NegPy closed unexpectedly")
    box.setText("The last session ended unexpectedly with multi-core CPU rendering turned on.")
    box.setInformativeText(
        "That setting is the most likely cause. Turning it off costs some speed on merges "
        "and exports, and is the safe setting.\n\nTurn multi-core CPU rendering off?"
    )
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    if box.exec() == QMessageBox.StandardButton.Yes:
        repo.save_global_setting("cpu_parallel", False)
        set_parallel_enabled(False)
        logger.warning("CPU parallel kernels disabled after an unclean shutdown")


def main() -> None:
    """
    Desktop entry point.
    """
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

        try:
            _bootstrap_environment()
        except UserDirectoryError as e:
            # Nothing works without this directory, so stop here with an explanation
            # rather than carry a traceback into a startup that can only fail again on
            # the first database write. One line for the log, which may itself be
            # unwritable here; the full text to stderr, since the log handler repeats it.
            logger.critical("User directory unusable: %s", e.failed_dir)
            print(f"\n{e}\n", file=sys.stderr)
            sys.exit(1)

        # Storage (sqlite, no Qt dependency), created before QApplication so the saved UI scale
        # can be applied through QT_SCALE_FACTOR, which Qt reads only at startup.
        repo = StorageRepository(APP_CONFIG.edits_db_path, APP_CONFIG.settings_db_path)
        repo.initialize()

        # Multi-core Numba kernels: override.toml, then the saved setting, then the platform
        # default (off on macOS). Read before the flags below are overwritten.
        prev_clean_exit = bool(repo.get_global_setting("clean_shutdown", True))
        prev_run_parallel = bool(repo.get_global_setting("cpu_parallel_active", False))
        stored_parallel = repo.get_global_setting("cpu_parallel", None)
        configure_cpu_parallel(resolve_cpu_parallel(APP_CONFIG.cpu_parallel, None if stored_parallel is None else bool(stored_parallel)))
        # Numba's workqueue layer aborts the process outright on concurrent entry, with no
        # exception, no dialog and nothing in the log. A marker is the only way an abort can
        # later be attributed to this setting rather than reported as a mystery.
        repo.save_global_setting("clean_shutdown", False)
        repo.save_global_setting("cpu_parallel_active", parallel_enabled())

        # Resolve flat-field gains by profile id from the on-disk store, which keeps the numpy
        # logic layer free of any storage dependency, then migrate any legacy DB-backed profiles
        # into that store.
        set_gain_provider(FlatFieldProfiles.load_gain)
        migrate_legacy_flatfield_profiles(repo)

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
        controller = AppController(session_manager)

        window = MainWindow(controller)
        if prev_clean_exit is False and prev_run_parallel:
            _offer_to_disable_cpu_parallel(repo, window)
        window.show()

        exit_code = app.exec()
        controller.cleanup()
        repo.save_global_setting("clean_shutdown", True)
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
