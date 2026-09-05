import os
import pytest

# Configure headless mode for CI/CD
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-runner"


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("metrics", "negpy performance metrics export")
    g.addoption(
        "--metrics-out",
        action="store",
        default=None,
        help="Write session metrics to this JSON path (overrides NEGPY_METRICS_OUT if set as non-empty).",
    )


@pytest.fixture(scope="session", autouse=True)
def qapp():
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    yield app
    app.quit()
    app.processEvents()


class FakeRepo:
    """Real global-setting storage; everything else a sidebar pokes (flat-field profiles,
    presets, …) falls through to a MagicMock."""

    def __init__(self, **data):
        from unittest.mock import MagicMock

        self._mock = MagicMock()
        self.data = dict(data)

    def get_global_setting(self, key, default=None):
        return self.data.get(key, default)

    def save_global_setting(self, key, value):
        self.data[key] = value

    def __getattr__(self, name):
        return getattr(self._mock, name)


class FakeController:
    """For panels too big for the plain MagicMock idiom: ControlsPanel and FileBrowser connect
    to real signals and hand real models to Qt, which a bare mock can't satisfy."""

    def __init__(self, repo=None):
        from unittest.mock import MagicMock

        from negpy.desktop.session import AppState

        self._mock = MagicMock()
        self.state = AppState()
        self.session = self._mock.session
        self.session.repo = repo if repo is not None else FakeRepo()
        self.session.state = self.state
        self.config_updated = self._mock.config_updated
        self.image_updated = self._mock.image_updated
        self.tool_sync_requested = self._mock.tool_sync_requested

    def __getattr__(self, name):
        return getattr(self._mock, name)


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtestloop(session):
    """Stop background threads before pytest-cov generates its coverage report.

    trylast=True means this wrapper's post-yield runs *before* pytest-cov's,
    giving us a window to quit Qt threads and destroy the wgpu device before
    GC destroys the Qt thread wrappers — preventing the SIGABRT on CI.
    """
    yield
    try:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            app.quit()
            app.processEvents()
    except Exception:
        pass
    try:
        from negpy.infrastructure.gpu.device import GPUDevice

        GPUDevice.destroy_singleton()
    except Exception:
        pass
