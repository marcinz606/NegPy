import os
import pytest

# Configure headless mode for CI/CD
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-runner"


@pytest.fixture(scope="session", autouse=True)
def qapp():
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    yield app
