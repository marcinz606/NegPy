import os
from src.kernel.system.version import get_app_version
from src.kernel.system.paths import get_resource_path


def test_get_app_version(tmp_path):
    # Mock VERSION file
    version_dir = tmp_path / "src" / "kernel" / "system"
    version_dir.mkdir(parents=True)

    # We need to mock get_resource_path or just hope it finds VERSION in root
    # Since we are in tests, root is the current directory.
    v = get_app_version()
    assert isinstance(v, str)


def test_get_resource_path():
    p = get_resource_path("src/kernel/system/paths.py")
    assert os.path.exists(p)
    assert os.path.isabs(p)
