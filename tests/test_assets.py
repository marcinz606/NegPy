import os
import pytest
from src.backend.assets import AssetManager
from src.config import APP_CONFIG


@pytest.fixture
def mock_asset_dir(tmp_path):
    # Override cache_dir for testing
    old_dir = APP_CONFIG.cache_dir
    test_dir = str(tmp_path / "test_cache")
    APP_CONFIG.cache_dir = test_dir
    yield test_dir
    APP_CONFIG.cache_dir = old_dir


def test_asset_manager_lifecycle(mock_asset_dir):
    # 1. Initialize
    AssetManager.initialize()
    assert os.path.exists(mock_asset_dir)

    # 2. Persist
    class MockUploadedFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content

        def getbuffer(self):
            return self.content

    mock_file = MockUploadedFile("test.arw", b"dummy_raw_data")
    path, f_hash = AssetManager.persist(mock_file, "session_1")

    assert path is not None
    assert os.path.exists(path)
    assert f_hash is not None
    assert "session_1" in path

    # 3. Remove
    AssetManager.remove(path)
    assert not os.path.exists(path)

    # 4. Clear All
    AssetManager.persist(mock_file, "session_2")
    AssetManager.clear_all()
    # Check that session subdirs are gone but base dir exists
    assert os.path.exists(mock_asset_dir)
    assert len(os.listdir(mock_asset_dir)) == 0
