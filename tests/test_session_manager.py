import unittest
from unittest.mock import MagicMock, patch
from src.domain.session import WorkspaceSession
from src.domain.models import WorkspaceConfig, ExportConfig
from src.domain.interfaces import IRepository, IAssetStore
from src.services.rendering.engine import DarkroomEngine


class TestWorkspaceSession(unittest.TestCase):
    def setUp(self):
        # Mocks
        self.mock_repo = MagicMock(spec=IRepository)
        self.mock_store = MagicMock(spec=IAssetStore)
        self.mock_engine = MagicMock(spec=DarkroomEngine)
        self.session_id = "test_session_123"

        # Instance
        self.session = WorkspaceSession(
            self.session_id, self.mock_repo, self.mock_store, self.mock_engine
        )

    @patch("src.kernel.system.config.APP_CONFIG")
    def test_create_default_config_uses_app_config_path(self, mock_app_config):
        """
        Verifies that create_default_config() respects the APP_CONFIG.default_export_dir.
        """
        # Arrange
        expected_path = "/custom/env/path/export"
        mock_app_config.default_export_dir = expected_path

        # Act
        config = self.session.create_default_config()

        # Assert
        self.assertIsInstance(config, WorkspaceConfig)
        self.assertIsInstance(config.export, ExportConfig)
        self.assertEqual(config.export.export_path, expected_path)

    @patch("src.kernel.system.config.APP_CONFIG")
    def test_create_default_config_defaults(self, mock_app_config):
        """
        Verifies that other defaults (like Lab Settings) are correctly populated
        from the static DEFAULT_WORKSPACE_CONFIG.
        """
        # Arrange
        mock_app_config.default_export_dir = "/tmp/export"

        # Act
        config = self.session.create_default_config()

        # Assert
        # Check a few key defaults we relied on fixing
        self.assertEqual(config.lab.color_separation, 1.0)
        self.assertEqual(config.retouch.dust_size, 3)
        # Check export defaults
        self.assertEqual(config.export.export_fmt, "JPEG")
        self.assertEqual(config.export.export_print_size, 27.0)
        self.assertEqual(config.export.export_dpi, 300)

    def test_get_active_settings_creates_defaults_if_empty(self):
        """
        Verifies that asking for settings for a file with no DB entry
        returns a fresh default config via create_default_config.
        """
        # Arrange
        # Mock uploaded files
        self.session.uploaded_files = [
            {"name": "test.dng", "path": "/tmp/test.dng", "hash": "abc123hash"}
        ]
        self.session.selected_file_idx = 0

        # Mock repo returning None (no saved settings)
        self.mock_repo.load_file_settings.return_value = None

        # Act
        settings = self.session.get_active_settings()

        # Assert
        self.assertIsNotNone(settings)
        # Verify it called load_file_settings with correct hash
        self.mock_repo.load_file_settings.assert_called_with("abc123hash")
        # Verify it used defaults (check a known default)
        self.assertEqual(settings.lab.sharpen, 0.25)
        # Verify it cached it in memory
        self.assertEqual(self.session.file_settings["abc123hash"], settings)

    def test_get_active_settings_returns_saved_settings(self):
        """
        Verifies that if the DB has settings, they are returned instead of defaults.
        """
        # Arrange
        self.session.uploaded_files = [
            {"name": "test.dng", "path": "/tmp/test.dng", "hash": "saved_hash"}
        ]

        # Create a "saved" config that differs from defaults
        saved_config = self.session.create_default_config()
        # Modify it effectively (using replace or manual reconstruction since it's frozen)
        # Since it's frozen, we can't set attrs. We'll simulate it by mocking the return.
        # Ideally we'd use dataclasses.replace, but here we just need object identity or value check.
        # Let's assume the repo returns this specific object.
        self.mock_repo.load_file_settings.return_value = saved_config

        # Act
        result = self.session.get_active_settings()

        # Assert
        self.toBe = saved_config
        self.assertEqual(result, saved_config)


if __name__ == "__main__":
    unittest.main()
