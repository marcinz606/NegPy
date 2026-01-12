import unittest
from unittest.mock import MagicMock, patch
from src.domain.session import WorkspaceSession
from src.domain.models import WorkspaceConfig, ExportConfig
from src.domain.interfaces import IRepository, IAssetStore
from src.services.rendering.engine import DarkroomEngine


class TestWorkspaceSession(unittest.TestCase):
    def setUp(self):
        self.mock_repo = MagicMock(spec=IRepository)
        self.mock_store = MagicMock(spec=IAssetStore)
        self.mock_engine = MagicMock(spec=DarkroomEngine)
        self.session_id = "test_session_123"

        self.session = WorkspaceSession(
            self.session_id, self.mock_repo, self.mock_store, self.mock_engine
        )

    @patch("src.kernel.system.config.APP_CONFIG")
    def test_create_default_config_uses_app_config_path(self, mock_app_config):
        """
        Verify env-dependent export path injection.
        """
        expected_path = "/custom/env/path/export"
        mock_app_config.default_export_dir = expected_path

        config = self.session.create_default_config()

        self.assertIsInstance(config, WorkspaceConfig)
        self.assertIsInstance(config.export, ExportConfig)
        self.assertEqual(config.export.export_path, expected_path)

    @patch("src.kernel.system.config.APP_CONFIG")
    def test_create_default_config_defaults(self, mock_app_config):
        """
        Verify static defaults (e.g. Lab settings).
        """
        mock_app_config.default_export_dir = "/tmp/export"

        config = self.session.create_default_config()

        self.assertEqual(config.lab.color_separation, 2.0)
        self.assertEqual(config.retouch.dust_size, 3)
        self.assertEqual(config.export.export_fmt, "JPEG")
        self.assertEqual(config.export.export_print_size, 30.0)
        self.assertEqual(config.export.export_dpi, 300)

    def test_get_active_settings_creates_defaults_if_empty(self):
        """
        Missing DB entry -> Fresh default config.
        """
        self.session.uploaded_files = [
            {"name": "test.dng", "path": "/tmp/test.dng", "hash": "abc123hash"}
        ]
        self.session.selected_file_idx = 0
        self.mock_repo.load_file_settings.return_value = None

        settings = self.session.get_active_settings()

        self.assertIsNotNone(settings)
        self.mock_repo.load_file_settings.assert_called_with("abc123hash")
        self.assertEqual(settings.lab.sharpen, 0.25)
        self.assertEqual(self.session.file_settings["abc123hash"], settings)

    def test_get_active_settings_returns_saved_settings(self):
        """
        DB entry exists -> Return saved config.
        """
        self.session.uploaded_files = [
            {"name": "test.dng", "path": "/tmp/test.dng", "hash": "saved_hash"}
        ]

        saved_config = self.session.create_default_config()
        self.mock_repo.load_file_settings.return_value = saved_config

        result = self.session.get_active_settings()

        self.assertEqual(result, saved_config)


if __name__ == "__main__":
    unittest.main()
