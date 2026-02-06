import unittest
from unittest.mock import MagicMock
from dataclasses import replace

from negpy.desktop.session import DesktopSessionManager
from negpy.domain.models import WorkspaceConfig, GeometryConfig, RetouchConfig
from negpy.infrastructure.storage.repository import StorageRepository


class TestDesktopSessionSync(unittest.TestCase):
    def setUp(self):
        self.mock_repo = MagicMock(spec=StorageRepository)
        self.mock_repo.load_file_settings.return_value = None
        self.session = DesktopSessionManager(self.mock_repo)

        self.session.state.uploaded_files = [
            {"name": "file1.dng", "path": "path1", "hash": "hash1"},
            {"name": "file2.dng", "path": "path2", "hash": "hash2"},
        ]

    def test_update_selection(self):
        self.session.update_selection([0, 1])
        self.assertEqual(self.session.state.selected_indices, [0, 1])

    def test_select_file_updates_selection(self):
        self.session.select_file(1)
        self.assertEqual(self.session.state.selected_file_idx, 1)
        self.assertEqual(self.session.state.selected_indices, [1])

    def test_sync_selected_settings_exclusions(self):
        source_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=1.5),
            geometry=GeometryConfig(rotation=1, fine_rotation=5.5, manual_crop_rect=(0, 0, 1, 1)),
            retouch=RetouchConfig(dust_remove=True, manual_dust_spots=[(0.1, 0.1, 5)]),
        )
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = source_config

        target_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=0.0),
            geometry=GeometryConfig(rotation=0, fine_rotation=0.0, manual_crop_rect=None),
            retouch=RetouchConfig(dust_remove=False, manual_dust_spots=[]),
        )
        self.mock_repo.load_file_settings.return_value = target_config

        self.session.update_selection([0, 1])
        self.session.sync_selected_settings()

        args, _ = self.mock_repo.save_file_settings.call_args
        self.assertEqual(args[0], "hash2")
        saved_config = args[1]

        self.assertEqual(saved_config.exposure.density, 1.5)
        self.assertEqual(saved_config.geometry.rotation, 1)

        # Excluded fields
        self.assertEqual(saved_config.geometry.fine_rotation, 0.0)
        self.assertIsNone(saved_config.geometry.manual_crop_rect)
        self.assertEqual(saved_config.retouch.manual_dust_spots, [])
        self.assertTrue(saved_config.retouch.dust_remove)


if __name__ == "__main__":
    unittest.main()
