import unittest
from unittest.mock import MagicMock
from dataclasses import replace

from negpy.desktop.session import DesktopSessionManager
from negpy.domain.models import WorkspaceConfig, GeometryConfig, RetouchConfig, ProcessConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.config import APP_CONFIG


class TestDesktopSessionSync(unittest.TestCase):
    def setUp(self):
        self.mock_repo = MagicMock(spec=StorageRepository)
        self.mock_repo.load_file_settings.return_value = None

        # Mock global settings with correct types
        def mock_get_global(key, default=None):
            if key == "last_export_config":
                return {}
            if key == "process_mode":
                return "C41"
            return default

        self.mock_repo.get_global_setting.side_effect = mock_get_global
        self.mock_repo.get_max_history_index.return_value = 0
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
            process=ProcessConfig(process_mode="E-6", e6_normalize=True),
        )
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = source_config

        target_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=0.0),
            geometry=GeometryConfig(rotation=0, fine_rotation=0.0, manual_crop_rect=None),
            retouch=RetouchConfig(dust_remove=False, manual_dust_spots=[]),
            process=ProcessConfig(process_mode="C41", e6_normalize=False),
        )
        self.mock_repo.load_file_settings.return_value = target_config

        self.session.update_selection([0, 1])
        self.session.sync_selected_settings()

        args, _ = self.mock_repo.save_file_settings.call_args
        self.assertEqual(args[0], "hash2")
        saved_config = args[1]

        self.assertEqual(saved_config.exposure.density, 1.5)
        self.assertEqual(saved_config.geometry.rotation, 1)
        self.assertEqual(saved_config.process.process_mode, "E-6")
        self.assertTrue(saved_config.process.e6_normalize)

        # Excluded fields
        self.assertEqual(saved_config.geometry.fine_rotation, 0.0)
        self.assertIsNone(saved_config.geometry.manual_crop_rect)
        self.assertEqual(saved_config.retouch.manual_dust_spots, [])
        self.assertTrue(saved_config.retouch.dust_remove)

    def test_undo_redo_persistence(self):
        self.session.select_file(0)
        initial_config = self.session.state.config

        # 1. First edit
        new_config_1 = replace(initial_config, exposure=replace(initial_config.exposure, density=1.5))
        self.session.update_config(new_config_1, persist=True)

        # Verify push to history (pushed initial state)
        self.mock_repo.save_history_step.assert_called_with("hash1", 0, initial_config)
        self.assertEqual(self.session.state.undo_index, 1)

        # 2. Undo
        self.mock_repo.load_history_step.return_value = initial_config
        self.session.undo()
        self.assertEqual(self.session.state.config.exposure.density, initial_config.exposure.density)
        self.assertEqual(self.session.state.undo_index, 0)

        # 3. Redo
        self.mock_repo.load_history_step.return_value = new_config_1
        self.session.redo()
        self.assertEqual(self.session.state.config.exposure.density, 1.5)
        self.assertEqual(self.session.state.undo_index, 1)

    def test_history_pruning(self):
        self.session.select_file(0)
        # Perform steps slightly over the limit
        num_edits = APP_CONFIG.max_history_steps + 2
        for i in range(num_edits):
            cfg = replace(self.session.state.config, exposure=replace(self.session.state.config.exposure, density=float(i)))
            self.session.update_config(cfg, persist=True)

        # Should have called prune_history
        self.mock_repo.prune_history.assert_called()
        self.assertGreater(self.session.state.undo_index, APP_CONFIG.max_history_steps)

    def test_history_restoration_on_file_switch(self):
        # 1. Mock file having 5 history steps in DB
        self.mock_repo.get_max_history_index.return_value = 5

        # 2. Select file
        self.session.select_file(1)

        # 3. Verify session state recovered the index
        self.assertEqual(self.session.state.undo_index, 5)
        self.assertEqual(self.session.state.max_history_index, 5)


if __name__ == "__main__":
    unittest.main()
