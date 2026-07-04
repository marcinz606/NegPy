import unittest
from unittest.mock import MagicMock
from dataclasses import replace

from negpy.desktop.session import AppState, AssetListModel, DesktopSessionManager
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

    def test_set_autodetect_enabled_persists(self):
        self.assertFalse(self.session.state.autodetect_enabled)
        self.session.set_autodetect_enabled(True)
        self.assertTrue(self.session.state.autodetect_enabled)
        self.mock_repo.save_global_setting.assert_called_with("autodetect_enabled", True)

    def test_set_autodetect_enabled_noop_when_unchanged(self):
        self.session.set_autodetect_enabled(False)
        self.mock_repo.save_global_setting.assert_not_called()

    def test_persist_writes_sticky_settings_in_one_batch(self):
        self.session.select_file(0)
        self.mock_repo.save_global_settings.reset_mock()

        cfg = replace(self.session.state.config, exposure=replace(self.session.state.config.exposure, density=1.5))
        self.session.update_config(cfg, persist=True)

        self.mock_repo.save_global_settings.assert_called_once()
        saved = self.mock_repo.save_global_settings.call_args.args[0]
        self.assertEqual(saved["last_density"], 1.5)
        self.assertIn("last_process_mode", saved)
        self.assertIn("last_export_config", saved)
        self.assertIn("last_dust_remove", saved)

    def test_processing_toggles_carry_to_new_files(self):
        # Globally remembered toggles must be applied to a fresh (sidecar-less) file.
        sticky = {
            "last_export_config": {},
            "last_auto_exposure": True,
            "last_auto_normalize_contrast": True,
            "last_cast_removal_strength": 0.5,
            "last_auto_cast_removal": True,
            "last_paper_dmin": True,
            "last_surround": True,
            "last_paper_profile": "ilford_mg_rc",
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertTrue(config.exposure.auto_exposure)
        self.assertTrue(config.exposure.auto_normalize_contrast)
        self.assertEqual(config.exposure.cast_removal_strength, 0.5)
        self.assertTrue(config.exposure.auto_cast_removal)
        self.assertTrue(config.exposure.paper_dmin)
        self.assertTrue(config.exposure.surround)
        self.assertEqual(config.exposure.paper_profile, "ilford_mg_rc")

    def test_roll_average_not_seeded_onto_fresh_files(self):
        # A roll baseline must not leak onto a fresh (sidecar-less) file.
        sticky = {
            "last_export_config": {},
            "last_use_luma_average": True,
            "last_use_colour_average": True,
            "last_locked_floors": [0.1, 0.2, 0.3],
            "last_locked_ceils": [1.1, 1.2, 1.3],
            "last_roll_name": "roll-A",
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertFalse(config.process.use_luma_average)
        self.assertFalse(config.process.use_colour_average)
        self.assertFalse(config.process.is_locked_initialized)
        self.assertIsNone(config.process.roll_name)

    def test_roll_average_not_persisted_as_global_sticky(self):
        # The five roll-average keys must no longer be written to global settings.
        self.session.select_file(0)
        self.mock_repo.save_global_settings.reset_mock()
        base = self.session.state.config
        loaded = replace(
            base,
            process=replace(
                base.process,
                use_luma_average=True,
                use_colour_average=True,
                locked_floors=(0.1, 0.2, 0.3),
                locked_ceils=(1.1, 1.2, 1.3),
                roll_name="roll-A",
            ),
        )
        self.session.update_config(loaded, persist=True)
        saved = self.mock_repo.save_global_settings.call_args.args[0]
        for key in ("last_use_luma_average", "last_use_colour_average", "last_locked_floors", "last_locked_ceils", "last_roll_name"):
            self.assertNotIn(key, saved)

    def test_saved_file_keeps_its_own_roll_baseline(self):
        # A file whose own config carries the baseline (only_global=True) is untouched.
        base = WorkspaceConfig(
            process=replace(
                WorkspaceConfig().process,
                use_luma_average=True,
                use_colour_average=True,
                locked_floors=(0.1, 0.2, 0.3),
                locked_ceils=(1.1, 1.2, 1.3),
            )
        )
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: {"last_export_config": {}}.get(key, default)
        config = self.session._apply_sticky_settings(base, only_global=True)
        self.assertTrue(config.process.use_luma_average)
        self.assertTrue(config.process.is_locked_initialized)

    def test_processing_toggles_not_applied_to_edited_files(self):
        # only_global=True (file has a sidecar) must not override per-file toggles.
        sticky = {"last_export_config": {}, "last_auto_exposure": True}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        base = WorkspaceConfig(exposure=replace(WorkspaceConfig().exposure, auto_exposure=False))
        config = self.session._apply_sticky_settings(base, only_global=True)
        self.assertFalse(config.exposure.auto_exposure)

    def test_temp_lock_reaims_wb_on_load(self):
        from negpy.features.exposure.logic import _TEMP_K_MAGENTA, _TEMP_K_YELLOW, wb_to_kelvin

        sticky = {"last_export_config": {}, "wb_temp_lock": 4500.0}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        base = WorkspaceConfig(exposure=replace(WorkspaceConfig().exposure, wb_magenta=0.2, wb_yellow=0.1))
        # Both branches re-aim: saved files (only_global=True) and fresh files.
        for only_global in (True, False):
            cfg = self.session._apply_sticky_settings(base, only_global=only_global)
            m, y = cfg.exposure.wb_magenta, cfg.exposure.wb_yellow
            self.assertAlmostEqual(wb_to_kelvin(m, y), 4500.0, places=4)
            # Re-aim moves along the locus only: the frame's tint is preserved.
            self.assertAlmostEqual((m - 0.2) / _TEMP_K_MAGENTA, (y - 0.1) / _TEMP_K_YELLOW, places=6)

    def test_temp_lock_absent_leaves_wb(self):
        base = WorkspaceConfig(exposure=replace(WorkspaceConfig().exposure, wb_magenta=0.2, wb_yellow=0.1))
        cfg = self.session._apply_sticky_settings(base, only_global=True)
        self.assertEqual(cfg.exposure.wb_magenta, 0.2)
        self.assertEqual(cfg.exposure.wb_yellow, 0.1)

    def test_contact_sheet_output_path_in_sticky_export(self):
        sticky = {
            "last_export_config": {"contact_sheet_output_path": "/saved/contact", "contact_sheet_cell_px": 800},
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(config.export.contact_sheet_output_path, "/saved/contact")
        self.assertEqual(config.export.contact_sheet_cell_px, 800)

    def test_contact_sheet_template_in_sticky_export(self):
        sticky = {
            "last_export_config": {
                "contact_sheet_template": "Tight 35mm",
                "contact_sheet_cell_px": 400,
                "contact_sheet_default_cell_px": 550,
            },
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(config.export.contact_sheet_template, "Tight 35mm")
        self.assertEqual(config.export.contact_sheet_cell_px, 400)
        self.assertEqual(config.export.contact_sheet_default_cell_px, 550)

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
        self.session.sync_selected_settings(frozenset({"process", "exposure", "color", "finish"}))

        args, _ = self.mock_repo.save_file_settings.call_args
        self.assertEqual(args[0], "hash2")
        saved_config = args[1]

        self.assertEqual(saved_config.exposure.density, 1.5)
        self.assertEqual(saved_config.process.process_mode, "E-6")
        self.assertTrue(saved_config.process.e6_normalize)

        # Geometry entirely preserved from target
        self.assertEqual(saved_config.geometry.rotation, 0)
        self.assertEqual(saved_config.geometry.fine_rotation, 0.0)
        self.assertIsNone(saved_config.geometry.manual_crop_rect)
        # Per-file retouch fields preserved from target
        self.assertEqual(saved_config.retouch.manual_dust_spots, [])
        self.assertTrue(saved_config.retouch.dust_remove)

    def test_sync_selected_settings_edits_with_geometry(self):
        source_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=1.5),
            geometry=GeometryConfig(rotation=1, fine_rotation=5.5, manual_crop_rect=(0.1, 0.1, 0.9, 0.9)),
            retouch=RetouchConfig(dust_remove=True, manual_dust_spots=[(0.1, 0.1, 5)]),
            process=ProcessConfig(process_mode="E-6", e6_normalize=True),
        )
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = source_config

        target_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=0.0),
            geometry=GeometryConfig(rotation=0, fine_rotation=0.0, manual_crop_rect=None),
            retouch=RetouchConfig(dust_remove=False, manual_dust_spots=[(0.5, 0.5, 3)]),
            process=ProcessConfig(process_mode="C41", e6_normalize=False),
        )
        self.mock_repo.load_file_settings.return_value = target_config

        self.session.update_selection([0, 1])
        self.session.sync_selected_settings(frozenset({"process", "exposure", "color", "finish", "crop", "rotation"}))

        args, _ = self.mock_repo.save_file_settings.call_args
        saved_config = args[1]

        # Crop and fine_rotation should now propagate from source
        self.assertEqual(saved_config.geometry.fine_rotation, 5.5)
        self.assertEqual(saved_config.geometry.manual_crop_rect, (0.1, 0.1, 0.9, 0.9))
        self.assertEqual(saved_config.geometry.rotation, 1)
        # Edits still synced
        self.assertEqual(saved_config.exposure.density, 1.5)
        # Dust spots still per-target
        self.assertEqual(saved_config.retouch.manual_dust_spots, [(0.5, 0.5, 3)])

    def test_sync_selected_settings_geometry_only(self):
        source_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=1.5),
            geometry=GeometryConfig(rotation=2, fine_rotation=3.0, manual_crop_rect=(0.0, 0.0, 0.5, 0.5)),
        )
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = source_config

        target_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=0.7),
            geometry=GeometryConfig(rotation=0, fine_rotation=0.0, manual_crop_rect=None),
        )
        self.mock_repo.load_file_settings.return_value = target_config

        self.session.update_selection([0, 1])
        self.session.sync_selected_settings(frozenset({"crop", "rotation"}))

        args, _ = self.mock_repo.save_file_settings.call_args
        saved_config = args[1]

        # Geometry comes from source
        self.assertEqual(saved_config.geometry.rotation, 2)
        self.assertEqual(saved_config.geometry.fine_rotation, 3.0)
        self.assertEqual(saved_config.geometry.manual_crop_rect, (0.0, 0.0, 0.5, 0.5))
        # Other config preserved from target
        self.assertEqual(saved_config.exposure.density, 0.7)

    def test_sync_selected_settings_invalid_aspect_is_noop(self):
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.update_selection([0, 1])
        self.session.sync_selected_settings(frozenset({"bogus"}))
        self.mock_repo.save_file_settings.assert_not_called()

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

    def _last_session_manifest(self):
        """Returns (paths, active_path) from the most recent _persist_session calls."""
        saved = {c.args[0]: c.args[1] for c in self.mock_repo.save_global_setting.call_args_list}
        return saved.get("session_files"), saved.get("session_active_path")

    def test_select_file_persists_manifest(self):
        self.session.select_file(1)
        paths, active = self._last_session_manifest()
        self.assertEqual(paths, ["path1", "path2"])
        self.assertEqual(active, "path2")

    def test_active_file_changing_snapshots_outgoing_when_dirty(self):
        # Fires before state mutates to the new file, carrying the outgoing identity.
        self.session.state.current_file_hash = "hash1"
        self.session.state.is_dirty = True
        seen = []
        self.session.active_file_changing.connect(lambda: seen.append(self.session.state.current_file_hash))
        self.session.select_file(1)
        self.assertEqual(seen, ["hash1"])

    def test_active_file_changing_not_emitted_when_clean(self):
        self.session.state.current_file_hash = "hash1"
        self.session.state.is_dirty = False
        fired = []
        self.session.active_file_changing.connect(lambda: fired.append(True))
        self.session.select_file(1)
        self.assertEqual(fired, [])

    def test_clear_files_persists_empty_manifest(self):
        self.session.clear_files()
        paths, active = self._last_session_manifest()
        self.assertEqual(paths, [])
        self.assertIsNone(active)


class TestAssetListModelFilter(unittest.TestCase):
    def setUp(self):
        self.state = AppState()
        self.state.uploaded_files = [
            {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
            {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
            {"name": "image.NEF", "path": "/tmp/image.NEF", "hash": "h3"},
            {"name": "note.txt", "path": "/tmp/note.txt", "hash": "h4"},
            {"name": "scan_42.tif", "path": "/tmp/scan_42.tif", "hash": "h5"},
        ]
        self.model = AssetListModel(self.state)

    def _names(self):
        return [self.state.uploaded_files[i]["name"] for i in self.model._sorted_indices]

    def test_empty_filter_shows_all(self):
        self.model.set_filter("", regex=False)
        self.assertEqual(len(self.model._sorted_indices), 5)

    def test_plain_substring_case_insensitive(self):
        ok = self.model.set_filter("IMG", regex=False)
        self.assertTrue(ok)
        self.assertEqual(set(self._names()), {"IMG_0001.cr2", "IMG_0002.cr2"})

    def test_plain_substring_matches_unrelated_prefix(self):
        self.model.set_filter("scan", regex=False)
        self.assertEqual(set(self._names()), {"scan_42.tif"})

    def test_plain_extension_match(self):
        self.model.set_filter(".cr2", regex=False)
        self.assertEqual(set(self._names()), {"IMG_0001.cr2", "IMG_0002.cr2"})

    def test_plain_no_match(self):
        self.model.set_filter("zzzzz", regex=False)
        self.assertEqual(self.model.rowCount(), 0)
        self.assertEqual(self.model._sorted_indices, [])

    def test_regex_success(self):
        ok = self.model.set_filter(r"^IMG_\d{4}\.cr2$", regex=True)
        self.assertTrue(ok)
        self.assertEqual(set(self._names()), {"IMG_0001.cr2", "IMG_0002.cr2"})

    def test_regex_invalid_preserves_previous_filter(self):
        self.model.set_filter("img", regex=False)
        before = list(self.model._sorted_indices)
        ok = self.model.set_filter("[", regex=True)
        self.assertFalse(ok)
        self.assertEqual(self.model._sorted_indices, before)

    def test_filter_after_sort_descending(self):
        self.model.set_sort_order("name")
        self.model.set_sort_descending(True)
        self.model.set_filter(".cr2", regex=False)
        self.assertEqual(self._names(), ["IMG_0002.cr2", "IMG_0001.cr2"])

    def test_display_actual_roundtrip_with_filter(self):
        self.model.set_filter("img", regex=False)
        for display in range(self.model.rowCount()):
            actual = self.model.display_to_actual(display)
            self.assertEqual(self.model.actual_to_display(actual), display)

    def test_visible_actual_indices_ordered(self):
        self.model.set_sort_order("name")
        self.model.set_sort_descending(False)
        self.model.set_filter(".cr2", regex=False)
        self.assertEqual(self.model.visible_actual_indices_ordered(), self.model._sorted_indices)
        self.assertEqual(self.model.visible_actual_indices(), set(self.model._sorted_indices))

    def test_filter_persists_through_refresh(self):
        self.model.set_filter("IMG", regex=False)
        self.state.uploaded_files.append({"name": "extra.txt", "path": "/tmp/extra.txt", "hash": "h6"})
        self.model.refresh()
        self.assertNotIn("extra.txt", self._names())
        self.assertEqual(set(self._names()), {"IMG_0001.cr2", "IMG_0002.cr2"})

    def test_clearing_filter_restores_full_list(self):
        self.model.set_filter("IMG", regex=False)
        self.model.set_filter("", regex=False)
        self.assertEqual(len(self.model._sorted_indices), 5)


if __name__ == "__main__":
    unittest.main()
