import unittest
from unittest.mock import MagicMock, patch
from dataclasses import replace

from negpy.desktop.session import AppState, AssetListModel, DesktopSessionManager
from negpy.domain.models import WorkspaceConfig, GeometryConfig, RetouchConfig, ProcessConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.config import APP_CONFIG


class TestDesktopSessionSync(unittest.TestCase):
    def setUp(self):
        self.mock_repo = MagicMock(spec=StorageRepository)
        self.mock_repo.load_file_settings.return_value = None
        self.mock_repo.load_file_settings_by_path.return_value = None

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

    def test_rediscovery_refreshes_same_path_in_place(self):
        refreshed = {
            "name": "file1 (RGB)",
            "path": "path1",
            "hash": "fresh-hash",
            "green_path": "path1-g",
            "blue_path": "path1-b",
        }

        self.session.add_files([], validated_info=[refreshed])

        self.assertEqual(len(self.session.state.uploaded_files), 2)
        self.assertEqual(self.session.state.uploaded_files[0], refreshed)

    def test_config_for_asset_saved_uses_saved_edits_and_global_overlays_only(self):
        defaults = WorkspaceConfig()
        saved = replace(
            defaults,
            exposure=replace(defaults.exposure, density=1.7),
            process=replace(defaults.process, process_mode="E-6"),
            geometry=replace(defaults.geometry, autocrop_ratio="4:3"),
        )
        sticky = {
            "last_export_config": {"jpeg_quality": 73},
            "last_protect_original_metadata": True,
            # Workflow defaults must not overwrite an edited/saved asset.
            "last_process_mode": "C41",
            "last_aspect_ratio": "1:1",
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        asset = {"name": "saved.dng", "path": "/roll/saved.dng", "hash": "saved-hash"}
        active_before = self.session.state.config

        with patch("negpy.desktop.session.load_or_promote", return_value=saved) as hydrate:
            config = self.session.config_for_asset(asset)

        hydrate.assert_called_once_with(self.mock_repo, "saved-hash", "/roll/saved.dng", half=0)
        self.assertEqual(config.exposure.density, 1.7)
        self.assertEqual(config.process.process_mode, "E-6")
        self.assertEqual(config.geometry.autocrop_ratio, "4:3")
        self.assertEqual(config.export.jpeg_quality, 73)
        self.assertTrue(config.metadata.protect_original_metadata)
        self.assertIs(self.session.state.config, active_before)

    def test_config_for_asset_fresh_starts_clean_not_from_active_creative_edits(self):
        defaults = WorkspaceConfig()
        active = replace(
            defaults,
            exposure=replace(defaults.exposure, density=2.4),
            lab=replace(defaults.lab, saturation=1.8),
        )
        self.session.state.config = active
        sticky = {
            "last_export_config": {},
            "last_process_mode": "E-6",
            "last_aspect_ratio": "1:1",
            "last_autocrop_offset": 7,
            "last_auto_exposure": True,
            "last_narrowband_scan": True,
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        asset = {"name": "fresh.dng", "path": "/roll/fresh.dng", "hash": "fresh-hash"}

        with patch("negpy.desktop.session.load_or_promote", return_value=None):
            config = self.session.config_for_asset(asset)

        self.assertEqual(config.exposure.density, defaults.exposure.density)
        self.assertEqual(config.lab.saturation, defaults.lab.saturation)
        self.assertTrue(config.exposure.auto_exposure)
        self.assertTrue(config.process.narrowband_scan)
        self.assertEqual(config.process.process_mode, "E-6")
        self.assertEqual(config.geometry.autocrop_ratio, "1:1")
        self.assertEqual(config.geometry.autocrop_offset, 7)
        self.assertIs(self.session.state.config, active)

    def test_config_for_asset_resolves_triplet_per_asset_and_resets_plain_asset(self):
        leaked = replace(
            WorkspaceConfig(),
            rgbscan=RgbScanConfig(enabled=True, green_path="/stale/g.dng", blue_path="/stale/b.dng", align=True),
        )
        self.session.state.config = leaked
        triplet = {
            "name": "triplet.dng",
            "path": "/roll/r.dng",
            "hash": "triplet-hash",
            "green_path": "/roll/g.dng",
            "blue_path": "/roll/b.dng",
            "align": False,
        }
        plain = {"name": "plain.dng", "path": "/roll/plain.dng", "hash": "plain-hash"}

        with patch("negpy.desktop.session.load_or_promote", side_effect=[leaked, leaked]):
            triplet_config = self.session.config_for_asset(triplet)
            plain_config = self.session.config_for_asset(plain)

        self.assertEqual(
            triplet_config.rgbscan,
            RgbScanConfig(enabled=True, green_path="/roll/g.dng", blue_path="/roll/b.dng", align=False),
        )
        self.assertEqual(plain_config.rgbscan, RgbScanConfig())
        self.assertIs(self.session.state.config, leaked)

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
        self.assertIn("last_paper_black", saved)
        self.assertIn("last_narrowband_scan", saved)
        self.assertIn("last_protect_original_metadata", saved)
        self.assertIn("last_cast_removal_strength", saved)

    def test_persist_active_batch_config_saves_before_exposing_state(self):
        original = self.session.state.config
        updated = replace(original, geometry=replace(original.geometry, fine_rotation=1.25))
        self.session.state.current_file_hash = "hash1"
        self.session.state.current_file_path = "path1"
        saved_signals = []
        self.session.settings_saved.connect(lambda: saved_signals.append(True))

        self.session.persist_active_batch_config(updated)

        self.mock_repo.save_file_settings.assert_called_with("hash1", updated, file_path="path1")
        self.assertIs(self.session.state.config, updated)
        self.assertTrue(self.session.state.is_dirty)
        self.assertEqual(saved_signals, [True])

    def test_persist_active_batch_config_keeps_state_unchanged_on_failure(self):
        original = self.session.state.config
        updated = replace(original, geometry=replace(original.geometry, fine_rotation=1.25))
        self.session.state.current_file_hash = "hash1"
        self.session.state.current_file_path = "path1"
        self.mock_repo.save_file_settings.side_effect = RuntimeError("database unavailable")

        with self.assertRaises(RuntimeError):
            self.session.persist_active_batch_config(updated)

        self.assertIs(self.session.state.config, original)
        self.assertFalse(self.session.state.is_dirty)

    def test_protect_original_metadata_carries_globally(self):
        sticky = {
            "last_export_config": {},
            "last_protect_original_metadata": True,
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertTrue(config.metadata.protect_original_metadata)

    def test_protect_original_metadata_applied_to_saved_files(self):
        sticky = {
            "last_export_config": {},
            "last_protect_original_metadata": True,
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        base = WorkspaceConfig(metadata=replace(WorkspaceConfig().metadata, protect_original_metadata=False))
        config = self.session._apply_sticky_settings(base, only_global=True)
        self.assertTrue(config.metadata.protect_original_metadata)

    def test_processing_toggles_carry_to_new_files(self):
        # Globally remembered toggles must be applied to a fresh (sidecar-less) file.
        sticky = {
            "last_export_config": {},
            "last_auto_exposure": True,
            "last_auto_normalize_contrast": True,
            "last_paper_dmin": True,
            "last_paper_black": True,
            "last_paper_profile": "ilford_mg_rc",
            "last_cast_removal_strength": 0.8,
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertTrue(config.exposure.auto_exposure)
        self.assertTrue(config.exposure.auto_normalize_contrast)
        self.assertTrue(config.exposure.paper_dmin)
        self.assertTrue(config.exposure.paper_black)
        self.assertEqual(config.exposure.paper_profile, "ilford_mg_rc")
        self.assertEqual(config.exposure.cast_removal_strength, 0.8)

    def test_cast_removal_zero_carries_to_new_files(self):
        """Sticky must carry an explicit zero, not just non-zero — default is 0.5."""
        sticky = {"last_export_config": {}, "last_cast_removal_strength": 0.0}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(config.exposure.cast_removal_strength, 0.0)

    def test_paper_black_carries_to_new_files(self):
        """Sticky must carry an explicit value over the file's base."""
        sticky = {"last_export_config": {}, "last_paper_black": False}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        base = WorkspaceConfig(exposure=replace(WorkspaceConfig().exposure, paper_black=True))
        config = self.session._apply_sticky_settings(base, only_global=False)
        self.assertFalse(config.exposure.paper_black)

    def test_legacy_true_black_sticky_migrates_inverted(self):
        """A pre-rename sticky (last_true_black) maps to paper_black inverted."""
        sticky = {"last_export_config": {}, "last_true_black": False}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertTrue(config.exposure.paper_black)

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

    def _seed_roll(self):
        self.session.state.uploaded_files = [
            {"name": "a.arw", "path": "pa", "hash": "hash1"},
            {"name": "b.arw", "path": "pb", "hash": "hash2"},
            {"name": "c.jpg", "path": "pc", "hash": "hash3"},
        ]
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = WorkspaceConfig()
        self.mock_repo.load_file_settings.return_value = WorkspaceConfig()

    def test_sync_roll_scope_respects_active_filter(self):
        # A filename filter is a view; "whole roll" applies only to visible frames.
        self._seed_roll()
        self.session.asset_model.set_filter(".arw", regex=False)  # hides c.jpg

        count = self.session.sync_selected_settings(frozenset({"exposure"}), scope="roll")

        saved = {c.args[0] for c in self.mock_repo.save_file_settings.call_args_list}
        self.assertEqual(count, 1)
        self.assertEqual(saved, {"hash2"})  # source + filtered-out c.jpg excluded

    def test_sync_roll_scope_no_filter_covers_all(self):
        self._seed_roll()
        self.session.asset_model.refresh()  # no filter → every frame visible

        count = self.session.sync_selected_settings(frozenset({"exposure"}), scope="roll")

        saved = {c.args[0] for c in self.mock_repo.save_file_settings.call_args_list}
        self.assertEqual(count, 2)
        self.assertEqual(saved, {"hash2", "hash3"})

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

    def test_reset_settings_drops_edits_and_bounds(self):
        self.session.select_file(0)
        dirty = replace(
            self.session.state.config,
            exposure=replace(self.session.state.config.exposure, density=1.8, grade=140.0),
            process=replace(
                self.session.state.config.process,
                local_floors=(0.1, 0.2, 0.3),
                local_ceils=(0.7, 0.8, 0.9),
                locked_floors=(0.05, 0.05, 0.05),
                locked_ceils=(0.95, 0.95, 0.95),
                lock_bounds=True,
            ),
            geometry=replace(self.session.state.config.geometry, rotation=2, manual_crop_rect=(0.1, 0.1, 0.9, 0.9)),
        )
        self.session.update_config(dirty, persist=True)

        self.session.reset_settings()

        self.assertEqual(self.session.state.config, WorkspaceConfig())
        self.assertFalse(self.session.state.config.process.is_local_initialized)
        self.assertFalse(self.session.state.config.process.is_locked_initialized)

    def test_reset_settings_is_recorded_not_wiping(self):
        self.session.select_file(0)
        edited = replace(self.session.state.config, exposure=replace(self.session.state.config.exposure, density=1.8))
        self.session.update_config(edited, persist=True)

        self.session.reset_settings()

        self.mock_repo.clear_history.assert_not_called()
        self.assertEqual(self.session.state.config, WorkspaceConfig())
        # Reset pushed the pre-reset config as a history step — it is undoable.
        self.mock_repo.save_history_step.assert_called_with("hash1", 1, edited)
        self.assertEqual(self.session.state.undo_index, 2)

    def test_sync_to_roll_records_target_history(self):
        self.mock_repo.get_max_history_index.return_value = 0
        self.mock_repo.load_history_step.return_value = None
        self.session.select_file(0)
        self.session.state.uploaded_files.append({"name": "file3.dng", "path": "path3", "hash": "hash3"})
        self.session.asset_model.refresh()
        self.mock_repo.save_history_step.reset_mock()

        count = self.session.sync_selected_settings(frozenset({"exposure"}), scope="roll")

        self.assertEqual(count, 2)
        # Each target got a two-step write: pre-apply at 0, post-apply at 1.
        steps = [(c.args[0], c.args[1]) for c in self.mock_repo.save_history_step.call_args_list]
        self.assertEqual(steps, [("hash2", 0), ("hash2", 1), ("hash3", 0), ("hash3", 1)])

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


class TestNavButtonBoundaries(unittest.TestCase):
    """Regression for #407: Next/Prev enable must be computed in display space
    (sorted/filtered order), matching session.next_file/prev_file — not raw
    uploaded_files (load-order) index space."""

    def setUp(self):
        self.state = AppState()
        # Loaded reverse-alphabetically; default sort is name-ascending, so
        # display order is the reverse of load order.
        self.state.uploaded_files = [
            {"name": "c.dng", "path": "/tmp/c.dng", "hash": "h1"},
            {"name": "b.dng", "path": "/tmp/b.dng", "hash": "h2"},
            {"name": "a.dng", "path": "/tmp/a.dng", "hash": "h3"},
        ]
        self.model = AssetListModel(self.state)

    def _enabled(self, actual_idx):
        display_idx = self.model.actual_to_display(actual_idx)
        prev_enabled = display_idx > 0
        next_enabled = 0 <= display_idx < self.model.rowCount() - 1
        return prev_enabled, next_enabled

    def test_first_display_file_is_last_loaded(self):
        # a.dng (actual 2) is the last-loaded but first in display order.
        prev_enabled, next_enabled = self._enabled(2)
        self.assertFalse(prev_enabled)
        self.assertTrue(next_enabled)

    def test_last_display_file_is_first_loaded(self):
        # c.dng (actual 0) is the first-loaded but last in display order.
        prev_enabled, next_enabled = self._enabled(0)
        self.assertTrue(prev_enabled)
        self.assertFalse(next_enabled)

    def test_filtered_out_selection_disables_both(self):
        self.model.set_filter("a.dng", regex=False)
        prev_enabled, next_enabled = self._enabled(0)  # c.dng no longer visible
        self.assertFalse(prev_enabled)
        self.assertFalse(next_enabled)


class TestSessionEmptied(unittest.TestCase):
    """Removing the last file must fully reset the active-image state and emit
    session_emptied, so the viewer blanks instead of keeping an unremovable frame."""

    def setUp(self):
        self.mock_repo = MagicMock(spec=StorageRepository)
        self.mock_repo.load_file_settings.return_value = None
        self.mock_repo.load_file_settings_by_path.return_value = None
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: default
        self.mock_repo.get_max_history_index.return_value = 0
        self.session = DesktopSessionManager(self.mock_repo)

        self.session.state.uploaded_files = [{"name": "file1.dng", "path": "path1", "hash": "hash1"}]
        self.session.state.selected_file_idx = 0
        self.session.state.selected_indices = [0]
        self.session.state.current_file_path = "path1"
        self.session.state.current_file_hash = "hash1"
        self.session.state.preview_raw = object()
        self.session.state.last_metrics["base_positive"] = object()

        self.emptied_count = 0
        self.session.session_emptied.connect(self._on_emptied)

    def _on_emptied(self):
        self.emptied_count += 1

    def _assert_active_image_reset(self):
        state = self.session.state
        self.assertEqual(state.selected_file_idx, -1)
        self.assertEqual(state.selected_indices, [])
        self.assertIsNone(state.current_file_path)
        self.assertIsNone(state.current_file_hash)
        self.assertIsNone(state.preview_raw)
        self.assertEqual(state.last_metrics, {})
        self.assertEqual(state.config, WorkspaceConfig())

    def test_remove_current_last_file_emits_and_resets(self):
        self.session.remove_current_file()
        self.assertEqual(self.emptied_count, 1)
        self.assertEqual(self.session.state.uploaded_files, [])
        self._assert_active_image_reset()

    def test_clear_files_emits_and_resets(self):
        self.session.clear_files()
        self.assertEqual(self.emptied_count, 1)
        self._assert_active_image_reset()

    def test_remove_selected_last_files_emits_and_resets(self):
        self.session.remove_selected_files()
        self.assertEqual(self.emptied_count, 1)
        self._assert_active_image_reset()

    def test_remove_with_remaining_files_does_not_emit(self):
        self.session.state.uploaded_files.append({"name": "file2.dng", "path": "path2", "hash": "hash2"})
        self.session.remove_current_file()
        self.assertEqual(self.emptied_count, 0)
        self.assertEqual(len(self.session.state.uploaded_files), 1)
        self.assertEqual(self.session.state.selected_file_idx, 0)


class TestTriageMarks(unittest.TestCase):
    def setUp(self):
        self.mock_repo = MagicMock(spec=StorageRepository)
        self.mock_repo.load_file_settings.return_value = None
        self.mock_repo.load_file_settings_by_path.return_value = None
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: {} if key == "last_export_config" else default
        self.mock_repo.get_max_history_index.return_value = 0
        self.mock_repo.load_file_marks.return_value = {}
        self.session = DesktopSessionManager(self.mock_repo)
        self.session.state.uploaded_files = [
            {"name": "f1.dng", "path": "p1", "hash": "hash1"},
            {"name": "f2.dng", "path": "p2", "hash": "hash2"},
            {"name": "f3.dng", "path": "p3", "hash": "hash3"},
        ]
        self.session.state.selected_file_idx = 0
        self.session.asset_model.refresh()

    def test_reject_toggles_and_persists(self):
        self.session.toggle_mark("excluded")
        self.assertTrue(self.session.state.uploaded_files[0]["excluded"])
        self.mock_repo.save_file_mark.assert_called_with("hash1", "excluded")

        self.session.toggle_mark("excluded")
        self.assertFalse(self.session.state.uploaded_files[0]["excluded"])
        self.mock_repo.save_file_mark.assert_called_with("hash1", None)

    def test_marks_are_mutually_exclusive(self):
        self.session.toggle_mark("keeper")
        self.session.toggle_mark("excluded")
        f = self.session.state.uploaded_files[0]
        self.assertTrue(f["excluded"])
        self.assertFalse(f["keeper"])

    def test_multi_selection_toggles_as_block(self):
        self.session.state.uploaded_files[0]["keeper"] = True
        self.session.state.selected_indices = [0, 1]

        # Mixed block: mark all (not clear the one already marked)
        self.session.toggle_mark("keeper")
        self.assertTrue(all(self.session.state.uploaded_files[i].get("keeper") for i in (0, 1)))

        # Uniform block: clear all
        self.session.toggle_mark("keeper")
        self.assertFalse(any(self.session.state.uploaded_files[i].get("keeper") for i in (0, 1)))

    def test_invalid_mark_is_noop(self):
        self.session.toggle_mark("starred")
        self.mock_repo.save_file_mark.assert_not_called()

    def test_sheet_filter_unrejected_hides_rejected(self):
        self.session.state.uploaded_files[1]["excluded"] = True
        self.session.asset_model.set_sheet_filter("unrejected")
        self.assertEqual(self.session.asset_model.visible_actual_indices(), {0, 2})

    def test_sheet_filter_keepers_only(self):
        self.session.state.uploaded_files[2]["keeper"] = True
        self.session.asset_model.set_sheet_filter("keepers")
        self.assertEqual(self.session.asset_model.visible_actual_indices(), {2})

    def test_sheet_filter_all_shows_rejected(self):
        self.session.state.uploaded_files[1]["excluded"] = True
        self.session.asset_model.set_sheet_filter("all")
        self.assertEqual(self.session.asset_model.visible_actual_indices(), {0, 1, 2})

    def test_add_files_restores_marks_from_repo(self):
        self.mock_repo.load_file_marks.return_value = {"hash9": "keeper", "hash2": "excluded"}
        self.session.add_files([], validated_info=[{"name": "f9.dng", "path": "p9", "hash": "hash9"}])
        files = self.session.state.uploaded_files
        self.assertTrue(files[3]["keeper"])
        self.assertTrue(files[1]["excluded"])
        self.assertFalse(files[0]["keeper"] or files[0]["excluded"])


class TestRollActionRecoveryRoundTrip(unittest.TestCase):
    """End-to-end with a real repository: a roll-wide sync is recoverable on each
    target frame with plain undo after switching to it."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.repo = StorageRepository(f"{self.tmp.name}/edits.db", f"{self.tmp.name}/settings.db")
        self.repo.initialize()
        self.session = DesktopSessionManager(self.repo)
        self.session.state.uploaded_files = [
            {"name": "f1.dng", "path": f"{self.tmp.name}/f1.dng", "hash": "hash1"},
            {"name": "f2.dng", "path": f"{self.tmp.name}/f2.dng", "hash": "hash2"},
        ]
        self.session.asset_model.refresh()

    def tearDown(self):
        self.tmp.cleanup()

    def test_hidden_masks_survive_restart(self):
        from negpy.features.local.models import LocalAdjustmentsConfig, PolygonMask

        # hash1 has two masks on disk; index 1 is hidden. The property clamps against the
        # hydrated mask list, so persistence only "counts" if that config reloads too.
        verts = ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9))
        two_masks = (PolygonMask(vertices=verts), PolygonMask(vertices=verts, strength=-0.3))
        cfg = replace(WorkspaceConfig(), local=LocalAdjustmentsConfig(masks=two_masks))
        self.repo.save_file_settings("hash1", cfg, file_path=self.session.state.uploaded_files[0]["path"])

        self.session.state.local_hidden_masks_by_hash = {"hash1": {1}, "hash2": set()}
        self.session.persist_hidden_masks()

        # A fresh manager on the same repo simulates an app restart.
        restarted = DesktopSessionManager(self.repo)
        self.assertEqual(restarted.state.local_hidden_masks_by_hash, {"hash1": {1}})

        restarted.state.uploaded_files = self.session.state.uploaded_files
        restarted.select_file(0)
        self.assertEqual(restarted.state.local_hidden_masks, {1})
        restarted.select_file(1)
        self.assertEqual(restarted.state.local_hidden_masks, set())

    def test_sync_then_undo_restores_target(self):
        target_before = replace(WorkspaceConfig(), exposure=replace(WorkspaceConfig().exposure, density=2.0))
        self.repo.save_file_settings("hash2", target_before, file_path=self.session.state.uploaded_files[1]["path"])

        self.session.select_file(0)
        source = replace(self.session.state.config, exposure=replace(self.session.state.config.exposure, density=1.5))
        self.session.update_config(source, persist=True)

        count = self.session.sync_selected_settings(frozenset({"exposure"}), scope="roll")
        self.assertEqual(count, 1)
        self.assertEqual(self.repo.load_file_settings("hash2").exposure.density, 1.5)

        self.session.select_file(1)
        self.assertEqual(self.session.state.config.exposure.density, 1.5)

        self.session.undo()
        self.assertEqual(self.session.state.config.exposure.density, 2.0)

        self.session.redo()
        self.assertEqual(self.session.state.config.exposure.density, 1.5)


if __name__ == "__main__":
    unittest.main()
