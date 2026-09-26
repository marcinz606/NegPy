import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from dataclasses import replace

from negpy.desktop.session import AppState, AssetListModel, DesktopSessionManager
from negpy.desktop.settings_catalog import all_rows
from negpy.domain.models import WorkspaceConfig, GeometryConfig, RetouchConfig, ProcessConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.config import APP_CONFIG, DEFAULT_WORKSPACE_CONFIG
from negpy.features.process.models import ProcessMode

_ROWS = {r.label: r for r in all_rows()}


def _row(label: str):
    return _ROWS[label]


class TestDesktopSessionSync(unittest.TestCase):
    def setUp(self):
        self.mock_repo = MagicMock(spec=StorageRepository)
        self.mock_repo.load_file_settings.return_value = None
        self.mock_repo.load_file_settings_by_path.return_value = None
        self.mock_repo.load_file_settings_many.return_value = {}

        # Mock global settings with correct types
        def mock_get_global(key, default=None):
            if key == "last_export_config":
                return {}
            if key == "process_mode":
                return ProcessMode.C41
            return default

        self.mock_repo.get_global_setting.side_effect = mock_get_global
        self.mock_repo.get_max_history_index.return_value = 0
        self.session = DesktopSessionManager(self.mock_repo)
        # Construction itself writes (startup migrations, e.g.); tests below assert
        # what one specific action writes, not the lifetime total since setUp.
        self.mock_repo.save_global_setting.reset_mock()

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
            process=replace(defaults.process, process_mode=ProcessMode.E6),
            geometry=replace(defaults.geometry, autocrop_ratio="4:3"),
        )
        sticky = {
            "sticky_config": {
                "jpeg_quality": 73,
                "protect_original_metadata": True,
                # Workflow defaults must not overwrite an edited/saved asset.
                "process_mode": ProcessMode.C41,
                "autocrop_ratio": "1:1",
            },
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        asset = {"name": "saved.dng", "path": "/roll/saved.dng", "hash": "saved-hash"}
        active_before = self.session.state.config

        with patch("negpy.desktop.session.load_or_promote", return_value=saved) as hydrate:
            config = self.session.config_for_asset(asset)

        hydrate.assert_called_once_with(self.mock_repo, "saved-hash", "/roll/saved.dng", half=0, composite=False, forked=False)
        self.assertEqual(config.exposure.density, 1.7)
        self.assertEqual(config.process.process_mode, ProcessMode.E6)
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
            "sticky_config": {
                "process_mode": ProcessMode.E6,
                "autocrop_ratio": "1:1",
                "autocrop_offset": 7,
                "auto_exposure": True,
            },
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
        self.assertEqual(config.process.process_mode, ProcessMode.E6)
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

    def test_config_for_asset_applies_the_active_rolls_defaults(self):
        rolls_store = {"r1": {"kind": "virtual", "name": "Portra", "defaults": {"linear_raw": True}}}
        globals_ = {"rolls_by_id": rolls_store}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: globals_.get(key, default)
        self.session.state.active_roll_id = "r1"
        asset = {"name": "a.dng", "path": "/roll/a.dng", "hash": "a-hash"}

        with patch("negpy.desktop.session.load_or_promote", return_value=None):
            config = self.session.config_for_asset(asset)

        self.assertTrue(config.process.linear_raw)

    def test_config_for_asset_applies_roll_defaults_outside_the_process_config(self):
        """Auto Crop, Lens Correction and Flat Field are roll cards on other config
        sections; a roll's own flat-field profile outranks the rig-global one."""
        rolls_store = {
            "r1": {
                "kind": "virtual",
                "name": "Portra",
                "defaults": {"autocrop_rebate_trim": 0.5, "distortion_k1": 0.02, "profile_id": "roll-ff", "apply": True},
            }
        }
        globals_ = {"rolls_by_id": rolls_store, "flatfield_active_profile": ""}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: globals_.get(key, default)
        self.session.state.active_roll_id = "r1"
        asset = {"name": "a.dng", "path": "/roll/a.dng", "hash": "a-hash"}

        with patch("negpy.desktop.session.load_or_promote", return_value=None):
            config = self.session.config_for_asset(asset)

        self.assertEqual(config.geometry.autocrop_rebate_trim, 0.5)
        self.assertEqual(config.geometry.distortion_k1, 0.02)
        self.assertEqual(config.flatfield.profile_id, "roll-ff")
        self.assertTrue(config.flatfield.apply)

    def test_config_for_asset_keeps_a_frames_own_auto_crop_when_that_card_is_locked(self):
        rolls_store = {
            "r1": {
                "kind": "virtual",
                "name": "Portra",
                "defaults": {"autocrop_rebate_trim": 0.5, "distortion_k1": 0.02},
                "frame_overrides": {"a-hash": ["autocrop"]},
            }
        }
        globals_ = {"rolls_by_id": rolls_store}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: globals_.get(key, default)
        self.session.state.active_roll_id = "r1"
        asset = {"name": "a.dng", "path": "/roll/a.dng", "hash": "a-hash"}

        with patch("negpy.desktop.session.load_or_promote", return_value=None):
            config = self.session.config_for_asset(asset)

        self.assertEqual(config.geometry.autocrop_rebate_trim, 1.0)
        self.assertEqual(config.geometry.distortion_k1, 0.02)

    def test_config_for_asset_ignores_roll_defaults_when_the_file_belongs_to_no_roll(self):
        rolls_store = {"r1": {"kind": "virtual", "name": "Portra", "defaults": {"linear_raw": True}}}
        globals_ = {"rolls_by_id": rolls_store}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: globals_.get(key, default)
        self.session.state.active_roll_id = None
        asset = {"name": "a.dng", "path": "/roll/a.dng", "hash": "a-hash"}

        with patch("negpy.desktop.session.load_or_promote", return_value=None):
            config = self.session.config_for_asset(asset)

        self.assertFalse(config.process.linear_raw)

    def test_config_for_asset_falls_back_to_the_files_own_roll_with_no_active_roll(self):
        """A library-wide search's mixed results (or a restored session with no single
        shared roll) leave nothing active for the session, but each file still belongs
        to a real roll on disk -- that roll's own settings, not the sticky "last used
        anywhere" guess, are what actually describe this specific frame."""
        rolls_store = {"r1": {"kind": "folder", "name": "Portra", "folder_path": "/roll", "defaults": {"linear_raw": True}}}
        globals_ = {"rolls_by_id": rolls_store}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: globals_.get(key, default)
        self.session.state.active_roll_id = None
        asset = {"name": "a.dng", "path": "/roll/a.dng", "hash": "a-hash"}

        with patch("negpy.desktop.session.load_or_promote", return_value=None):
            config = self.session.config_for_asset(asset)

        self.assertTrue(config.process.linear_raw)

    def test_config_for_asset_prefers_a_folder_roll_over_a_virtual_one_for_the_same_path(self):
        """A folder roll is the file's actual physical home; a virtual roll is a curated
        collection that may not carry its capture facts at all."""
        rolls_store = {
            "folder1": {"kind": "folder", "name": "Portra", "folder_path": "/roll", "defaults": {"linear_raw": True}},
            "virtual1": {"kind": "virtual", "name": "Favorites", "member_paths": ["/roll/a.dng"], "defaults": {"linear_raw": False}},
        }
        globals_ = {"rolls_by_id": rolls_store}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: globals_.get(key, default)
        self.session.state.active_roll_id = None
        asset = {"name": "a.dng", "path": "/roll/a.dng", "hash": "a-hash"}

        with patch("negpy.desktop.session.load_or_promote", return_value=None):
            config = self.session.config_for_asset(asset)

        self.assertTrue(config.process.linear_raw)

    def test_config_for_asset_respects_a_frames_locked_card(self):
        rolls_store = {
            "r1": {
                "kind": "virtual",
                "name": "Portra",
                "defaults": {"linear_raw": True},
                "frame_overrides": {"a-hash": ["sensor"]},
            }
        }
        globals_ = {"rolls_by_id": rolls_store}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: globals_.get(key, default)
        self.session.state.active_roll_id = "r1"
        asset = {"name": "a.dng", "path": "/roll/a.dng", "hash": "a-hash"}
        saved = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, linear_raw=False))

        with patch("negpy.desktop.session.load_or_promote", return_value=saved):
            config = self.session.config_for_asset(asset)

        self.assertFalse(config.process.linear_raw)

    def test_config_for_asset_reads_roll_state_by_the_unforked_hash(self):
        """A lock (or a roll default's frame_overrides entry) is about this physical
        frame, not its current edit identity -- a forked asset's suffixed hash must
        still resolve against the plain hash's lock."""
        from negpy.services.assets.rolls import roll_edit_hash

        rolls_store = {
            "r1": {
                "kind": "virtual",
                "name": "Portra",
                "defaults": {"linear_raw": True},
                "frame_overrides": {"a-hash": ["sensor"]},
            }
        }
        globals_ = {"rolls_by_id": rolls_store}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: globals_.get(key, default)
        self.session.state.active_roll_id = "r1"
        forked_hash = roll_edit_hash("a-hash", "r1")
        asset = {"name": "a.dng", "path": "/roll/a.dng", "hash": forked_hash}
        saved = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, linear_raw=False))

        with patch("negpy.desktop.session.load_or_promote", return_value=saved):
            config = self.session.config_for_asset(asset)

        # sensor (Calibration) is locked for a-hash, so linear_raw keeps its own value
        # even though the asset is currently showing its forked edit identity.
        self.assertFalse(config.process.linear_raw)

    def test_set_autodetect_enabled_persists(self):
        self.assertFalse(self.session.state.autodetect_enabled)
        self.session.set_autodetect_enabled(True)
        self.assertTrue(self.session.state.autodetect_enabled)
        self.mock_repo.save_global_setting.assert_called_with("autodetect_enabled", True)

    def test_set_autodetect_enabled_noop_when_unchanged(self):
        self.session.set_autodetect_enabled(False)
        self.mock_repo.save_global_setting.assert_not_called()

    def test_set_sticky_zoom_persists(self):
        self.assertFalse(self.session.state.sticky_zoom)
        self.session.set_sticky_zoom(True)
        self.assertTrue(self.session.state.sticky_zoom)
        self.mock_repo.save_global_setting.assert_called_with("sticky_zoom", True)

    def test_set_sticky_zoom_noop_when_unchanged(self):
        self.session.set_sticky_zoom(False)
        self.mock_repo.save_global_setting.assert_not_called()

    def test_set_invert_zoom_scroll_persists(self):
        self.assertFalse(self.session.state.invert_zoom_scroll)
        self.session.set_invert_zoom_scroll(True)
        self.assertTrue(self.session.state.invert_zoom_scroll)
        self.mock_repo.save_global_setting.assert_called_with("invert_zoom_scroll", True)

    def test_set_invert_zoom_scroll_noop_when_unchanged(self):
        self.session.set_invert_zoom_scroll(False)
        self.mock_repo.save_global_setting.assert_not_called()

    def test_persist_writes_sticky_settings_in_one_batch(self):
        self.session.select_file(0)
        self.mock_repo.save_global_settings.reset_mock()

        cfg = replace(self.session.state.config, exposure=replace(self.session.state.config.exposure, density=1.5))
        self.session.update_config(cfg, persist=True)

        self.mock_repo.save_global_settings.assert_called_once()
        saved = self.mock_repo.save_global_settings.call_args.args[0]
        snapshot = saved["sticky_config"]
        self.assertEqual(snapshot["density"], 1.5)
        for field in ("process_mode", "jpeg_quality", "dust_remove", "paper_black", "protect_original_metadata", "cast_removal_strength"):
            self.assertIn(field, snapshot)
        # Carried on their own keys: no catalog row reaches them.
        self.assertIn("last_export_config", saved)
        self.assertIn("last_narrowband_scan", saved)

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
        sticky = {"sticky_config": {"protect_original_metadata": True}}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertTrue(config.metadata.protect_original_metadata)

    def test_protect_original_metadata_applied_to_saved_files(self):
        sticky = {"sticky_config": {"protect_original_metadata": True}}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        base = WorkspaceConfig(metadata=replace(WorkspaceConfig().metadata, protect_original_metadata=False))
        config = self.session._apply_sticky_settings(base, only_global=True)
        self.assertTrue(config.metadata.protect_original_metadata)

    def test_description_fields_sticky_applied_when_unset(self):
        sticky = {
            "last_export_config": {},
            "last_description_fields": ["camera", "film", "developer", "scanning"],
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        base = WorkspaceConfig()  # description_fields is None
        config = self.session._apply_sticky_settings(base, only_global=True)
        self.assertEqual(config.metadata.description_fields, ("camera", "film", "developer", "scanning"))

    def test_description_fields_per_frame_not_overwritten_by_sticky(self):
        sticky = {
            "last_export_config": {},
            "last_description_fields": ["camera", "film", "developer", "scanning"],
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        base = WorkspaceConfig(
            metadata=replace(WorkspaceConfig().metadata, description_fields=("camera", "iso")),
        )
        config = self.session._apply_sticky_settings(base, only_global=True)
        self.assertEqual(config.metadata.description_fields, ("camera", "iso"))

    def test_persist_sticky_settings_does_not_write_description_fields(self):
        """Any metadata save must not clobber last Description… confirm."""
        config = WorkspaceConfig(
            metadata=replace(WorkspaceConfig().metadata, description_fields=("camera", "developer")),
        )
        self.session._persist_sticky_settings(config)
        saved = self.mock_repo.save_global_settings.call_args.args[0]
        self.assertNotIn("last_description_fields", saved)

    def test_processing_toggles_carry_to_new_files(self):
        # Globally remembered toggles must be applied to a fresh (sidecar-less) file.
        sticky = {
            "sticky_config": {
                "auto_exposure": True,
                "auto_normalize_contrast": True,
                "paper_dmin": True,
                "paper_black": True,
                "paper_profile": "ilford_mg_rc",
                "cast_removal_strength": 0.8,
            },
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
        sticky = {"sticky_config": {"cast_removal_strength": 0.0}}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(config.exposure.cast_removal_strength, 0.0)

    def test_legacy_rig_k1_seeds_geometry_once(self):
        """Distortion left the flat-field profile; a rig that still carries one must not
        lose it, and must not override a frame's own value."""
        prof = SimpleNamespace(id="rig-a", k1=-0.05)
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: (
            "rig-a" if key == "flatfield_active_profile" else default
        )
        with patch("negpy.desktop.session.FlatFieldProfiles.get", return_value=prof):
            seeded = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
            own = WorkspaceConfig(geometry=replace(WorkspaceConfig().geometry, distortion_k1=0.012))
            kept = self.session._apply_sticky_settings(own, only_global=False)

        self.assertEqual(seeded.geometry.distortion_k1, -0.05)
        self.assertEqual(kept.geometry.distortion_k1, 0.012)

    def test_paper_black_carries_to_new_files(self):
        """Sticky must carry an explicit value over the file's base."""
        sticky = {"sticky_config": {"paper_black": False}}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        base = WorkspaceConfig(exposure=replace(WorkspaceConfig().exposure, paper_black=True))
        config = self.session._apply_sticky_settings(base, only_global=False)
        self.assertFalse(config.exposure.paper_black)

    def test_legacy_true_black_sticky_migrates_inverted(self):
        """A pre-rename sticky (last_true_black) maps to paper_black inverted."""
        from negpy.desktop.sticky import migrate_legacy

        legacy = {"last_true_black": False}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: legacy.get(key, default)
        migrate_legacy(self.mock_repo)
        key, snapshot = self.mock_repo.save_global_setting.call_args.args
        self.assertEqual(key, "sticky_config")
        self.assertTrue(snapshot["paper_black"])

    def test_opt_in_row_carries_once_chosen(self):
        """A look setting is not sticky by default, but becomes so when ticked."""
        sticky = {"sticky_config": {"density": 2.2}}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(config.exposure.density, WorkspaceConfig().exposure.density)

        sticky["sticky_rows"] = ["exposure.density"]
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(config.exposure.density, 2.2)

    def test_unticked_default_row_stops_carrying(self):
        sticky = {"sticky_config": {"process_mode": ProcessMode.E6}, "sticky_rows": []}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(config.process.process_mode, WorkspaceConfig().process.process_mode)

    def test_only_global_tier_ignores_look_rows(self):
        """A saved edit keeps its own look even when the user made a look row sticky."""
        sticky = {
            "sticky_config": {"density": 2.2, "jpeg_quality": 73},
            "sticky_rows": ["exposure.density", "export.jpeg_quality"],
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=True)
        self.assertEqual(config.exposure.density, WorkspaceConfig().exposure.density)
        self.assertEqual(config.export.jpeg_quality, 73)

    def test_saved_file_keeps_cached_bounds(self):
        """The global tier must not touch a bounds-input field, or resolve_bounds re-runs."""
        sticky = {
            "sticky_config": {"process_mode": ProcessMode.E6, "jpeg_quality": 73},
            "sticky_rows": ["process.process_mode", "export.jpeg_quality"],
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        base = WorkspaceConfig(
            process=replace(
                WorkspaceConfig().process,
                locked_floors=(0.1, 0.2, 0.3),
                locked_ceils=(1.1, 1.2, 1.3),
            )
        )
        config = self.session._apply_sticky_settings(base, only_global=True)
        self.assertTrue(config.process.is_locked_initialized)
        self.assertEqual(config.process.locked_floors, (0.1, 0.2, 0.3))

    def test_export_remainder_carries_without_a_catalog_row(self):
        """The output folder has no row, so it rides last_export_config unconditionally."""
        sticky = {"last_export_config": {"export_path": "/out", "jpeg_quality": 99}, "sticky_rows": []}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(config.export.export_path, "/out")
        # jpeg_quality has a row, and no row was chosen, so it must not ride along.
        self.assertEqual(config.export.jpeg_quality, WorkspaceConfig().export.jpeg_quality)

    def test_roll_average_not_seeded_onto_fresh_files(self):
        # A roll baseline must not leak onto a fresh (sidecar-less) file.
        sticky = {
            "last_export_config": {},
            "last_use_luma_average": True,
            "last_use_color_average": True,
            "last_locked_floors": [0.1, 0.2, 0.3],
            "last_locked_ceils": [1.1, 1.2, 1.3],
            "last_roll_name": "roll-A",
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertFalse(config.process.use_luma_average)
        self.assertFalse(config.process.use_color_average)
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
                use_color_average=True,
                locked_floors=(0.1, 0.2, 0.3),
                locked_ceils=(1.1, 1.2, 1.3),
                roll_name="roll-A",
            ),
        )
        self.session.update_config(loaded, persist=True)
        saved = self.mock_repo.save_global_settings.call_args.args[0]
        for key in ("last_use_luma_average", "last_use_color_average", "last_locked_floors", "last_locked_ceils", "last_roll_name"):
            self.assertNotIn(key, saved)

    def test_saved_file_keeps_its_own_roll_baseline(self):
        # A file whose own config carries the baseline (only_global=True) is untouched.
        base = WorkspaceConfig(
            process=replace(
                WorkspaceConfig().process,
                use_luma_average=True,
                use_color_average=True,
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

    def test_sticky_settings_enabled_defaults_true(self):
        self.assertTrue(self.session.state.sticky_settings_enabled)

    def test_set_sticky_settings_enabled_persists(self):
        self.session.set_sticky_settings_enabled(False)
        self.assertFalse(self.session.state.sticky_settings_enabled)
        self.mock_repo.save_global_setting.assert_called_with("sticky_settings_enabled", False)

    def test_set_sticky_settings_enabled_noop_when_unchanged(self):
        self.session.set_sticky_settings_enabled(True)
        self.mock_repo.save_global_setting.assert_not_called()

    def test_master_switch_off_gates_the_whole_row_overlay_on_a_new_file(self):
        """Off, every catalog row - look and rig alike - stays at its WorkspaceConfig() default."""
        sticky = {
            "sticky_config": {
                "saturation": 1.8,
                "cast_removal_strength": 0.9,
                "process_mode": ProcessMode.E6,
                "flip_horizontal": True,
                "jpeg_quality": 55,
            },
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        self.session.state.sticky_settings_enabled = False
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        default = WorkspaceConfig()
        self.assertEqual(config.lab.saturation, default.lab.saturation)
        self.assertEqual(config.exposure.cast_removal_strength, default.exposure.cast_removal_strength)
        self.assertEqual(config.process.process_mode, default.process.process_mode)
        self.assertFalse(config.geometry.flip_horizontal)
        self.assertEqual(config.export.jpeg_quality, default.export.jpeg_quality)

    def test_master_switch_on_still_applies_the_row_overlay(self):
        """Unchanged default behaviour: enabled carries the look onto a fresh file."""
        sticky = {"sticky_config": {"saturation": 1.8}}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        self.assertTrue(self.session.state.sticky_settings_enabled)
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(config.lab.saturation, 1.8)

    def test_master_switch_off_leaves_hard_coded_carries_untouched(self):
        """The rig facts and workspace state are not picker rows, so the switch never gates them."""
        prof = SimpleNamespace(id="rig-a", k1=-0.05)
        sticky = {
            "last_export_config": {"export_path": "/out"},
            "last_linear_raw": True,
            "flatfield_active_profile": "rig-a",
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        self.session.state.sticky_settings_enabled = False
        with patch("negpy.desktop.session.FlatFieldProfiles.get", return_value=prof):
            config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(config.export.export_path, "/out")
        self.assertTrue(config.process.linear_raw)
        self.assertEqual(config.flatfield.profile_id, "rig-a")
        self.assertTrue(config.flatfield.apply)

    def test_master_switch_off_does_not_affect_only_global_branch(self):
        """only_global=True (an already-edited file) ignores the switch entirely."""
        sticky = {
            "sticky_config": {"density": 2.2, "jpeg_quality": 73},
            "sticky_rows": ["exposure.density", "export.jpeg_quality"],
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        self.session.state.sticky_settings_enabled = False
        config = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=True)
        self.assertEqual(config.exposure.density, WorkspaceConfig().exposure.density)
        self.assertEqual(config.export.jpeg_quality, 73)

    def test_master_switch_off_does_not_clear_row_picks(self):
        """Turning the switch off is non-destructive: the per-row picks survive untouched."""
        from negpy.desktop.sticky import STICKY_CONFIG_KEY, STICKY_ROWS_KEY

        sticky = {"sticky_config": {"density": 2.2}, "sticky_rows": ["exposure.density"]}
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)

        self.session.set_sticky_settings_enabled(False)
        called_keys = {c.args[0] for c in self.mock_repo.save_global_setting.call_args_list}
        self.assertNotIn(STICKY_ROWS_KEY, called_keys)
        self.assertNotIn(STICKY_CONFIG_KEY, called_keys)

        off = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(off.exposure.density, WorkspaceConfig().exposure.density)

        self.session.state.sticky_settings_enabled = True
        on = self.session._apply_sticky_settings(WorkspaceConfig(), only_global=False)
        self.assertEqual(on.exposure.density, 2.2)

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
            geometry=GeometryConfig(rotation=1, fine_rotation=5.5, crop_rect=(0, 0, 1, 1)),
            retouch=RetouchConfig(dust_remove=True, manual_dust_spots=[(0.1, 0.1, 5)]),
            process=ProcessConfig(process_mode=ProcessMode.E6),
        )
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = source_config

        target_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=0.0),
            geometry=GeometryConfig(rotation=0, fine_rotation=0.0, crop_rect=None),
            retouch=RetouchConfig(dust_remove=False, manual_dust_spots=[]),
            process=ProcessConfig(process_mode=ProcessMode.C41),
        )
        self.mock_repo.load_file_settings.return_value = target_config

        self.session.update_selection([0, 1])
        self.session.sync_selected_settings([_row("Print Density"), _row("Film Mode"), _row("Optical Removal")])

        args, _ = self.mock_repo.save_file_settings.call_args
        self.assertEqual(args[0], "hash2")
        saved_config = args[1]

        self.assertEqual(saved_config.exposure.density, 1.5)
        self.assertEqual(saved_config.process.process_mode, ProcessMode.E6)

        # Geometry not selected → entirely preserved from target
        self.assertEqual(saved_config.geometry.rotation, 0)
        self.assertEqual(saved_config.geometry.fine_rotation, 0.0)
        self.assertIsNone(saved_config.geometry.crop_rect)
        # Per-file retouch fields preserved from target even though Dust Removal was synced
        self.assertEqual(saved_config.retouch.manual_dust_spots, [])
        self.assertTrue(saved_config.retouch.dust_remove)

    def _sync_bounds_from(self, process):
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.current_file_path = "/tmp/f003.tif"
        self.session.state.config = WorkspaceConfig(process=process)
        self.mock_repo.load_file_settings.return_value = WorkspaceConfig()
        self.session.update_selection([0, 1])
        self.session.sync_selected_settings([], bounds_flags=(True, True))
        return self.mock_repo.save_file_settings.call_args.args[1].process

    def test_sync_bounds_from_a_metered_frame_names_that_frame(self):
        synced = self._sync_bounds_from(ProcessConfig(local_floors=(0.1, 0.1, 0.1), local_ceils=(0.9, 0.9, 0.9)))
        self.assertEqual(synced.baseline_source, "frame:f003.tif")

    def test_sync_bounds_from_a_frame_on_a_baseline_passes_its_source_on(self):
        synced = self._sync_bounds_from(
            ProcessConfig(use_luma_average=True, locked_floors=(0.2, 0.2, 0.2), locked_ceils=(0.8, 0.8, 0.8), baseline_source="scene:s1")
        )
        self.assertEqual(synced.baseline_source, "scene:s1")

    def test_sync_selected_settings_edits_with_geometry(self):
        source_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=1.5),
            geometry=GeometryConfig(rotation=1, fine_rotation=5.5, crop_rect=(0.1, 0.1, 0.9, 0.9)),
            retouch=RetouchConfig(dust_remove=True, manual_dust_spots=[(0.1, 0.1, 5)]),
            process=ProcessConfig(process_mode=ProcessMode.E6),
        )
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = source_config

        target_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=0.0),
            geometry=GeometryConfig(rotation=0, fine_rotation=0.0, crop_rect=None),
            retouch=RetouchConfig(dust_remove=False, manual_dust_spots=[(0.5, 0.5, 3)]),
            process=ProcessConfig(process_mode=ProcessMode.C41),
        )
        self.mock_repo.load_file_settings.return_value = target_config

        self.session.update_selection([0, 1])
        self.session.sync_selected_settings([_row("Print Density"), _row("Fine Rotation"), _row("Rotation"), _row("Crop")])

        args, _ = self.mock_repo.save_file_settings.call_args
        saved_config = args[1]

        # Crop and fine_rotation should now propagate from source
        self.assertEqual(saved_config.geometry.fine_rotation, 5.5)
        self.assertEqual(saved_config.geometry.crop_rect, (0.1, 0.1, 0.9, 0.9))
        self.assertEqual(saved_config.geometry.rotation, 1)
        # Edits still synced
        self.assertEqual(saved_config.exposure.density, 1.5)
        # Dust spots still per-target
        self.assertEqual(saved_config.retouch.manual_dust_spots, [(0.5, 0.5, 3)])

    def test_sync_selected_settings_geometry_only(self):
        source_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=1.5),
            geometry=GeometryConfig(rotation=2, fine_rotation=3.0, crop_rect=(0.0, 0.0, 0.5, 0.5)),
        )
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = source_config

        target_config = WorkspaceConfig(
            exposure=replace(WorkspaceConfig().exposure, density=0.7),
            geometry=GeometryConfig(rotation=0, fine_rotation=0.0, crop_rect=None),
        )
        self.mock_repo.load_file_settings.return_value = target_config

        self.session.update_selection([0, 1])
        self.session.sync_selected_settings([_row("Rotation"), _row("Fine Rotation"), _row("Crop")])

        args, _ = self.mock_repo.save_file_settings.call_args
        saved_config = args[1]

        # Geometry comes from source
        self.assertEqual(saved_config.geometry.rotation, 2)
        self.assertEqual(saved_config.geometry.fine_rotation, 3.0)
        self.assertEqual(saved_config.geometry.crop_rect, (0.0, 0.0, 0.5, 0.5))
        # Other config preserved from target
        self.assertEqual(saved_config.exposure.density, 0.7)

    def test_sync_selected_settings_resets_crop_offset_to_default(self):
        # #656: pushing the source's default value (offset 0) must clear the target's.
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = WorkspaceConfig(geometry=GeometryConfig(autocrop_offset=0))
        self.mock_repo.load_file_settings.return_value = WorkspaceConfig(geometry=GeometryConfig(autocrop_offset=3))

        self.session.update_selection([0, 1])
        self.session.sync_selected_settings([_row("Crop Offset")])

        args, _ = self.mock_repo.save_file_settings.call_args
        self.assertEqual(args[1].geometry.autocrop_offset, 0)

    def test_sync_fresh_target_keeps_sticky_workflow_not_bare_defaults(self):
        """P0-5 Variant B: a target frame with no saved edits must fall back to the
        sticky-aware hydrated config, not bare WorkspaceConfig(), so syncing one field
        doesn't silently reset its scan/process-mode to dataclass defaults."""
        sticky = {
            "sticky_config": {"process_mode": ProcessMode.E6},
            "last_narrowband_scan": True,
        }
        self.mock_repo.get_global_setting.side_effect = lambda key, default=None: sticky.get(key, default)
        self.mock_repo.load_file_settings.return_value = None  # fresh target, no DB row

        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = replace(WorkspaceConfig(), exposure=replace(WorkspaceConfig().exposure, density=1.5))

        self.session.update_selection([0, 1])
        with patch("negpy.desktop.session.load_or_promote", return_value=None):
            self.session.sync_selected_settings([_row("Print Density")])

        args, _ = self.mock_repo.save_file_settings.call_args
        saved = args[1]
        self.assertEqual(saved.exposure.density, 1.5)  # the one synced field
        # Sticky workflow settings survive because the base was config_for_asset, not defaults.
        self.assertTrue(saved.process.narrowband_scan)
        self.assertEqual(saved.process.process_mode, ProcessMode.E6)

    def test_sync_selected_settings_empty_is_noop(self):
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.update_selection([0, 1])
        self.session.sync_selected_settings([])
        self.mock_repo.save_file_settings.assert_not_called()

    def test_sync_selected_settings_emits_frames_edited_offscreen(self):
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = WorkspaceConfig(exposure=replace(WorkspaceConfig().exposure, density=1.5))
        self.mock_repo.load_file_settings.return_value = WorkspaceConfig()
        offscreen = []
        self.session.frames_edited_offscreen.connect(offscreen.append)

        self.session.update_selection([0, 1])
        self.session.sync_selected_settings([_row("Print Density")])

        self.assertEqual(offscreen, [["hash2"]])

    def test_sync_selected_settings_empty_emits_nothing(self):
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        offscreen = []
        self.session.frames_edited_offscreen.connect(offscreen.append)

        self.session.update_selection([0, 1])
        self.session.sync_selected_settings([])

        self.assertEqual(offscreen, [])

    def test_apply_pasted_fields_applies_subset_and_renders(self):
        self.session.state.current_file_hash = "hash1"
        self.session.state.config = replace(WorkspaceConfig(), lab=replace(WorkspaceConfig().lab, saturation=1.9))
        self.session.state.clipboard = replace(
            WorkspaceConfig(),
            exposure=replace(WorkspaceConfig().exposure, density=2.2),
            lab=replace(WorkspaceConfig().lab, saturation=0.3),
        )
        rendered = []
        self.session.state_changed.connect(lambda: rendered.append(True))

        self.session.apply_pasted_fields([_row("Print Density")])

        self.assertEqual(self.session.state.config.exposure.density, 2.2)  # pasted
        self.assertEqual(self.session.state.config.lab.saturation, 1.9)  # not selected → kept
        self.assertTrue(rendered)

    def test_apply_pasted_fields_noop_when_clipboard_empty(self):
        self.session.state.current_file_hash = "hash1"
        self.session.state.clipboard = None
        before = self.session.state.config
        self.session.apply_pasted_fields([_row("Print Density")])
        self.assertIs(self.session.state.config, before)

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

        count = self.session.sync_selected_settings([_row("Print Density")], scope="roll")

        saved = {c.args[0] for c in self.mock_repo.save_file_settings.call_args_list}
        self.assertEqual(count, 1)
        self.assertEqual(saved, {"hash2"})  # source + filtered-out c.jpg excluded

    def test_sync_roll_scope_no_filter_covers_all(self):
        self._seed_roll()
        self.session.asset_model.refresh()  # no filter → every frame visible

        count = self.session.sync_selected_settings([_row("Print Density")], scope="roll")

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

    def test_history_changed_sees_new_config(self):
        """The HUD reads state.config from this signal, so the step must already be live."""
        self.session.select_file(0)
        seen = []
        self.session.history_changed.connect(lambda: seen.append(self.session.state.config.exposure.density))

        cfg = replace(self.session.state.config, exposure=replace(self.session.state.config.exposure, density=1.5))
        self.session.update_config(cfg, persist=True)

        self.assertEqual(seen, [1.5])

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
            geometry=replace(self.session.state.config.geometry, rotation=2, crop_rect=(0.1, 0.1, 0.9, 0.9)),
        )
        self.session.update_config(dirty, persist=True)

        self.session.reset_settings()

        self.assertEqual(self.session.state.config, DEFAULT_WORKSPACE_CONFIG)
        self.assertFalse(self.session.state.config.process.is_local_initialized)
        self.assertFalse(self.session.state.config.process.is_locked_initialized)

    def test_reset_settings_is_mode_aware_on_cast_removal(self):
        """A transparency's own neutral Cast Removal point is 0, not the flat 0.5
        DEFAULT_WORKSPACE_CONFIG carries for a negative (cast_removal_for_mode).
        resolve_asset_process_mode reads the mode to reset into from the asset dict, not
        the pre-reset WorkspaceConfig, so the fixture carries it there."""
        self.session.state.uploaded_files[0]["process_mode"] = str(ProcessMode.E6)
        self.session.select_file(0)
        dirty = replace(
            self.session.state.config,
            process=replace(self.session.state.config.process, process_mode=ProcessMode.E6),
            exposure=replace(self.session.state.config.exposure, cast_removal_strength=0.8),
        )
        self.session.update_config(dirty, persist=True)

        self.session.reset_settings()

        self.assertEqual(self.session.state.config.process.process_mode, ProcessMode.E6)
        self.assertEqual(self.session.state.config.exposure.cast_removal_strength, 0.0)

    def test_reset_process_section_resyncs_cast_removal_for_the_new_mode(self):
        """Resetting Process can change process_mode (back to DEFAULT_WORKSPACE_CONFIG's
        C41), which must re-sync Cast Removal too -- otherwise an E6-tuned 0.0 survives
        onto a negative, where the flat default belongs."""
        self.session.select_file(0)
        dirty = replace(
            self.session.state.config,
            process=replace(self.session.state.config.process, process_mode=ProcessMode.E6),
            exposure=replace(self.session.state.config.exposure, cast_removal_strength=0.0),
        )
        self.session.update_config(dirty, persist=True)

        self.session.reset_section("process")

        self.assertEqual(self.session.state.config.process.process_mode, ProcessMode.C41)
        self.assertEqual(self.session.state.config.exposure.cast_removal_strength, 1.0)

    def test_reset_settings_is_recorded_not_wiping(self):
        self.session.select_file(0)
        edited = replace(self.session.state.config, exposure=replace(self.session.state.config.exposure, density=1.8))
        self.session.update_config(edited, persist=True)

        self.session.reset_settings()

        self.mock_repo.clear_history.assert_not_called()
        self.assertEqual(self.session.state.config, DEFAULT_WORKSPACE_CONFIG)
        # Reset pushed the pre-reset config as a history step — it is undoable.
        self.mock_repo.save_history_step.assert_called_with("hash1", 1, edited)
        self.assertEqual(self.session.state.undo_index, 2)

    def test_reset_roll_resets_the_active_frame_in_place(self):
        self.session.select_file(0)
        dirty = replace(self.session.state.config, exposure=replace(self.session.state.config.exposure, density=1.8))
        self.session.update_config(dirty, persist=True)

        self.session.reset_roll(self.session.state.uploaded_files)

        self.assertEqual(self.session.state.config, self.session._reset_frame(self.session.state.uploaded_files[0]))

    def test_reset_roll_writes_other_frames_straight_to_the_db(self):
        self.session.select_file(0)  # hash1 active; hash2 is the "other" frame

        self.session.reset_roll(self.session.state.uploaded_files)

        expected = self.session._reset_frame(self.session.state.uploaded_files[1])
        self.mock_repo.save_file_settings.assert_any_call("hash2", expected, file_path="path2")
        # Recorded as an external history step, undoable after switching to it.
        steps = [c.args for c in self.mock_repo.save_history_step.call_args_list if c.args[0] == "hash2"]
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[1][2], expected)

    def test_reset_roll_does_not_touch_a_frame_outside_the_given_list(self):
        self.session.select_file(0)
        only_hash1 = [self.session.state.uploaded_files[0]]

        self.session.reset_roll(only_hash1)

        for c in self.mock_repo.save_file_settings.call_args_list:
            self.assertNotEqual(c.args[0], "hash2")

    def test_rehome_folder_paths_repoints_every_matching_asset(self):
        self.session.state.uploaded_files = [
            {"name": "a.tif", "path": "/scans/roll_a/a.tif", "hash": "ha"},
            {"name": "b.tif", "path": "/scans/roll_a/sub/b.tif", "hash": "hb"},
            {"name": "c.tif", "path": "/elsewhere/c.tif", "hash": "hc"},
        ]

        self.session.rehome_folder_paths("/scans/roll_a", "/scans/roll_b")

        paths = [f["path"] for f in self.session.state.uploaded_files]
        self.assertEqual(paths, ["/scans/roll_b/a.tif", "/scans/roll_b/sub/b.tif", "/elsewhere/c.tif"])

    def test_rehome_folder_paths_updates_the_active_file_path(self):
        self.session.state.uploaded_files = [{"name": "a.tif", "path": "/scans/roll_a/a.tif", "hash": "ha"}]
        self.session.state.current_file_path = "/scans/roll_a/a.tif"

        self.session.rehome_folder_paths("/scans/roll_a", "/scans/roll_b")

        self.assertEqual(self.session.state.current_file_path, "/scans/roll_b/a.tif")

    def test_rehome_folder_paths_rewrites_composite_part_paths(self):
        self.session.state.uploaded_files = [
            {
                "name": "triplet",
                "path": "/scans/roll_a/r.tif",
                "hash": "ha",
                "green_path": "/scans/roll_a/g.tif",
                "blue_path": "/scans/roll_a/b.tif",
            },
            {
                "name": "stitch",
                "path": "/scans/roll_a/1.tif",
                "hash": "hb",
                "stitch_paths": ["/scans/roll_a/1.tif", "/scans/roll_a/2.tif"],
                "stitch_transforms": [[1, 0, 0], [0, 1, 0]],
                "stitch_canvas": [100, 100],
                "stitch_sizes": [[50, 100], [50, 100]],
            },
        ]

        self.session.rehome_folder_paths("/scans/roll_a", "/scans/roll_b")

        triplet, stitch = self.session.state.uploaded_files
        self.assertEqual(triplet["green_path"], "/scans/roll_b/g.tif")
        self.assertEqual(triplet["blue_path"], "/scans/roll_b/b.tif")
        self.assertEqual(stitch["stitch_paths"], ["/scans/roll_b/1.tif", "/scans/roll_b/2.tif"])

    def test_rehome_folder_paths_is_a_noop_when_nothing_matches(self):
        self.session.state.uploaded_files = [{"name": "c.tif", "path": "/elsewhere/c.tif", "hash": "hc"}]
        self.mock_repo.save_global_setting.reset_mock()

        self.session.rehome_folder_paths("/scans/roll_a", "/scans/roll_b")

        self.assertEqual(self.session.state.uploaded_files[0]["path"], "/elsewhere/c.tif")
        self.mock_repo.save_global_setting.assert_not_called()

    def test_sync_to_roll_records_target_history(self):
        self.mock_repo.get_max_history_index.return_value = 0
        self.mock_repo.load_history_step.return_value = None
        self.session.select_file(0)
        self.session.state.uploaded_files.append({"name": "file3.dng", "path": "path3", "hash": "hash3"})
        self.session.asset_model.refresh()
        self.mock_repo.save_history_step.reset_mock()

        count = self.session.sync_selected_settings([_row("Print Density")], scope="roll")

        self.assertEqual(count, 2)
        # Each target got a two-step write: pre-apply at 0, post-apply at 1.
        steps = [(c.args[0], c.args[1]) for c in self.mock_repo.save_history_step.call_args_list]
        self.assertEqual(steps, [("hash2", 0), ("hash2", 1), ("hash3", 0), ("hash3", 1)])

    def test_rotate_selected_frames_applies_each_frames_own_geometry(self):
        self.session.state.selected_file_idx = 0
        self.session.state.config = WorkspaceConfig(geometry=GeometryConfig(rotation=1))
        self.mock_repo.load_file_settings.return_value = WorkspaceConfig(geometry=GeometryConfig(rotation=3))
        self.session.update_selection([0, 1])

        touched = self.session.rotate_selected_frames(1)

        self.assertEqual(touched, ["hash2-v3"])
        args, kwargs = self.mock_repo.save_file_settings.call_args
        self.assertEqual(args[0], "hash2")
        # Frame 1 rotates from its OWN stored rotation (3), not the active frame's new value.
        self.assertEqual(args[1].geometry.rotation, 0)
        self.assertEqual(kwargs["file_path"], "path2")

    def test_rotate_selected_frames_message_excludes_active_when_deselected(self):
        self.session.state.uploaded_files.append({"name": "file3.dng", "path": "path3", "hash": "hash3"})
        self.session.state.selected_file_idx = 0
        self.mock_repo.load_file_settings.return_value = WorkspaceConfig(geometry=GeometryConfig(rotation=0))
        self.session.update_selection([1, 2])
        messages = []
        self.session.settings_synced.connect(messages.append)

        touched = self.session.rotate_selected_frames(1, active_included=False)

        self.assertEqual(touched, ["hash2-v3", "hash3-v3"])
        self.assertEqual(messages, ["Rotated 2 frames"])

    def test_rotate_selected_frames_noop_on_single_selection(self):
        self.session.state.selected_file_idx = 0
        self.session.update_selection([0])

        touched = self.session.rotate_selected_frames(1)

        self.assertEqual(touched, [])
        self.mock_repo.save_file_settings.assert_not_called()

    def test_rotate_selected_frames_records_undoable_history(self):
        self.mock_repo.get_max_history_index.return_value = 0
        self.mock_repo.load_history_step.return_value = None
        self.mock_repo.save_history_step.reset_mock()
        self.session.state.selected_file_idx = 0
        target_config = WorkspaceConfig(geometry=GeometryConfig(rotation=0))
        self.mock_repo.load_file_settings.return_value = target_config
        self.session.update_selection([0, 1])

        self.session.rotate_selected_frames(1)

        steps = [(c.args[0], c.args[1]) for c in self.mock_repo.save_history_step.call_args_list]
        self.assertEqual(steps, [("hash2", 0), ("hash2", 1)])

    def test_flip_selected_frames_applies_each_frames_own_geometry(self):
        self.session.state.selected_file_idx = 0
        self.session.state.config = WorkspaceConfig(geometry=GeometryConfig(flip_horizontal=False))
        self.mock_repo.load_file_settings.return_value = WorkspaceConfig(geometry=GeometryConfig(flip_horizontal=True))
        self.session.update_selection([0, 1])

        touched = self.session.flip_selected_frames(True)

        self.assertEqual(touched, ["hash2-v3"])
        args, kwargs = self.mock_repo.save_file_settings.call_args
        self.assertEqual(args[0], "hash2")
        # Frame 1 mirrors from its OWN stored flip (True), not the active frame's new value.
        self.assertFalse(args[1].geometry.flip_horizontal)
        self.assertEqual(kwargs["file_path"], "path2")

    def test_flip_selected_frames_noop_on_single_selection(self):
        self.session.state.selected_file_idx = 0
        self.session.update_selection([0])

        touched = self.session.flip_selected_frames(True)

        self.assertEqual(touched, [])
        self.mock_repo.save_file_settings.assert_not_called()

    def test_flip_selected_frames_records_undoable_history(self):
        self.mock_repo.get_max_history_index.return_value = 0
        self.mock_repo.load_history_step.return_value = None
        self.mock_repo.save_history_step.reset_mock()
        self.session.state.selected_file_idx = 0
        target_config = WorkspaceConfig(geometry=GeometryConfig(flip_horizontal=False))
        self.mock_repo.load_file_settings.return_value = target_config
        self.session.update_selection([0, 1])

        self.session.flip_selected_frames(True)

        steps = [(c.args[0], c.args[1]) for c in self.mock_repo.save_history_step.call_args_list]
        self.assertEqual(steps, [("hash2", 0), ("hash2", 1)])

    def test_rotate_selected_frames_skips_a_duplicate_content_hash(self):
        # Two open paths sharing the active frame's content hash (e.g. the same file
        # loaded from two folders) must not turn the shared edit row twice.
        self.session.state.uploaded_files.append({"name": "file1-copy.dng", "path": "path1-copy", "hash": "hash1"})
        self.session.state.selected_file_idx = 0
        self.session.state.current_file_hash = "hash1"
        self.mock_repo.load_file_settings.return_value = WorkspaceConfig(geometry=GeometryConfig(rotation=0))
        self.session.update_selection([0, 1, 2])

        touched = self.session.rotate_selected_frames(1)

        self.assertEqual(touched, ["hash2-v3"])
        self.mock_repo.save_file_settings.assert_called_once()
        self.assertEqual(self.mock_repo.save_file_settings.call_args.args[0], "hash2")

    def test_reset_roll_settings_resets_every_visible_frame(self):
        self._seed_roll()
        self.session.asset_model.refresh()
        dirty = replace(self.session.state.config, exposure=replace(self.session.state.config.exposure, density=1.8))
        self.session.update_config(dirty, persist=True)
        self.mock_repo.load_file_settings.return_value = dirty

        count = self.session.reset_roll_settings(scope="roll")

        self.assertEqual(count, 3)
        self.assertEqual(self.session.state.config, DEFAULT_WORKSPACE_CONFIG)
        saved = {c.args[0]: c.args[1] for c in self.mock_repo.save_file_settings.call_args_list}
        self.assertEqual(saved["hash1"], DEFAULT_WORKSPACE_CONFIG)
        self.assertEqual(saved["hash2"], DEFAULT_WORKSPACE_CONFIG)
        self.assertEqual(saved["hash3"], DEFAULT_WORKSPACE_CONFIG)

    def test_reset_roll_settings_selection_scope_resets_only_selected_frames(self):
        self._seed_roll()
        self.session.asset_model.refresh()
        self.session.state.selected_indices = [0, 1]  # c.jpg (index 2) left untouched
        dirty = replace(self.session.state.config, exposure=replace(self.session.state.config.exposure, density=1.8))
        self.session.update_config(dirty, persist=True)
        self.mock_repo.load_file_settings.return_value = dirty

        count = self.session.reset_roll_settings(scope="selection")

        self.assertEqual(count, 2)
        self.assertEqual(self.session.state.config, DEFAULT_WORKSPACE_CONFIG)
        saved = {c.args[0] for c in self.mock_repo.save_file_settings.call_args_list}
        self.assertEqual(saved, {"hash1", "hash2"})
        self.assertNotIn("hash3", saved)  # not in the selection, left untouched

    def test_reset_roll_settings_respects_active_filter(self):
        # Mirrors sync_selected_settings: "whole roll" means the visible (filtered) frames.
        self._seed_roll()
        self.session.asset_model.set_filter(".arw", regex=False)  # hides c.jpg

        count = self.session.reset_roll_settings(scope="roll")

        self.assertEqual(count, 2)
        saved = {c.args[0] for c in self.mock_repo.save_file_settings.call_args_list}
        self.assertEqual(saved, {"hash1", "hash2"})  # c.jpg filtered out, not touched

    def test_reset_roll_settings_preserves_asset_derived_process_mode(self):
        # A composite's inherited film process is what it *is*, not an edit — a blind
        # WorkspaceConfig() overlay would wipe it, unlike _asset_defaults.
        self._seed_roll()
        self.session.asset_model.refresh()
        self.session.state.uploaded_files[1]["process_mode"] = ProcessMode.E6
        stale = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, process_mode=ProcessMode.C41))
        self.mock_repo.load_file_settings.return_value = stale

        self.session.reset_roll_settings(scope="roll")

        saved = {c.args[0]: c.args[1] for c in self.mock_repo.save_file_settings.call_args_list}
        self.assertEqual(saved["hash2"].process.process_mode, ProcessMode.E6)

    def test_reset_roll_settings_emits_frames_edited_offscreen(self):
        self._seed_roll()
        self.session.asset_model.refresh()
        offscreen = []
        self.session.frames_edited_offscreen.connect(offscreen.append)

        self.session.reset_roll_settings(scope="roll")

        self.assertEqual(offscreen, [["hash2", "hash3"]])  # active frame (hash1) excluded

    def test_reset_roll_settings_selection_scope_emits_frames_edited_offscreen(self):
        self._seed_roll()
        self.session.asset_model.refresh()
        self.session.state.selected_indices = [0, 1]  # c.jpg (index 2) left out of scope
        offscreen = []
        self.session.frames_edited_offscreen.connect(offscreen.append)

        self.session.reset_roll_settings(scope="selection")

        self.assertEqual(offscreen, [["hash2"]])  # active frame excluded, hash3 out of scope

    def test_reset_roll_settings_single_frame_emits_nothing(self):
        self._seed_roll()
        self.session.asset_model.refresh()
        self.session.state.selected_indices = [0]  # only the active frame
        offscreen = []
        self.session.frames_edited_offscreen.connect(offscreen.append)

        self.session.reset_roll_settings(scope="selection")

        self.assertEqual(offscreen, [])

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

    def test_a_stale_plain_filter_survives_switching_to_a_semantic_query(self):
        """A plain filter set before search-by-meaning was ever turned on stays live
        in _filter_text; set_semantic_query only ever touches _semantic_query. Dormant
        while the semantic query is active (_rebuild_indices returns on it first), it
        resurfaces the moment the semantic query is cleared -- which is exactly what a
        library-wide search hand-off does to avoid re-running its own outlier check on
        an already-curated set. clear_filters is the fix: it clears both in one go."""
        import numpy as np

        self.model.set_filter("img", regex=False)
        self.model.set_semantic_query(np.zeros(2, dtype=np.float32))
        self.assertEqual(len(self.model._sorted_indices), 0)  # no cached embeddings -- dormant filter, not this

        self.model.set_semantic_query(None)
        self.assertEqual(set(self._names()), {"IMG_0001.cr2", "IMG_0002.cr2"})  # the stale filter resurfaces

        self.model.set_filter("img", regex=False)
        self.model.set_semantic_query(np.zeros(2, dtype=np.float32))
        self.model.clear_filters()
        self.assertEqual(len(self.model._sorted_indices), 5)  # both cleared -- the full hand-off shows through


class TestAssetListModelSceneSort(unittest.TestCase):
    """Scene sort: each scene's frames as one block in scene order, frames in no scene last."""

    def setUp(self):
        self.state = AppState()
        self.state.uploaded_files = [
            {"name": "a.nef", "path": "/r/a.nef", "hash": "h1"},
            {"name": "b.nef", "path": "/r/b.nef", "hash": "h2", "scene": (2, "s2", "Night")},
            {"name": "c.nef", "path": "/r/c.nef", "hash": "h3", "scene": (1, "s1", "Beach")},
            {"name": "d.nef", "path": "/r/d.nef", "hash": "h4", "scene": (2, "s2", "Night")},
            {"name": "e.nef", "path": "/r/e.nef", "hash": "h5", "scene": (1, "s1", "Beach")},
        ]
        self.model = AssetListModel(self.state)
        self.model.set_sort_order("scene")

    def _names(self):
        return [self.state.uploaded_files[i]["name"] for i in self.model._sorted_indices]

    def test_scenes_in_order_then_frames_in_no_scene(self):
        self.assertEqual(self._names(), ["c.nef", "e.nef", "b.nef", "d.nef", "a.nef"])
        self.assertEqual(self.model.scene_runs(), [(1, 0, 1), (2, 2, 3), (None, 4, 4)])

    def test_descending_reverses_scenes_and_names_but_keeps_no_scene_last(self):
        self.model.set_sort_descending(True)
        self.assertEqual(self._names(), ["d.nef", "b.nef", "e.nef", "c.nef", "a.nef"])

    def test_reads_as_name_without_scenes_and_keeps_the_choice(self):
        for f in self.state.uploaded_files:
            f.pop("scene", None)
        self.model.refresh()

        self.assertEqual(self.model.effective_sort_order, "name")
        self.assertEqual(self._names(), ["a.nef", "b.nef", "c.nef", "d.nef", "e.nef"])
        self.assertEqual(self.model.scene_runs(), [])
        self.assertEqual(self.model._sort_order, "scene")

    def test_a_filter_keeps_the_blocks(self):
        self.model.set_filter("[bcd]", regex=True)
        self.assertEqual(self._names(), ["c.nef", "b.nef", "d.nef"])
        self.assertEqual(self.model.scene_runs(), [(1, 0, 0), (2, 1, 2)])

    def test_no_blocks_under_other_orders_or_a_semantic_query(self):
        import numpy as np

        self.model.set_sort_order("date")
        self.assertEqual(self.model.scene_runs(), [])
        self.model.set_sort_order("scene")
        self.model._semantic_query = np.zeros(4)
        self.assertEqual(self.model.scene_runs(), [])


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

    def test_clear_files_drops_cached_embeddings(self):
        import numpy as np

        self.session.state.embeddings["h1"] = np.zeros(2, dtype=np.float32)
        self.session.clear_files()
        self.assertEqual(self.session.state.embeddings, {})

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
        # The path rides along so a library search (which never hashes) can see the mark.
        self.mock_repo.save_file_mark.assert_called_with("hash1", "excluded", file_path="p1")

        self.session.toggle_mark("excluded")
        self.assertFalse(self.session.state.uploaded_files[0]["excluded"])
        self.mock_repo.save_file_mark.assert_called_with("hash1", None, file_path="p1")

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

    def test_a_path_turned_away_as_a_duplicate_is_recorded(self):
        """A duplicate never reaches uploaded_files, so a caller that decides what is
        new by path (the Hot Folder poll) has nothing to learn from the file list and
        would re-offer it every round. The rejected path is kept for them instead."""
        loaded = self.session.state.uploaded_files[0]

        self.session.add_files([], validated_info=[{"name": "copy.dng", "path": "/hot/copy.dng", "hash": loaded["hash"]}])

        self.assertNotIn("/hot/copy.dng", [f["path"] for f in self.session.state.uploaded_files])
        self.assertIn("/hot/copy.dng", self.session.state.duplicate_paths)

    def test_clearing_the_session_forgets_the_duplicates_it_turned_away(self):
        """The rejection only held while the frame it clashed with was loaded."""
        self.session.state.duplicate_paths.add("/hot/copy.dng")

        self.session.clear_files()

        self.assertEqual(self.session.state.duplicate_paths, set())


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
        from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask

        # hash1 has two masks on disk; index 1 is hidden. The property clamps against the
        # hydrated mask list, so persistence only "counts" if that config reloads too.
        verts = ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9))
        two_masks = (LocalMask(vertices=verts), LocalMask(vertices=verts, stops=0.3))
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

        count = self.session.sync_selected_settings([_row("Print Density")], scope="roll")
        self.assertEqual(count, 1)
        self.assertEqual(self.repo.load_file_settings("hash2").exposure.density, 1.5)

        self.session.select_file(1)
        self.assertEqual(self.session.state.config.exposure.density, 1.5)

        self.session.undo()
        self.assertEqual(self.session.state.config.exposure.density, 2.0)

        self.session.redo()
        self.assertEqual(self.session.state.config.exposure.density, 1.5)


class TestThumbnailKeying(unittest.TestCase):
    """Thumbnails are keyed by asset identity, not display name."""

    def test_same_named_files_in_two_folders_keep_distinct_thumbnails(self):
        from PyQt6.QtCore import Qt

        from negpy.services.assets.thumbnails import asset_thumbnail_key

        state = AppState()
        state.uploaded_files = [
            {"name": "IMG_0042.tif", "path": "/a/IMG_0042.tif", "hash": "h1"},
            {"name": "IMG_0042.tif", "path": "/b/IMG_0042.tif", "hash": "h2"},
        ]
        state.thumbnails = {asset_thumbnail_key(f): f"thumb-{i}" for i, f in enumerate(state.uploaded_files)}
        model = AssetListModel(state)

        first = model.data(model.index(0, 0), Qt.ItemDataRole.DecorationRole)
        second = model.data(model.index(1, 0), Qt.ItemDataRole.DecorationRole)
        self.assertEqual({first, second}, {"thumb-0", "thumb-1"})

    def test_triplet_key_is_namespaced_away_from_its_red_exposure(self):
        from negpy.services.assets.thumbnails import asset_thumbnail_key

        red = {"name": "a.nef", "path": "/a.nef", "hash": "h1"}
        triplet = {**red, "name": "a (RGB)", "green_path": "/g.nef", "blue_path": "/b.nef"}
        self.assertTrue(asset_thumbnail_key(red).startswith("h1"))
        self.assertNotEqual(asset_thumbnail_key(triplet), asset_thumbnail_key(red))
        self.assertIn("-rgb", asset_thumbnail_key(triplet))

    def test_unloading_one_frame_keeps_its_twins_thumbnail(self):
        repo = MagicMock(spec=StorageRepository)
        repo.get_global_setting.return_value = None
        repo.load_file_settings.return_value = None
        repo.load_file_settings_by_path.return_value = None
        repo.load_file_settings_many.return_value = {}
        repo.get_max_history_index.return_value = 0
        session = DesktopSessionManager(repo)
        session.state.uploaded_files = [
            {"name": "IMG_0042.tif", "path": "/a/IMG_0042.tif", "hash": "h1"},
            {"name": "IMG_0042.tif", "path": "/b/IMG_0042.tif", "hash": "h2"},
        ]
        from negpy.services.assets.thumbnails import asset_thumbnail_key

        keys = [asset_thumbnail_key(f) for f in session.state.uploaded_files]
        session.state.thumbnails = {keys[0]: "thumb-a", keys[1]: "thumb-b"}
        session.state.selected_file_idx = 0
        session.state.selected_indices = [0]

        session.remove_current_file()

        self.assertEqual(session.state.thumbnails, {keys[1]: "thumb-b"})

    def test_push_external_history_flags_stale_under_the_thumbnail_cache_key(self):
        """push_external_history (a bulk apply reaching a non-active file) adds
        asset_thumbnail_key(asset) to stale_thumbnails, not the bare hash -- the read
        side (the film strip's dot, the tooltip) must key the same way or the flag,
        though set, never matches anything and the indicator never shows."""
        from PyQt6.QtCore import Qt

        from negpy.desktop.view.sidebar.files import _ThumbnailDelegate
        from negpy.services.assets.thumbnails import asset_thumbnail_key

        repo = MagicMock(spec=StorageRepository)
        repo.get_global_setting.return_value = None
        repo.load_file_settings.return_value = None
        repo.load_file_settings_by_path.return_value = None
        repo.load_file_settings_many.return_value = {}
        repo.get_max_history_index.return_value = 0
        session = DesktopSessionManager(repo)
        asset = {"name": "a.nef", "path": "/a.nef", "hash": "h1"}
        session.state.uploaded_files = [asset]

        session.push_external_history("h1", WorkspaceConfig(), WorkspaceConfig())

        self.assertIn(asset_thumbnail_key(asset), session.state.stale_thumbnails)

        model = AssetListModel(session.state)
        tooltip = model.data(model.index(0, 0), Qt.ItemDataRole.ToolTipRole)
        self.assertIn("predates a settings change", tooltip)

        delegate = _ThumbnailDelegate(state=session.state)
        self.assertTrue(delegate._is_stale_thumbnail(asset))


class TestSearchFacts(unittest.TestCase):
    """Metadata filtering against a real repository: the sheet filter sees a frame's
    saved edit, and the facts follow later writes."""

    def setUp(self):
        import tempfile

        from negpy.features.metadata.models import MetadataConfig

        self.tmp = tempfile.TemporaryDirectory()
        self.repo = StorageRepository(f"{self.tmp.name}/edits.db", f"{self.tmp.name}/settings.db")
        self.repo.initialize()
        self.session = DesktopSessionManager(self.repo)
        self.session.state.uploaded_files = [
            {"name": "a.dng", "path": f"{self.tmp.name}/a.dng", "hash": "hash1", "mtime": 1710000000.0},
            {"name": "b.dng", "path": f"{self.tmp.name}/b.dng", "hash": "hash2", "mtime": 1710000000.0},
        ]
        portra = replace(WorkspaceConfig(), metadata=MetadataConfig(film="Portra 400", film_iso=400, camera_model="F3"))
        self.repo.save_file_settings("hash1", portra, file_path=self.session.state.uploaded_files[0]["path"])
        self.session.asset_model.refresh()

    def tearDown(self):
        self.tmp.cleanup()

    def _visible(self) -> set:
        files = self.session.state.uploaded_files
        return {files[i]["name"] for i in self.session.asset_model.visible_actual_indices_ordered()}

    def test_metadata_term_filters_the_sheet(self):
        self.session.asset_model.set_filter("film:portra", regex=False)
        self.assertEqual(self._visible(), {"a.dng"})

    def test_numeric_and_flag_terms(self):
        self.session.asset_model.set_filter("iso:>=400", regex=False)
        self.assertEqual(self._visible(), {"a.dng"})
        self.session.asset_model.set_filter("-edited:", regex=False)
        self.assertEqual(self._visible(), {"b.dng"})

    def test_bare_word_still_matches_the_filename(self):
        self.session.asset_model.set_filter("b.dng", regex=False)
        self.assertEqual(self._visible(), {"b.dng"})

    def test_facts_are_cached_until_a_write_invalidates_them(self):
        first = self.session.search_facts()
        self.assertIs(self.session.search_facts(), first)

        self.session.settings_saved.emit()
        self.assertIsNot(self.session.search_facts(), first)

    def test_facts_follow_a_later_settings_write(self):
        from negpy.features.metadata.models import MetadataConfig

        self.session.asset_model.set_filter("film:velvia", regex=False)
        self.assertEqual(self._visible(), set())

        velvia = replace(WorkspaceConfig(), metadata=MetadataConfig(film="Velvia 50"))
        self.repo.save_file_settings("hash2", velvia, file_path=self.session.state.uploaded_files[1]["path"])
        self.session.settings_saved.emit()

        self.session.asset_model.set_filter("film:velvia", regex=False)
        self.assertEqual(self._visible(), {"b.dng"})


class ResetKeepsScanSetup(unittest.TestCase):
    """A reset clears the look but keeps the rig: Linear RAW and Narrowband come back as
    the scan setup gives them to a fresh file, on every reset path."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = StorageRepository(f"{self.tmp.name}/edits.db", f"{self.tmp.name}/settings.db")
        self.repo.initialize()
        self.repo.save_global_settings({"last_linear_raw": True, "last_narrowband_scan": True})
        self.session = DesktopSessionManager(self.repo)
        self.session.state.uploaded_files = [
            {"name": "a.raw", "path": f"{self.tmp.name}/a.raw", "hash": "hash1"},
            {"name": "b.raw", "path": f"{self.tmp.name}/b.raw", "hash": "hash2"},
        ]
        edited = replace(
            DEFAULT_WORKSPACE_CONFIG,
            process=replace(DEFAULT_WORKSPACE_CONFIG.process, linear_raw=True, narrowband_scan=True),
            exposure=replace(DEFAULT_WORKSPACE_CONFIG.exposure, density=1.8),
        )
        for f in self.session.state.uploaded_files:
            self.repo.save_file_settings(f["hash"], edited, file_path=f["path"])
        self.session.asset_model.refresh()
        self.session.select_file(0)

    def _assert_reset_keeping_rig(self, config: WorkspaceConfig) -> None:
        self.assertTrue(config.process.narrowband_scan)
        self.assertTrue(config.process.linear_raw)
        self.assertEqual(config.exposure.density, DEFAULT_WORKSPACE_CONFIG.exposure.density)

    def test_reset_settings(self):
        self.session.reset_settings()

        self._assert_reset_keeping_rig(self.session.state.config)

    def test_reset_roll_settings(self):
        self.session.reset_roll_settings(scope="roll")

        self._assert_reset_keeping_rig(self.session.state.config)
        self._assert_reset_keeping_rig(self.repo.load_file_settings("hash2"))

    def test_reset_roll(self):
        self.session.reset_roll(self.session.state.uploaded_files)

        self._assert_reset_keeping_rig(self.session.state.config)
        self._assert_reset_keeping_rig(self.repo.load_file_settings("hash2"))

    def test_both_roll_resets_write_the_same_frame(self):
        self.session.reset_roll_settings(scope="roll")
        via_settings = self.repo.load_file_settings("hash2")
        self.session.reset_roll(self.session.state.uploaded_files)

        self.assertEqual(self.repo.load_file_settings("hash2"), via_settings)

    def _locked_roll(self) -> str:
        """The reported roll: Narrowband on for the roll, a frame locked to its own off."""
        from negpy.services.assets import rolls

        roll_id = rolls.create_virtual_roll(self.repo, "Pakon", [f["path"] for f in self.session.state.uploaded_files])
        rolls.set_roll_defaults(self.repo, roll_id, narrowband_scan=True)
        for h in ("hash1", "hash2"):
            rolls.set_frame_override(self.repo, roll_id, h, "sensor", locked=True)
        self.repo.save_global_settings({"last_narrowband_scan": False})
        self.session.state.active_roll_id = roll_id
        return roll_id

    def test_a_reset_in_a_roll_releases_the_frames_locks(self):
        from negpy.services.assets import rolls

        roll_id = self._locked_roll()

        self.session.reset_roll_settings(scope="roll")

        for h in ("hash1", "hash2"):
            self.assertEqual(rolls.frame_override_cards(self.repo, roll_id, h), set())
        asset = self.session.state.uploaded_files[1]
        self.assertTrue(self.session.config_for_asset(asset).process.narrowband_scan)

    def test_a_reset_in_a_roll_shows_the_roll_values_at_once(self):
        self._locked_roll()

        self.session.reset_settings()

        self.assertTrue(self.session.state.config.process.narrowband_scan)

    def test_undoing_a_reset_in_a_roll_keeps_the_frames_own_values(self):
        from negpy.services.assets import rolls

        roll_id = self._locked_roll()
        own = self.session.state.config
        own = replace(own, process=replace(own.process, narrowband_scan=False))
        self.session.update_config(own, persist=True)

        self.session.reset_settings()
        self.session.undo()

        self.assertFalse(self.session.state.config.process.narrowband_scan)
        self.assertIn("sensor", rolls.frame_override_cards(self.repo, roll_id, "hash1"))
        self.session.repo.save_file_settings("hash1", self.session.state.config, file_path=self.session.state.uploaded_files[0]["path"])
        self.assertFalse(self.session.config_for_asset(self.session.state.uploaded_files[0]).process.narrowband_scan)

    def test_undo_leaves_a_card_pinned_at_the_rolls_value_pinned(self):
        from negpy.services.assets import rolls

        roll_id = self._locked_roll()
        dirty = self.session.state.config
        self.session.update_config(replace(dirty, exposure=replace(dirty.exposure, density=1.2)), persist=True)
        rolls.set_frame_override(self.repo, roll_id, "hash1", "film", locked=True)

        self.session.undo()

        self.assertIn("film", rolls.frame_override_cards(self.repo, roll_id, "hash1"))

    def _own_narrowband_then_reset(self) -> str:
        """The frame's own Narrowband off as a history step, then a reset that hands the
        frame back to the roll: Narrowband on, the Calibration card unlocked."""
        roll_id = self._locked_roll()
        own = self.session.state.config
        self.session.update_config(replace(own, process=replace(own.process, narrowband_scan=False)), persist=True)
        self.session.save_work_print("own")
        self.session.reset_settings()
        return roll_id

    def test_a_history_jump_locks_a_card_that_differs_from_the_roll(self):
        from negpy.services.assets import rolls

        roll_id = self._own_narrowband_then_reset()

        self.session.jump_to_step(self.session.state.undo_index - 1)

        self.assertFalse(self.session.state.config.process.narrowband_scan)
        self.assertIn("sensor", rolls.frame_override_cards(self.repo, roll_id, "hash1"))

    def test_loading_a_work_print_locks_a_card_that_differs_from_the_roll(self):
        from negpy.services.assets import rolls

        roll_id = self._own_narrowband_then_reset()

        self.session.load_work_print("own")

        self.assertFalse(self.session.state.config.process.narrowband_scan)
        self.assertIn("sensor", rolls.frame_override_cards(self.repo, roll_id, "hash1"))

    def test_a_paste_locks_a_card_that_differs_from_the_roll(self):
        from negpy.services.assets import rolls

        roll_id = self._locked_roll()
        rolls.set_roll_defaults(self.repo, roll_id, hue_trim=0.0)
        own = self.session.state.config
        self.session.update_config(replace(own, process=replace(own.process, hue_trim=2.0)), persist=True)
        self.session.copy_settings()
        self.session.reset_settings()
        self.assertNotIn("sensor", rolls.frame_override_cards(self.repo, roll_id, "hash1"))

        self.session.apply_pasted_fields([r for r in all_rows() if "hue_trim" in r.fields], include_bounds=False)

        self.assertEqual(self.session.state.config.process.hue_trim, 2.0)
        self.assertIn("sensor", rolls.frame_override_cards(self.repo, roll_id, "hash1"))

    def test_reset_settings_with_a_roll_open_and_no_frame_selected(self):
        self._locked_roll()
        self.session.state.selected_file_idx = -1

        self.session.reset_settings()

    def test_a_white_light_scan_setup_resets_to_off(self):
        self.repo.save_global_settings({"last_linear_raw": False, "last_narrowband_scan": False})

        self.session.reset_settings()

        self.assertFalse(self.session.state.config.process.narrowband_scan)
        self.assertFalse(self.session.state.config.process.linear_raw)


if __name__ == "__main__":
    unittest.main()
