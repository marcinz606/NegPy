import os
import unittest
from dataclasses import replace
from unittest.mock import MagicMock

import negpy.kernel.system.paths as paths
from negpy.desktop.settings_catalog import (
    DEFAULT_STICKY_IDS,
    GLOBAL_TIER_SECTIONS,
    all_rows,
    rows_by_id,
)
from negpy.desktop.sticky import (
    ALWAYS_STICKY_PROCESS,
    EXPORT_REMAINDER,
    STICKY_CONFIG_KEY,
    STICKY_ROWS_KEY,
    load_sticky_rows,
    migrate_auto_crop_sticky_row,
    migrate_carried_chroma_zero,
    migrate_legacy,
    migrate_legacy_export_destination,
    save_sticky_rows,
    sticky_snapshot,
)
from negpy.domain.models import ExportConfig, WorkspaceConfig


def _repo(settings=None):
    store = dict(settings or {})
    repo = MagicMock()
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    repo.store = store
    return repo


class TestCatalogStickyIds(unittest.TestCase):
    def test_row_ids_are_unique(self):
        self.assertEqual(len(rows_by_id()), len(all_rows()))

    def test_default_sticky_ids_all_resolve(self):
        by_id = rows_by_id()
        for row_id in DEFAULT_STICKY_IDS:
            self.assertIn(row_id, by_id)

    def test_export_remainder_completes_the_catalog(self):
        """Every ExportConfig field either has a row or rides the remainder. Without this
        a new export field would silently stop carrying over."""
        catalog_export = {f for r in all_rows() if r.section == "export" for f in r.fields}
        self.assertEqual(catalog_export | EXPORT_REMAINDER, set(ExportConfig.__dataclass_fields__))
        self.assertFalse(catalog_export & EXPORT_REMAINDER)

    def test_global_tier_sections_are_real_config_attrs(self):
        sections = {r.section for r in all_rows()}
        self.assertTrue(GLOBAL_TIER_SECTIONS <= sections)


class TestStickyStore(unittest.TestCase):
    def test_defaults_when_user_never_chose(self):
        rows = load_sticky_rows(_repo())
        self.assertEqual({r.id for r in rows}, set(DEFAULT_STICKY_IDS))

    def test_stored_choice_wins(self):
        repo = _repo({STICKY_ROWS_KEY: ["exposure.density"]})
        self.assertEqual([r.id for r in load_sticky_rows(repo)], ["exposure.density"])

    def test_empty_choice_means_nothing_carries(self):
        repo = _repo({STICKY_ROWS_KEY: []})
        self.assertEqual(load_sticky_rows(repo), [])

    def test_unknown_ids_are_dropped(self):
        repo = _repo({STICKY_ROWS_KEY: ["exposure.density", "gone.retired_field"]})
        self.assertEqual([r.id for r in load_sticky_rows(repo)], ["exposure.density"])

    def test_save_round_trips(self):
        repo = _repo()
        save_sticky_rows(repo, ["lab.saturation", "exposure.density"])
        self.assertEqual({r.id for r in load_sticky_rows(repo)}, {"lab.saturation", "exposure.density"})

    def test_snapshot_excludes_description_fields(self):
        """It carries on its own key so the last Description… confirm wins for the roll."""
        cfg = WorkspaceConfig()
        cfg = replace(cfg, metadata=replace(cfg.metadata, description_fields=("camera", "iso")))
        self.assertNotIn("description_fields", sticky_snapshot(cfg))

    def test_snapshot_covers_every_other_catalog_field(self):
        snapshot = sticky_snapshot(WorkspaceConfig())
        expected = {f for r in all_rows() for f in r.fields} - {"description_fields"}
        self.assertEqual(set(snapshot), expected)


class TestLegacyMigration(unittest.TestCase):
    def test_seeds_from_legacy_keys(self):
        repo = _repo(
            {
                "last_process_mode": "B&W Negative",
                "last_aspect_ratio": "1:1",
                "last_lab_config": {"saturation": 1.4},
            }
        )
        migrate_legacy(repo)
        snapshot = repo.store[STICKY_CONFIG_KEY]
        self.assertEqual(snapshot["process_mode"], "B&W Negative")
        self.assertEqual(snapshot["autocrop_ratio"], "1:1")
        self.assertEqual(snapshot["saturation"], 1.4)

    def test_true_black_maps_inverted(self):
        repo = _repo({"last_true_black": False})
        migrate_legacy(repo)
        self.assertTrue(repo.store[STICKY_CONFIG_KEY]["paper_black"])

    def test_explicit_paper_black_beats_legacy(self):
        repo = _repo({"last_true_black": False, "last_paper_black": False})
        migrate_legacy(repo)
        self.assertFalse(repo.store[STICKY_CONFIG_KEY]["paper_black"])

    def test_no_op_when_snapshot_exists(self):
        repo = _repo({STICKY_CONFIG_KEY: {"density": 2.0}, "last_process_mode": "B&W Negative"})
        migrate_legacy(repo)
        self.assertEqual(repo.store[STICKY_CONFIG_KEY], {"density": 2.0})

    def test_no_op_on_a_fresh_install(self):
        repo = _repo()
        migrate_legacy(repo)
        self.assertNotIn(STICKY_CONFIG_KEY, repo.store)

    def test_drops_keys_with_no_catalog_row(self):
        repo = _repo({"last_process_mode": "B&W Negative", "last_export_config": {"export_path": "/out"}})
        migrate_legacy(repo)
        self.assertNotIn("export_path", repo.store[STICKY_CONFIG_KEY])


class TestAutoCropStickyRowMigration(unittest.TestCase):
    def test_adds_auto_crop_to_a_saved_choice_once(self):
        repo = _repo({STICKY_ROWS_KEY: ["geometry.autocrop_mode"]})
        migrate_auto_crop_sticky_row(repo)
        self.assertEqual(set(repo.store[STICKY_ROWS_KEY]), {"geometry.autocrop_mode", "geometry.crop_from_auto"})

        save_sticky_rows(repo, ["geometry.autocrop_mode"])
        migrate_auto_crop_sticky_row(repo)
        self.assertEqual(repo.store[STICKY_ROWS_KEY], ["geometry.autocrop_mode"])

    def test_a_user_who_never_chose_gets_it_from_the_defaults(self):
        repo = _repo()
        migrate_auto_crop_sticky_row(repo)
        self.assertNotIn(STICKY_ROWS_KEY, repo.store)
        self.assertIn("geometry.crop_from_auto", {r.id for r in load_sticky_rows(repo)})


class TestCarriedChromaZeroMigration(unittest.TestCase):
    def test_sets_a_carried_chroma_of_zero_back_to_one_once(self):
        repo = _repo({STICKY_CONFIG_KEY: {"saturation": 0.0, "process_mode": "C41"}})
        migrate_carried_chroma_zero(repo)
        self.assertEqual(repo.store[STICKY_CONFIG_KEY], {"saturation": 1.0, "process_mode": "C41"})

        repo.store[STICKY_CONFIG_KEY] = {"saturation": 0.0}
        migrate_carried_chroma_zero(repo)
        self.assertEqual(repo.store[STICKY_CONFIG_KEY], {"saturation": 0.0})

    def test_leaves_any_other_chroma_and_a_fresh_install_alone(self):
        repo = _repo({STICKY_CONFIG_KEY: {"saturation": 0.4}})
        migrate_carried_chroma_zero(repo)
        self.assertEqual(repo.store[STICKY_CONFIG_KEY], {"saturation": 0.4})
        fresh = _repo()
        migrate_carried_chroma_zero(fresh)
        self.assertNotIn(STICKY_CONFIG_KEY, fresh.store)


class TestExportDestinationMigration(unittest.TestCase):
    def _legacy_path(self):
        return os.path.join(paths.get_default_user_dir(), "export")

    def test_drops_an_untouched_legacy_destination(self):
        repo = _repo(
            {
                "last_export_config": {
                    "output_mode": "absolute",
                    "export_path": self._legacy_path(),
                    "output_subfolder": "export",
                    "jpeg_quality": 90,
                }
            }
        )
        migrate_legacy_export_destination(repo)
        remaining = repo.store["last_export_config"]
        self.assertNotIn("output_mode", remaining)
        self.assertNotIn("export_path", remaining)
        self.assertNotIn("output_subfolder", remaining)
        self.assertEqual(remaining["jpeg_quality"], 90)

    def test_leaves_a_chosen_absolute_path_alone(self):
        repo = _repo({"last_export_config": {"output_mode": "absolute", "export_path": "/Volumes/Scans/export"}})
        migrate_legacy_export_destination(repo)
        self.assertEqual(repo.store["last_export_config"]["export_path"], "/Volumes/Scans/export")

    def test_leaves_a_non_absolute_mode_alone(self):
        repo = _repo({"last_export_config": {"output_mode": "subfolder_of_source", "output_subfolder": "scans"}})
        migrate_legacy_export_destination(repo)
        self.assertEqual(repo.store["last_export_config"]["output_subfolder"], "scans")

    def test_runs_once(self):
        repo = _repo({"last_export_config": {"output_mode": "absolute", "export_path": self._legacy_path()}})
        migrate_legacy_export_destination(repo)
        repo.store["last_export_config"] = {"output_mode": "absolute", "export_path": self._legacy_path()}
        migrate_legacy_export_destination(repo)
        self.assertIn("output_mode", repo.store["last_export_config"])

    def test_no_op_on_a_fresh_install(self):
        repo = _repo()
        migrate_legacy_export_destination(repo)
        self.assertNotIn("last_export_config", repo.store)


class TestAlwaysSticky(unittest.TestCase):
    def test_scan_prefs_stay_out_of_the_replace_look_set(self):
        """Preset "Replace Look" must not reset the decode mode, or it forces a re-decode."""
        from negpy.desktop.view.sidebar.presets import _LOOK_ROWS

        look_fields = {f for r in _LOOK_ROWS for f in r.fields}
        for _key, field in ALWAYS_STICKY_PROCESS:
            self.assertNotIn(field, look_fields)


if __name__ == "__main__":
    unittest.main()


def test_load_sticky_config_parses_a_stored_snapshot_once():
    """Every frame hydration reads the sticky snapshot; a roll hydrates hundreds of frames."""
    from unittest.mock import MagicMock, patch

    from negpy.desktop import sticky
    from negpy.domain.models import WorkspaceConfig

    repo = MagicMock()
    repo.get_global_setting.side_effect = lambda key, default=None: {"process_mode": "C41", "density": 1.0}
    with patch.object(WorkspaceConfig, "from_flat_dict", wraps=WorkspaceConfig.from_flat_dict) as parse:
        first = sticky.load_sticky_config(repo)
        assert sticky.load_sticky_config(repo) is first
        assert parse.call_count == 1

        repo.get_global_setting.side_effect = lambda key, default=None: {"process_mode": "C41", "density": 1.2}
        assert sticky.load_sticky_config(repo).exposure.density == 1.2
        assert parse.call_count == 2
