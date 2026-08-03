import tempfile
import unittest
from dataclasses import replace

from negpy.desktop.session import DesktopSessionManager
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository


def _variant(density: float) -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(base, exposure=replace(base.exposure, density=density))


class TestWorkPrintStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = StorageRepository(f"{self._tmp.name}/edits.db", f"{self._tmp.name}/settings.db")
        self.repo.initialize()

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        cfg = _variant(1.4)
        self.repo.save_work_print("h", "Work print 1", cfg)
        self.assertEqual(self.repo.list_work_prints("h"), ["Work print 1"])
        self.assertEqual(self.repo.load_work_print("h", "Work print 1"), cfg)

    def test_listed_newest_first(self):
        self.repo.save_work_print("h", "first", _variant(1.0))
        self.repo.save_work_print("h", "second", _variant(1.1))
        self.assertEqual(self.repo.list_work_prints("h"), ["second", "first"])

    def test_scoped_to_the_frame(self):
        self.repo.save_work_print("h", "mine", _variant(1.0))
        self.assertEqual(self.repo.list_work_prints("other"), [])
        self.assertIsNone(self.repo.load_work_print("other", "mine"))

    def test_same_name_replaces(self):
        self.repo.save_work_print("h", "one", _variant(1.0))
        self.repo.save_work_print("h", "one", _variant(1.9))
        self.assertEqual(self.repo.list_work_prints("h"), ["one"])
        self.assertEqual(self.repo.load_work_print("h", "one").exposure.density, 1.9)

    def test_rename_keeps_the_settings(self):
        cfg = _variant(1.4)
        self.repo.save_work_print("h", "old", cfg)
        self.repo.rename_work_print("h", "old", "new")
        self.assertEqual(self.repo.list_work_prints("h"), ["new"])
        self.assertEqual(self.repo.load_work_print("h", "new"), cfg)

    def test_delete(self):
        self.repo.save_work_print("h", "one", _variant(1.0))
        self.repo.delete_work_print("h", "one")
        self.assertEqual(self.repo.list_work_prints("h"), [])

    def test_missing_name_loads_none(self):
        self.assertIsNone(self.repo.load_work_print("h", "nothing"))

    def test_rehome_carries_work_prints_to_the_new_hash(self):
        self.repo.save_file_settings("old", _variant(1.0), file_path="/frame.raw")
        self.repo.save_work_print("old", "one", _variant(1.4))
        self.repo.rehome_file_settings("old", "new", "/frame.raw")
        self.assertEqual(self.repo.list_work_prints("new"), ["one"])
        self.assertEqual(self.repo.list_work_prints("old"), [])

    def test_clearing_saved_edits_drops_work_prints(self):
        self.repo.save_work_print("h", "one", _variant(1.0))
        self.repo.clear_saved_edits()
        self.assertEqual(self.repo.list_work_prints("h"), [])

    def test_reset_everything_drops_work_prints(self):
        self.repo.save_work_print("h", "one", _variant(1.0))
        self.repo.reset_everything()
        self.assertEqual(self.repo.list_work_prints("h"), [])

    def test_counted_in_the_database_stats(self):
        self.repo.save_work_print("h", "one", _variant(1.0))
        self.assertEqual(self.repo.database_stats()["work_prints"], 1)

    def test_survives_history_pruning(self):
        """The whole point: an undo stack that rolls over must not take versions with it."""
        self.repo.save_work_print("h", "kept", _variant(1.4))
        for i in range(20):
            self.repo.save_history_step("h", i, _variant(1.0 + i / 100.0))
        self.repo.prune_history("h", max_steps=10)
        self.repo.truncate_history_above("h", 2)
        self.assertEqual(self.repo.list_work_prints("h"), ["kept"])


class TestWorkPrintController(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = StorageRepository(f"{self._tmp.name}/edits.db", f"{self._tmp.name}/settings.db")
        self.repo.initialize()
        self.session = DesktopSessionManager(self.repo)
        self.session.state.current_file_hash = "h"
        self.session.state.config = _variant(1.4)

    def tearDown(self):
        self._tmp.cleanup()

    def test_saving_names_the_next_work_print(self):
        self.assertEqual(self.session.next_work_print_name(), "Work print 1")
        self.session.save_work_print("Work print 1")
        self.assertEqual(self.session.next_work_print_name(), "Work print 2")

    def test_saving_stores_the_live_edit(self):
        self.session.save_work_print("wp")
        self.assertEqual(self.repo.load_work_print("h", "wp"), _variant(1.4))

    def test_loading_a_work_print_is_undoable(self):
        self.session.save_work_print("wp")
        self.session.update_config(_variant(0.6), persist=True)
        self.session.load_work_print("wp")
        self.assertEqual(self.session.state.config, _variant(1.4))
        self.session.undo()
        self.assertEqual(self.session.state.config, _variant(0.6))

    def test_clicking_an_unchanged_work_print_adds_no_history_step(self):
        self.session.save_work_print("wp")
        index = self.session.state.undo_index
        self.session.load_work_print("wp")
        self.session.load_work_print("wp")
        self.assertEqual(self.session.state.undo_index, index)
        self.assertEqual(self.session.state.max_history_index, index)

    def test_loading_a_missing_work_print_changes_nothing(self):
        self.session.load_work_print("nothing")
        self.assertEqual(self.session.state.config, _variant(1.4))

    def test_no_frame_no_work_prints(self):
        self.session.state.current_file_hash = ""
        self.session.save_work_print("wp")
        self.assertEqual(self.session.work_prints(), [])


if __name__ == "__main__":
    unittest.main()
