import tempfile
import unittest
from dataclasses import replace

from negpy.desktop.session import DesktopSessionManager
from negpy.infrastructure.storage.repository import StorageRepository


class TestUndoAcrossFrameSwitch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        repo = StorageRepository(f"{self._tmp.name}/edits.db", f"{self._tmp.name}/settings.db")
        repo.initialize()
        self.session = DesktopSessionManager(repo)
        self.session.state.uploaded_files = [
            {"name": "a.tif", "path": f"{self._tmp.name}/a.tif", "hash": "ha"},
            {"name": "b.tif", "path": f"{self._tmp.name}/b.tif", "hash": "hb"},
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def _edit(self, density: float) -> None:
        cfg = self.session.state.config
        self.session.update_config(replace(cfg, exposure=replace(cfg.exposure, density=density)), persist=True)

    def _density(self) -> float:
        return self.session.state.config.exposure.density

    def test_one_edit_survives_a_frame_switch_as_an_undo_step(self):
        self.session.select_file(0)
        c0 = self._density()
        self._edit(0.4)
        self.session.select_file(1)
        self.session.select_file(0)

        self.session.undo()
        self.assertEqual(self._density(), c0)

    def test_two_edits_undo_one_step_at_a_time_after_a_switch(self):
        self.session.select_file(0)
        c0 = self._density()
        self._edit(0.4)
        self._edit(0.8)
        self.session.select_file(1)
        self.session.select_file(0)

        self.assertEqual(self._density(), 0.8)
        self.session.undo()
        self.assertEqual(self._density(), 0.4)
        self.session.undo()
        self.assertEqual(self._density(), c0)
        self.session.redo()
        self.session.redo()
        self.assertEqual(self._density(), 0.8)
