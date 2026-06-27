import tempfile
import unittest
from dataclasses import replace

from negpy.desktop.controller import history_step_label
from negpy.desktop.session import DesktopSessionManager
from negpy.domain.models import LabConfig, WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository


def _exposure_variant() -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(base, exposure=replace(base.exposure, auto_exposure=not base.exposure.auto_exposure))


class TestHistoryStepLabel(unittest.TestCase):
    def test_base_when_no_previous(self):
        self.assertEqual(history_step_label(None, WorkspaceConfig(), 0), "0 · base")

    def test_names_changed_section(self):
        prev = WorkspaceConfig()
        label = history_step_label(prev, _exposure_variant(), 3)
        self.assertIn("exposure", label)
        self.assertTrue(label.startswith("3 · "))

    def test_unchanged_marker(self):
        cfg = WorkspaceConfig()
        self.assertEqual(history_step_label(cfg, cfg, 2), "2 · —")

    def test_only_changed_section_listed(self):
        prev = WorkspaceConfig()
        cfg = replace(prev, lab=replace(prev.lab, **_first_diff_field(prev.lab, LabConfig)))
        label = history_step_label(prev, cfg, 1)
        self.assertIn("lab", label)
        self.assertNotIn("exposure", label)


def _first_diff_field(current, cls):
    """Flip one field of a config so it differs from its default."""
    from dataclasses import fields

    f = fields(cls)[0]
    val = getattr(current, f.name)
    new = (not val) if isinstance(val, bool) else (val + 1 if isinstance(val, (int, float)) else val)
    return {f.name: new}


class TestJumpToStep(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = StorageRepository(f"{self._tmp.name}/edits.db", f"{self._tmp.name}/settings.db")
        self.repo.initialize()
        self.session = DesktopSessionManager(self.repo)
        self.session.state.current_file_hash = "h"

        self.cfg_a = _exposure_variant()
        self.cfg_b = replace(self.cfg_a, lab=replace(self.cfg_a.lab, **_first_diff_field(self.cfg_a.lab, LabConfig)))
        self.session.update_config(self.cfg_a, persist=True)
        self.session.update_config(self.cfg_b, persist=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_history_recorded(self):
        self.assertEqual(self.session.state.undo_index, 2)
        self.assertEqual(self.session.state.max_history_index, 2)

    def test_jump_back_loads_old_config(self):
        self.session.jump_to_step(0)
        self.assertEqual(self.session.state.undo_index, 0)
        self.assertEqual(self.session.state.config, WorkspaceConfig())

    def test_jump_from_top_persists_live_state(self):
        # Jumping away from the unsaved top must keep it reachable.
        self.session.jump_to_step(0)
        self.session.jump_to_step(2)
        self.assertEqual(self.session.state.undo_index, 2)
        self.assertEqual(self.session.state.config, self.cfg_b)

    def test_jump_is_noop_on_current(self):
        self.session.jump_to_step(2)
        self.assertEqual(self.session.state.config, self.cfg_b)

    def test_edit_after_jump_truncates_future(self):
        self.session.jump_to_step(0)
        self.session.update_config(_exposure_variant(), persist=True)
        self.assertEqual(self.session.state.max_history_index, 1)
        indices = [i for i, _ in self.repo.load_all_history("h")]
        self.assertNotIn(2, indices)


if __name__ == "__main__":
    unittest.main()
