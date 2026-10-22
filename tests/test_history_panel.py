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


def _exposure(**changes) -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(base, exposure=replace(base.exposure, **changes))


class TestHistoryStepLabel(unittest.TestCase):
    def test_the_first_step_is_as_opened(self):
        self.assertEqual(history_step_label(None, WorkspaceConfig()), ("As opened", ""))

    def test_one_control_names_its_card_and_the_control(self):
        self.assertEqual(history_step_label(WorkspaceConfig(), _exposure_variant()), ("Tone", "Auto Density"))

    def test_controls_in_one_card_are_listed_after_it(self):
        self.assertEqual(history_step_label(WorkspaceConfig(), _exposure(wb_magenta=0.1, wb_yellow=0.1)), ("Filtration", "Magenta, Yellow"))

    def test_more_than_two_are_counted(self):
        label = history_step_label(WorkspaceConfig(), _exposure(wb_cyan=0.1, wb_magenta=0.1, wb_yellow=0.1))
        self.assertEqual(label, ("Filtration", "Cyan, Magenta +1"))

    def test_steps_across_cards_name_the_cards(self):
        prev = WorkspaceConfig()
        cfg = replace(_exposure(wb_magenta=0.1), lab=replace(prev.lab, sharpen=prev.lab.sharpen + 0.5))
        self.assertEqual(history_step_label(prev, cfg), ("Filtration, Lab", ""))

    def test_a_change_no_catalog_row_lists_names_its_card(self):
        prev = WorkspaceConfig()
        cfg = replace(prev, geometry=replace(prev.geometry, crop_to_valid=not prev.geometry.crop_to_valid))
        self.assertEqual(history_step_label(prev, cfg), ("Crop", ""))

    def test_bounds_a_render_measured_are_not_a_change(self):
        prev = WorkspaceConfig()
        cfg = replace(_exposure(wb_magenta=0.1), process=replace(prev.process, local_floors=(0.1, 0.1, 0.1), meter_key="k"))
        self.assertEqual(history_step_label(prev, cfg), ("Filtration", "Magenta"))

    def test_unchanged_marker(self):
        cfg = WorkspaceConfig()
        self.assertEqual(history_step_label(cfg, cfg), ("—", ""))


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
        self.start_cfg = self.session.state.config

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
        self.assertEqual(self.session.state.config, self.start_cfg)

    def test_jump_from_top_persists_live_state(self):
        # Jumping away from the unsaved top must keep it reachable.
        self.session.jump_to_step(0)
        self.session.jump_to_step(2)
        self.assertEqual(self.session.state.undo_index, 2)
        self.assertEqual(self.session.state.config, self.cfg_b)

    def test_jump_is_noop_on_current(self):
        self.session.jump_to_step(2)
        self.assertEqual(self.session.state.config, self.cfg_b)

    def test_persisting_an_unchanged_config_records_no_step(self):
        """A step whose config equals the one before it renders as a dead '· —' row."""
        self.session.update_config(self.cfg_b, persist=True)
        self.assertEqual(self.session.state.undo_index, 2)
        self.assertEqual(self.session.state.max_history_index, 2)

    def test_persisting_an_unchanged_config_keeps_the_redo_branch(self):
        self.session.jump_to_step(1)
        self.session.update_config(self.session.state.config, persist=True)
        self.assertEqual(self.session.state.max_history_index, 2)
        self.assertIn(2, [i for i, _ in self.repo.load_all_history("h")])

    def test_edit_after_jump_truncates_future(self):
        self.session.jump_to_step(0)
        self.session.update_config(_exposure_variant(), persist=True)
        self.assertEqual(self.session.state.max_history_index, 1)
        indices = [i for i, _ in self.repo.load_all_history("h")]
        self.assertNotIn(2, indices)


if __name__ == "__main__":
    unittest.main()


_ROWS = [
    {"index": 0, "name": "As opened", "detail": "", "is_current": False, "undone": False},
    {"index": 1, "name": "Tone", "detail": "Print Density", "is_current": True, "undone": False},
    {"index": 2, "name": "Filtration", "detail": "", "is_current": False, "undone": True},
]


def _history_panel(prints=()):
    from unittest.mock import MagicMock

    from negpy.desktop.session import AppState
    from negpy.desktop.view.sidebar.history import HistoryPanel

    controller = MagicMock()
    controller.state = AppState()
    controller.state.current_file_hash = "h"
    controller.state.undo_index, controller.state.max_history_index = 1, 2
    controller.history_steps.return_value = _ROWS
    controller.session.work_prints.return_value = list(prints)
    controller.session.work_print_apply_metadata.return_value = False
    return controller, HistoryPanel(controller)


def test_rows_mark_the_current_and_the_undone_steps_newest_first(qapp):
    from negpy.desktop.view.sidebar.history import _TAG_ROLE

    _controller, panel = _history_panel()
    rows = [(panel.list.item(i).text(), panel.list.item(i).data(_TAG_ROLE)) for i in range(panel.list.count())]
    assert rows == [("Filtration", "undone"), ("Tone", "now"), ("As opened", "")]
    assert panel.undo_btn.isEnabled() and panel.redo_btn.isEnabled()


def test_with_no_work_prints_the_card_says_what_they_are(qapp):
    _controller, panel = _history_panel()
    assert panel.work_prints.isHidden() and panel.apply_metadata_btn.isHidden()
    assert not panel.work_prints_hint.isHidden()

    _controller, panel = _history_panel(prints=["Soft proof"])
    assert not panel.work_prints.isHidden() and not panel.apply_metadata_btn.isHidden()
    assert panel.work_prints_hint.isHidden()


def test_a_press_on_a_rows_more_button_opens_its_menu_and_leaves_the_step(qapp):
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    from negpy.desktop.view.widgets.flat_rows import more_rect

    controller, panel = _history_panel()
    panel.resize(420, 600)
    panel.show()
    qapp.processEvents()
    asked = []
    panel.list.menu_requested.disconnect()
    panel.list.menu_requested.connect(asked.append)
    point = QPointF(more_rect(panel.list.visualItemRect(panel.list.item(0))).center())
    for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
        event = QMouseEvent(
            kind,
            point,
            panel.list.viewport().mapToGlobal(point),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        qapp.sendEvent(panel.list.viewport(), event)

    assert len(asked) == 1
    controller.jump_to_history_step.assert_not_called()
