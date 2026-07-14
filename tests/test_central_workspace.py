"""UI contracts for the central whole-roll contact sheet."""

from types import SimpleNamespace

from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.widgets.central_workspace import CentralWorkspaceStack
from negpy.desktop.view.widgets.roll_slot_model import RollPreviewSlot
from negpy.desktop.view.widgets.roll_slot_selector import RollSlotSelector


def test_central_workspace_reuses_selector_and_preserves_its_state(qapp) -> None:
    old_parent = QWidget()
    selector = RollSlotSelector(old_parent)
    selector.set_slots([RollPreviewSlot(1), RollPreviewSlot(2), RollPreviewSlot(3)])
    selector.set_scan_material(selector.scan_material())
    selector.set_selected_slot_ids([1, 3])

    canvas = QWidget()
    workspace = CentralWorkspaceStack(canvas)
    workspace.mount_roll_selector(selector)

    assert workspace.roll_selector is selector
    assert selector.parentWidget() is workspace.roll_preview_page
    assert selector.selected_slot_ids() == [1, 3]
    assert workspace.currentWidget() is canvas

    workspace.show_roll_preview()

    assert workspace.is_showing_roll_preview()
    assert selector.isVisibleTo(workspace)
    assert selector.selected_slot_ids() == [1, 3]


def test_returning_to_canvas_does_not_discard_loaded_preview(qapp) -> None:
    canvas = QWidget()
    selector = RollSlotSelector()
    selector.set_slots([RollPreviewSlot(1), RollPreviewSlot(2)])
    selector.set_selected_slot_ids([2])
    workspace = CentralWorkspaceStack(canvas)
    workspace.mount_roll_selector(selector)
    workspace.show_roll_preview()

    workspace.show_canvas()

    assert workspace.currentWidget() is canvas
    assert not workspace.is_showing_roll_preview()
    assert selector.model.rowCount() == 2
    assert selector.selected_slot_ids() == [2]


def test_show_roll_preview_is_safe_before_a_selector_is_mounted(qapp) -> None:
    canvas = QWidget()
    workspace = CentralWorkspaceStack(canvas)

    workspace.show_roll_preview()

    assert workspace.currentWidget() is canvas


def test_loaded_preview_event_requires_a_populated_shared_model(qapp) -> None:
    canvas = QWidget()
    selector = RollSlotSelector()
    workspace = CentralWorkspaceStack(canvas)
    workspace.mount_roll_selector(selector)

    workspace.show_loaded_roll_preview("token", object())
    assert workspace.currentWidget() is canvas

    selector.set_slots([RollPreviewSlot(1)])
    workspace.show_loaded_roll_preview("token", object())
    assert workspace.is_showing_roll_preview()


def test_roll_completion_returns_only_after_a_frame_was_produced(qapp) -> None:
    canvas = QWidget()
    selector = RollSlotSelector()
    selector.set_slots([RollPreviewSlot(1)])
    workspace = CentralWorkspaceStack(canvas)
    workspace.mount_roll_selector(selector)
    workspace.show_roll_preview()

    workspace.show_canvas_after_roll_scan(SimpleNamespace(rgb_paths=()))
    assert workspace.is_showing_roll_preview()

    workspace.show_canvas_after_roll_scan(SimpleNamespace(rgb_paths=("frame001.tif",)))
    assert workspace.currentWidget() is canvas


def test_only_recovery_required_error_leaves_contact_sheet(qapp) -> None:
    canvas = QWidget()
    selector = RollSlotSelector()
    selector.set_slots([RollPreviewSlot(1)])
    workspace = CentralWorkspaceStack(canvas)
    workspace.mount_roll_selector(selector)
    workspace.show_roll_preview()

    workspace.show_canvas_after_roll_error(SimpleNamespace(recovery_required=False))
    assert workspace.is_showing_roll_preview()

    workspace.show_canvas_after_roll_error(SimpleNamespace(recovery_required=True))
    assert workspace.currentWidget() is canvas
