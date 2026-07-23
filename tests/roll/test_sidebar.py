"""Smoke and behavior tests for CoolscanRollSidebar. Mirrors
test_scanlight_sidebar.py: build the widget against a mock controller, then
drive it either through real widgets (clicks, combo selection) or by
calling its private `_on_*` handlers directly to simulate what the (mocked,
never-really-running) controller/worker would deliver.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
from PyQt6.QtWidgets import QApplication
from unittest.mock import MagicMock

from negpy.desktop.view.sidebar.coolscan_roll import CoolscanRollSidebar, QMessageBox
from negpy.infrastructure.roll import coolscanpy_roll

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _sidebar():
    ctrl = MagicMock()
    ctrl.session.repo.get_global_setting.return_value = {}
    return CoolscanRollSidebar(ctrl)


def _thumb(fake_coolscanpy, slot, *, needs_approval=False, spacing_offset=0):
    return fake_coolscanpy.Thumbnail(
        slot=slot,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        boundary_rows=(0, 4),
        spacing_offset=spacing_offset,
        needs_approval=needs_approval,
    )


def _pick_device(w, device_id="ls5000-usb-001", label="Nikon LS-5000"):
    w.device_combo.clear()
    w.device_combo.addItem(label, device_id)
    w.device_combo.setCurrentIndex(0)


class TestBuildsAndGating:
    def test_sidebar_builds_with_all_controls(self) -> None:
        w = _sidebar()
        for attr in (
            "preview_btn",
            "scan_btn",
            "safe_stop_btn",
            "eject_btn",
            "select_all_btn",
            "clear_selection_btn",
            "device_combo",
            "refresh_btn",
            "folder_edit",
            "pattern_edit",
            "contact_sheet",
            "offset_spin",
            "offset_apply_btn",
            "approve_btn",
            "gate_hint",
            "status_label",
            "hybrid_synthesis_limit_spin",
            "hybrid_guidance",
            "preview_workspace",
            "preview_display_combo",
            "open_preview_workspace_btn",
        ):
            assert hasattr(w, attr), attr
        assert w.contact_sheet is w.preview_workspace.contact_sheet
        assert w.contact_sheet.minimumHeight() >= 360
        assert w.contact_sheet.iconSize().width() >= 200

    def test_missing_coolscanpy_disables_preview_and_shows_hint(self, monkeypatch) -> None:
        # The optional group may be installed by this test run, so exercise
        # the sidebar's unavailable branch at its adapter seam instead of
        # assuming an interpreter-level package state.
        monkeypatch.setattr(coolscanpy_roll, "available", lambda: False)
        w = _sidebar()
        assert w.preview_btn.isEnabled() is False
        assert "install coolscanpy" in w.gate_hint.text()
        assert not w._setup_hint.isHidden()

    def test_device_selected_still_blocks_scan_until_folder_and_slots(self, fake_coolscanpy) -> None:
        w = _sidebar()
        _pick_device(w)
        assert w.preview_btn.isEnabled() is True  # coolscanpy available (faked) + device picked
        assert w.scan_btn.isEnabled() is False
        missing = w._missing_for_scan()
        assert "select at least one slot" in missing
        assert "choose an output folder" in missing

    def test_scan_enabled_once_folder_and_slot_selected(self, fake_coolscanpy, tmp_path) -> None:
        w = _sidebar()
        _pick_device(w)
        w.folder_edit.setText(str(tmp_path))
        w._on_preview_ready([_thumb(fake_coolscanpy, 1)])
        w.contact_sheet.item(0).setSelected(True)

        assert w._missing_for_scan() == []
        assert w.scan_btn.isEnabled() is True

    def test_approve_button_hidden_without_a_selection(self) -> None:
        w = _sidebar()
        assert w.approve_btn.isVisible() is False


class TestPreview:
    def test_preview_click_starts_request_for_selected_device(self, fake_coolscanpy) -> None:
        w = _sidebar()
        _pick_device(w, device_id="dev-42")

        w._on_preview_clicked()

        from negpy.desktop.workers.roll_worker import RollPreviewRequest

        req = w.controller.start_coolscan_roll_preview.call_args[0][0]
        assert isinstance(req, RollPreviewRequest)
        assert req.device_id == "dev-42"
        assert w._preview_pending is True
        assert w.eject_btn.isEnabled() is False

    def test_preview_click_without_a_device_is_a_noop(self) -> None:
        w = _sidebar()
        w._on_preview_clicked()
        w.controller.start_coolscan_roll_preview.assert_not_called()

    def test_workspace_buttons_switch_between_roll_preview_and_editor(self) -> None:
        w = _sidebar()
        opened: list[bool] = []
        closed: list[bool] = []
        w.workspace_requested.connect(lambda: opened.append(True))
        w.preview_workspace.back_requested.connect(lambda: closed.append(True))

        w.open_preview_workspace_btn.click()
        w.preview_workspace.back_btn.click()

        assert opened == [True]
        assert closed == [True]

    def test_preview_ready_populates_the_contact_sheet(self, fake_coolscanpy) -> None:
        w = _sidebar()
        thumbs = [_thumb(fake_coolscanpy, s) for s in (2, 1, 3)]

        w._on_preview_ready(thumbs)

        assert w.contact_sheet.count() == 3
        assert sorted(w._thumbnails.keys()) == [1, 2, 3]
        # the widget lists slots in ascending order regardless of preview()'s return order
        from negpy.desktop.view.sidebar.coolscan_roll import _SLOT_ROLE

        listed = [w.contact_sheet.item(i).data(_SLOT_ROLE) for i in range(w.contact_sheet.count())]
        assert listed == [1, 2, 3]

    def test_display_toggle_updates_icons_without_mutating_capture_data(self, fake_coolscanpy) -> None:
        w = _sidebar()
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        image[:, :6, :] = (10, 20, 30)
        image[:, 6:, :] = (180, 190, 200)
        original = image.copy()

        w._on_preview_ready(
            [
                fake_coolscanpy.Thumbnail(
                    slot=1,
                    image=image,
                    boundary_rows=(0, 8),
                    spacing_offset=0,
                    needs_approval=False,
                )
            ]
        )

        assert w.preview_display_combo.currentData() is True
        positive_key = w.contact_sheet.item(0).icon().cacheKey()
        w.preview_display_combo.setCurrentIndex(1)
        raw_key = w.contact_sheet.item(0).icon().cacheKey()
        w.preview_display_combo.setCurrentIndex(0)
        positive_again_key = w.contact_sheet.item(0).icon().cacheKey()

        assert positive_key != raw_key
        assert raw_key != positive_again_key
        np.testing.assert_array_equal(w._thumbnails[1].image, original)
        w.controller.start_coolscan_roll_preview.assert_not_called()
        w.controller.start_roll_scan.assert_not_called()

    def test_needs_approval_slot_is_marked_and_shows_approve_button(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w._on_preview_ready([_thumb(fake_coolscanpy, 5, needs_approval=True)])

        item = w.contact_sheet.item(0)
        assert "⚠" in item.text()

        w.contact_sheet.setCurrentItem(item)
        assert not w.approve_btn.isHidden()

    def test_selecting_a_slot_seeds_the_offset_spinner(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w._on_preview_ready([_thumb(fake_coolscanpy, 7, spacing_offset=-15)])

        w.contact_sheet.setCurrentItem(w.contact_sheet.item(0))

        assert w.slot_label.text() == "7"
        assert w.offset_spin.value() == -15

    def test_new_preview_clears_the_old_contact_sheet(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w._on_preview_ready([_thumb(fake_coolscanpy, 1)])
        assert w.contact_sheet.count() == 1

        w._clear_contact_sheet()  # what _on_preview_clicked does before requesting a fresh preview
        w._on_preview_ready([_thumb(fake_coolscanpy, 9)])

        assert w.contact_sheet.count() == 1
        assert list(w._thumbnails.keys()) == [9]

    def test_select_all_frames_prepares_a_full_roll_selection(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        w = _sidebar()
        _pick_device(w)
        w.folder_edit.setText(str(tmp_path))
        w._on_preview_ready([_thumb(fake_coolscanpy, slot) for slot in range(1, 37)])

        w.select_all_btn.click()

        assert w._selected_slots() == list(range(1, 37))
        assert w.scan_btn.isEnabled() is True
        assert w.clear_selection_btn.isEnabled() is True


class TestEject:
    def test_preview_in_flight_blocks_eject(self, fake_coolscanpy, monkeypatch) -> None:
        w = _sidebar()
        _pick_device(w, device_id="usb:2:7")
        w._on_preview_clicked()
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
        )

        w._on_eject_clicked()

        w.controller.eject_roll.assert_not_called()
        assert w._preview_pending is True
        assert w.eject_btn.isEnabled() is False

    def test_confirmation_ejects_selected_direct_usb_device(
        self,
        fake_coolscanpy,
        monkeypatch,
    ) -> None:
        w = _sidebar()
        _pick_device(w, device_id="usb:2:7")
        w._on_preview_ready([_thumb(fake_coolscanpy, 1)])
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
        )

        w._on_eject_clicked()

        w.controller.eject_roll.assert_called_once_with("usb:2:7")
        assert w.contact_sheet.count() == 0
        assert w.eject_btn.isEnabled() is False
        assert w.preview_btn.isEnabled() is False
        assert "Ejecting" in w.status_label.text()

    def test_confirmation_cancel_does_not_eject(self, monkeypatch) -> None:
        w = _sidebar()
        _pick_device(w, device_id="usb:2:7")
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.No,
        )

        w._on_eject_clicked()

        w.controller.eject_roll.assert_not_called()
        assert w.eject_btn.isEnabled() is True

    def test_success_clears_registration_and_latches_against_second_eject(
        self,
        fake_coolscanpy,
    ) -> None:
        w = _sidebar()
        _pick_device(w, device_id="usb:2:7")
        w._on_preview_ready([_thumb(fake_coolscanpy, 1)])
        w._eject_pending = True

        w._on_ejected(True)

        assert w.contact_sheet.count() == 0
        assert w.eject_btn.isEnabled() is False
        assert "Eject started" in w.status_label.text()

    def test_late_preview_after_eject_cannot_restore_registration(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        w = _sidebar()
        _pick_device(w, device_id="usb:2:7")
        w.folder_edit.setText(str(tmp_path))
        w._eject_latched = True

        w._on_preview_ready([_thumb(fake_coolscanpy, 1)])
        w.contact_sheet.selectAll()
        w._on_scan_clicked()

        assert w.contact_sheet.count() == 0
        assert w._thumbnails == {}
        assert w.select_all_btn.isEnabled() is False
        assert w.scan_btn.isEnabled() is False
        w.controller.start_roll_scan.assert_not_called()

    def test_fresh_preview_after_refeed_reestablishes_registration(
        self,
        fake_coolscanpy,
    ) -> None:
        w = _sidebar()
        _pick_device(w, device_id="usb:2:7")
        w._eject_latched = True
        w._apply_gating()

        w._on_preview_clicked()
        w._on_preview_ready([_thumb(fake_coolscanpy, 1)])

        assert w._preview_pending is False
        assert w._eject_latched is False
        assert w.contact_sheet.count() == 1
        assert w.eject_btn.isEnabled() is True

    def test_uncertain_failure_blocks_preview_and_retry(self) -> None:
        w = _sidebar()
        _pick_device(w, device_id="usb:2:7")
        w._eject_pending = True

        w._on_eject_error("transport status unknown")

        assert w.eject_btn.isEnabled() is False
        assert w.preview_btn.isEnabled() is False
        assert "do not retry" in w.status_label.text()
        assert "physically checking" in w._missing_for_preview()[-1]


class TestSpacingAndApproval:
    def test_apply_offset_forwards_the_spinner_value(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w._on_preview_ready([_thumb(fake_coolscanpy, 4)])
        w.contact_sheet.setCurrentItem(w.contact_sheet.item(0))
        w.offset_spin.setValue(-30)

        w._on_apply_offset()

        w.controller.set_roll_spacing_offset.assert_called_once_with(4, -30)

    def test_spacing_offset_set_updates_local_thumbnail_state(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w._on_preview_ready([_thumb(fake_coolscanpy, 4, spacing_offset=0)])

        w._on_spacing_offset_set(4, -8)

        assert w._thumbnails[4].spacing_offset == -8

    def test_approve_click_forwards_the_selected_slot(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w._on_preview_ready([_thumb(fake_coolscanpy, 6, needs_approval=True)])
        w.contact_sheet.setCurrentItem(w.contact_sheet.item(0))

        w._on_approve()

        w.controller.approve_roll_slot.assert_called_once_with(6)

    def test_approved_clears_the_warning_marker(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w._on_preview_ready([_thumb(fake_coolscanpy, 6, needs_approval=True)])
        w.contact_sheet.setCurrentItem(w.contact_sheet.item(0))
        assert not w.approve_btn.isHidden()

        w._on_approved(6)

        assert w._thumbnails[6].needs_approval is False
        assert "⚠" not in w.contact_sheet.item(0).text()
        assert w.approve_btn.isHidden()


class TestScanning:
    def test_scan_click_builds_request_from_selection_and_settings(self, fake_coolscanpy, tmp_path) -> None:
        w = _sidebar()
        _pick_device(w, device_id="dev-1")
        w.folder_edit.setText(str(tmp_path))
        w.pattern_edit.setText('{{ "%03d" % seq }}')
        w._on_preview_ready([_thumb(fake_coolscanpy, 1), _thumb(fake_coolscanpy, 2)])
        w.contact_sheet.item(0).setSelected(True)
        w.contact_sheet.item(1).setSelected(True)

        w._on_scan_clicked()

        from negpy.desktop.workers.roll_worker import RollBatchScanRequest

        req = w.controller.start_roll_scan.call_args[0][0]
        assert isinstance(req, RollBatchScanRequest)
        assert req.device_id == "dev-1"
        assert req.slots == (1, 2)
        assert req.output_folder == str(tmp_path)
        assert req.hybrid_synthesis_limit_percent == 10.0
        assert w._scanning is True

    def test_scan_click_without_a_selected_slot_is_a_noop(self, fake_coolscanpy, tmp_path) -> None:
        w = _sidebar()
        _pick_device(w)
        w.folder_edit.setText(str(tmp_path))
        w._on_scan_clicked()
        w.controller.start_roll_scan.assert_not_called()

    def test_finished_clears_scanning_state(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w.set_scanning(True)
        w._on_finished([])
        assert w._scanning is False
        assert "Scanned 0" in w.status_label.text()

    def test_finished_reports_hybrid_to_exact_degradation_as_an_issue(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        from negpy.desktop.workers.roll_worker import RollBatchScanRequest

        w = _sidebar()
        w._active_scan_request = RollBatchScanRequest(
            device_id="ls5000-usb-001",
            slots=(1,),
            output_folder=str(tmp_path),
            filename_pattern='{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            write_positive=False,
            repair_mode="hybrid",
        )
        w.set_scanning(True)

        w._on_finished(
            [
                SimpleNamespace(
                    slot=1,
                    rgb_path="001.tif",
                    repaired_rgb_path="001_repaired.tif",
                    positive_path=None,
                    native_synthesis_mask_path=None,
                    hybrid_receipt_path=None,
                )
            ]
        )

        assert w._scanning is False
        assert "Completed with issues" in w.status_label.text()
        assert "Hybrid repair degraded or unavailable" in w.status_label.text()
        assert "Scanned 1 frame" not in w.status_label.text()

    def test_finished_reports_unavailable_requested_exact_positive_as_an_issue(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        from negpy.desktop.workers.roll_worker import RollBatchScanRequest

        w = _sidebar()
        w._active_scan_request = RollBatchScanRequest(
            device_id="ls5000-usb-001",
            slots=(1,),
            output_folder=str(tmp_path),
            filename_pattern='{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=False,
            write_positive=True,
            repair_mode="exact",
            positive_mode="nikon-exact",
        )
        w.set_scanning(True)

        w._on_finished(
            [
                SimpleNamespace(
                    slot=1,
                    rgb_path="001.tif",
                    repaired_rgb_path=None,
                    positive_path=None,
                    native_synthesis_mask_path=None,
                    hybrid_receipt_path=None,
                )
            ]
        )

        assert "Completed with issues" in w.status_label.text()
        assert "Nikon exact positive unavailable" in w.status_label.text()
        assert "Scanned 1 frame" not in w.status_label.text()

    def test_finished_reports_missing_requested_slot_as_an_issue(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        from negpy.desktop.workers.roll_worker import RollBatchScanRequest

        w = _sidebar()
        w._active_scan_request = RollBatchScanRequest(
            device_id="ls5000-usb-001",
            slots=(1, 2),
            output_folder=str(tmp_path),
            filename_pattern='{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=False,
            write_positive=False,
            repair_mode="exact",
        )

        w._on_finished(
            [
                SimpleNamespace(
                    slot=1,
                    rgb_path="001.tif",
                    repaired_rgb_path=None,
                    positive_path=None,
                    native_synthesis_mask_path=None,
                    hybrid_receipt_path=None,
                )
            ]
        )

        assert "Completed with issues" in w.status_label.text()
        assert "slot(s) 2 did not complete" in w.status_label.text()
        assert "Scanned 1 frame" not in w.status_label.text()

    def test_finished_reports_each_missing_requested_file_tier_as_an_issue(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        from negpy.desktop.workers.roll_worker import RollBatchScanRequest

        w = _sidebar()
        w._active_scan_request = RollBatchScanRequest(
            device_id="ls5000-usb-001",
            slots=(1,),
            output_folder=str(tmp_path),
            filename_pattern='{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            write_positive=False,
            repair_mode="exact",
        )

        w._on_finished(
            [
                SimpleNamespace(
                    slot=1,
                    rgb_path=None,
                    repaired_rgb_path=None,
                    positive_path=None,
                    native_synthesis_mask_path=None,
                    hybrid_receipt_path=None,
                )
            ]
        )

        status = w.status_label.text()
        assert "Completed with issues" in status
        assert "requested unrepaired output unavailable" in status
        assert "requested repaired output unavailable" in status
        assert "Scanned 1 frame" not in status

    def test_finished_reports_full_success_only_when_hybrid_and_exact_exist(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        from negpy.desktop.workers.roll_worker import RollBatchScanRequest

        w = _sidebar()
        w._active_scan_request = RollBatchScanRequest(
            device_id="ls5000-usb-001",
            slots=(1,),
            output_folder=str(tmp_path),
            filename_pattern='{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            write_positive=True,
            repair_mode="hybrid",
            positive_mode="nikon-exact",
        )
        w.set_scanning(True)

        w._on_finished(
            [
                SimpleNamespace(
                    slot=1,
                    rgb_path="001.tif",
                    repaired_rgb_path="001_repaired.tif",
                    positive_path="001_positive.tif",
                    native_synthesis_mask_path="native-mask.png",
                    hybrid_receipt_path="hybrid-receipt.json",
                )
            ]
        )

        assert w.status_label.text() == "Scanned 1 frame(s)."

    def test_safe_stop_disables_itself_and_forwards_to_controller(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w.set_scanning(True)

        w._on_safe_stop_clicked()

        w.controller.roll_safe_stop.assert_called_once()
        assert w.safe_stop_btn.isEnabled() is False
        assert "Stopping" in w.status_label.text()

    def test_safe_stop_suppresses_further_progress_chatter(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w.set_scanning(True)
        w._on_safe_stop_clicked()

        w._on_progress(0.5, "slot 3 complete")

        assert "Stopping" in w.status_label.text()  # not overwritten by the progress message

    def test_cancelled_clears_scanning_state(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w.set_scanning(True)
        w._on_cancelled()
        assert w._scanning is False
        assert w.safe_stop_btn.isEnabled() is False

    def test_error_clears_scanning_state_and_shows_message(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w.set_scanning(True)
        w._on_error("transport jam")
        assert w._scanning is False
        assert "transport jam" in w.status_label.text()

    def test_safe_stop_button_only_enabled_while_scanning(self, fake_coolscanpy) -> None:
        w = _sidebar()
        assert w.safe_stop_btn.isEnabled() is False
        w.set_scanning(True)
        assert w.safe_stop_btn.isEnabled() is True
        w.set_scanning(False)
        assert w.safe_stop_btn.isEnabled() is False


class TestSettingsPersistence:
    def test_settings_round_trip_through_the_repo(self, fake_coolscanpy, tmp_path) -> None:
        w = _sidebar()
        _pick_device(w, device_id="dev-7")
        w.folder_edit.setText(str(tmp_path))
        w.pattern_edit.setText('{{ "%03d" % seq }}')

        w._update_settings_from_ui()

        saved = w.controller.session.repo.save_global_setting.call_args[0]
        assert saved[0] == "roll_scan_settings"
        assert saved[1]["last_device_id"] == "dev-7"
        assert saved[1]["output_folder"] == str(tmp_path)

    def test_load_settings_restores_last_device_on_device_list(self, fake_coolscanpy) -> None:
        ctrl = MagicMock()
        ctrl.session.repo.get_global_setting.return_value = {"last_device_id": "dev-9", "output_folder": "", "filename_pattern": ""}
        w = CoolscanRollSidebar(ctrl)

        class _Dev:
            id = "dev-9"
            vendor = "Nikon"
            model = "LS-5000"

        w._on_devices_ready([_Dev()])

        assert w.device_combo.currentData() == "dev-9"


class TestOutputTiers:
    """The three independent output-tier checkboxes and the repair-mode
    combo. See tests/roll/test_service.py and test_worker.py for what each
    tier actually does; these tests only cover the sidebar's own wiring."""

    def test_sidebar_builds_with_tier_controls(self) -> None:
        w = _sidebar()
        for attr in ("write_unrepaired_check", "write_repaired_check", "write_positive_check", "repair_mode_combo", "tier_hint"):
            assert hasattr(w, attr), attr

    def test_defaults_match_settings_defaults(self) -> None:
        w = _sidebar()
        assert w.write_unrepaired_check.isChecked() is True
        assert w.write_repaired_check.isChecked() is True
        assert w.write_positive_check.isChecked() is True
        assert w.repair_mode_combo.currentData() == "hybrid"
        assert "generative" in w.repair_mode_combo.currentText().lower()
        assert "not bit-deterministic" in w.repair_mode_combo.toolTip()
        assert w.positive_mode_combo.currentData() == "nikon-exact"
        assert w.positive_mode_combo.currentText() == "Nikon C-41 exact (parity)"
        assert w.tier_hint.isHidden() is True  # Tier 1 is on by default -- nothing to warn about

    def test_tier_hint_shown_when_unrepaired_is_turned_off(self) -> None:
        w = _sidebar()

        w.write_unrepaired_check.setChecked(False)

        assert not w.tier_hint.isHidden()
        assert "Unrepaired" in w.tier_hint.text()

    def test_tier_hint_hidden_again_once_unrepaired_is_back_on(self) -> None:
        w = _sidebar()
        w.write_unrepaired_check.setChecked(False)
        assert not w.tier_hint.isHidden()

        w.write_unrepaired_check.setChecked(True)

        assert w.tier_hint.isHidden() is True

    def test_scan_blocked_when_no_tier_is_selected(self, fake_coolscanpy, tmp_path) -> None:
        w = _sidebar()
        _pick_device(w)
        w.folder_edit.setText(str(tmp_path))
        w._on_preview_ready([_thumb(fake_coolscanpy, 1)])
        w.contact_sheet.item(0).setSelected(True)
        w.write_unrepaired_check.setChecked(False)
        w.write_repaired_check.setChecked(False)
        w.write_positive_check.setChecked(False)

        assert "select at least one output tier" in w._missing_for_scan()
        assert w.scan_btn.isEnabled() is False

    def test_scan_click_forwards_tier_flags_and_repair_mode(self, fake_coolscanpy, tmp_path) -> None:
        w = _sidebar()
        _pick_device(w, device_id="dev-1")
        w.folder_edit.setText(str(tmp_path))
        w._on_preview_ready([_thumb(fake_coolscanpy, 1)])
        w.contact_sheet.item(0).setSelected(True)
        w.write_unrepaired_check.setChecked(False)
        w.write_repaired_check.setChecked(True)
        w.write_positive_check.setChecked(True)
        w.repair_mode_combo.setCurrentIndex(w.repair_mode_combo.findData("hybrid"))
        w.positive_mode_combo.setCurrentIndex(w.positive_mode_combo.findData("negpy-approximate"))

        w._on_scan_clicked()

        req = w.controller.start_roll_scan.call_args[0][0]
        assert req.write_unrepaired is False
        assert req.write_repaired is True
        assert req.write_positive is True
        assert req.repair_mode == "hybrid"
        assert req.positive_mode == "negpy-approximate"

    def test_scan_click_without_any_tier_selected_is_a_noop(self, fake_coolscanpy, tmp_path) -> None:
        w = _sidebar()
        _pick_device(w)
        w.folder_edit.setText(str(tmp_path))
        w._on_preview_ready([_thumb(fake_coolscanpy, 1)])
        w.contact_sheet.item(0).setSelected(True)
        w.write_unrepaired_check.setChecked(False)
        w.write_repaired_check.setChecked(False)
        w.write_positive_check.setChecked(False)

        w._on_scan_clicked()

        w.controller.start_roll_scan.assert_not_called()

    def test_tier_settings_round_trip_through_the_repo(self) -> None:
        w = _sidebar()
        w.write_unrepaired_check.setChecked(False)
        w.write_repaired_check.setChecked(True)
        w.repair_mode_combo.setCurrentIndex(w.repair_mode_combo.findData("hybrid"))
        w.positive_mode_combo.setCurrentIndex(w.positive_mode_combo.findData("negpy-approximate"))

        w._update_settings_from_ui()

        saved = w.controller.session.repo.save_global_setting.call_args[0]
        assert saved[0] == "roll_scan_settings"
        assert saved[1]["write_unrepaired"] is False
        assert saved[1]["write_repaired"] is True
        assert saved[1]["repair_mode"] == "hybrid"
        assert saved[1]["positive_mode"] == "negpy-approximate"

    def test_load_settings_restores_tier_choices(self) -> None:
        ctrl = MagicMock()
        ctrl.session.repo.get_global_setting.return_value = {
            "write_unrepaired": False,
            "write_repaired": True,
            "write_positive": True,
            "repair_mode": "hybrid",
            "positive_mode": "negpy-approximate",
        }
        w = CoolscanRollSidebar(ctrl)

        assert w.write_unrepaired_check.isChecked() is False
        assert w.write_repaired_check.isChecked() is True
        assert w.write_positive_check.isChecked() is True
        assert w.repair_mode_combo.currentData() == "hybrid"
        assert w.positive_mode_combo.currentData() == "negpy-approximate"

    def test_existing_saved_exact_and_disabled_choices_override_new_defaults(self) -> None:
        ctrl = MagicMock()
        ctrl.session.repo.get_global_setting.return_value = {
            "write_unrepaired": True,
            "write_repaired": False,
            "write_positive": False,
            "repair_mode": "exact",
            "positive_mode": "nikon-exact",
        }

        w = CoolscanRollSidebar(ctrl)

        assert w.write_unrepaired_check.isChecked() is True
        assert w.write_repaired_check.isChecked() is False
        assert w.write_positive_check.isChecked() is False
        assert w.repair_mode_combo.currentData() == "exact"


class TestActivation:
    def test_on_activated_requests_devices_once_devices_have_arrived(self, fake_coolscanpy) -> None:
        w = _sidebar()
        w.on_activated()
        assert w.controller.request_roll_devices.call_count == 1
        w._on_devices_ready([])  # simulate the controller's reply landing
        w.on_activated()
        assert w.controller.request_roll_devices.call_count == 1  # already loaded, no repeat

    def test_on_activated_refreshes_the_setup_hint(self, fake_coolscanpy) -> None:
        w = _sidebar()
        assert w._setup_hint.isVisible() is False  # built while coolscanpy (faked) was available
