import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QDialog

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.widgets import assign_capture_times_dialog as module
from negpy.desktop.view.widgets.assign_capture_times_dialog import AssignCaptureTimesDialog, open_assign_capture_times_dialog
from negpy.features.metadata.capture_plan import Place, PlanFrame
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.reference import ReferencePhoto
from negpy.infrastructure.storage.repository import StorageRepository
from tests.conftest import FakeRepo

HURGHADA = Place(27.2574, 33.8128, "Hurghada", "", "Egypt")
TIME, SET_BY, WARNINGS = 1, 4, 5


def _frames(n: int = 4) -> list[PlanFrame]:
    return [PlanFrame(key=f"h{i}", name=f"F{i + 1:02d}") for i in range(n)]


def _photo(path: str, hh: int, mm: int, day: int = 25) -> ReferencePhoto:
    return ReferencePhoto(path=path, local=datetime(2026, 10, day, hh, mm), offset=timedelta(hours=3), latitude=27.14, longitude=33.88)


def _open(qapp, n: int = 4, **kw) -> AssignCaptureTimesDialog:
    frames = _frames(n)
    return AssignCaptureTimesDialog(None, {"roll": frames, "selection": frames[:2]}, {}, scope="roll", **kw)


@pytest.fixture
def dialog(qapp):
    dlg = _open(qapp)
    yield dlg
    dlg.done(QDialog.DialogCode.Rejected)


def _set_day(dlg: AssignCaptureTimesDialog, text: str = "2026-10-25") -> None:
    dlg.day_edit.setText(text)
    dlg.day_edit.editingFinished.emit()


def _ready(dlg: AssignCaptureTimesDialog, *photos: ReferencePhoto) -> None:
    _set_day(dlg)
    dlg.plan.set_own(0, place=HURGHADA)
    dlg._add_photo_files([])
    dlg.plan.add_references(list(photos))
    dlg.preview_choice.setCurrentIndex(dlg.preview_choice.findData("frames"))
    dlg._refresh()


def _times(dlg: AssignCaptureTimesDialog) -> list[str]:
    return [dlg.table.item(r, TIME).text()[-8:] for r in range(dlg.table.rowCount())]


def _choose(dlg: AssignCaptureTimesDialog, path: str) -> None:
    item = next(
        dlg.references.item(i) for i in range(dlg.references.count()) if dlg.references.item(i).data(Qt.ItemDataRole.UserRole) == path
    )
    dlg._on_reference_clicked(item)


def _slots(dlg: AssignCaptureTimesDialog) -> list[int]:
    return [h.index for h in dlg.strip.hits() if h.kind == "slot"]


def test_a_roll_without_a_day_blocks_apply(dialog) -> None:
    assert not dialog.apply_btn.isEnabled()
    assert "F01" in dialog.status.text()


def test_a_day_and_a_place_time_every_frame(dialog) -> None:
    _ready(dialog)
    assert dialog.apply_btn.isEnabled()
    assert _times(dialog) == ["12:00:00", "12:01:00", "12:02:00", "12:03:00"]


def test_a_day_picked_on_the_calendar_sets_the_segment_day(dialog) -> None:
    assert dialog.day_calendar_btn.toolTip()
    dialog._set_day(date(2026, 10, 26))
    assert dialog.day_edit.text() == "2026-10-26" and dialog.plan.values_at(0).day == date(2026, 10, 26)


def test_choosing_a_photo_shows_a_slot_in_every_gap(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0))
    assert _slots(dialog) == []
    _choose(dialog, "/r/a.heic")
    assert _slots(dialog) == [-1, 0, 1, 2, 3]
    assert dialog.stop_btn.isVisibleTo(dialog)


def test_a_click_on_a_slot_places_the_photo_between_frames(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0))
    _choose(dialog, "/r/a.heic")
    dialog.strip.slotClicked.emit(1)
    assert dialog.plan.placed == {"/r/a.heic": 1}
    assert _times(dialog) == ["10:56:00", "10:58:00", "11:02:00", "11:04:00"]
    assert dialog.action_text.text() == "a.heic sits between F02 and F03."
    assert _slots(dialog) == []


def test_a_frame_click_while_placing_explains_the_gaps(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0))
    _choose(dialog, "/r/a.heic")
    dialog.strip.frameClicked.emit(2)
    assert "between frames" in dialog.action_text.text() and dialog.plan.placed == {}


def test_a_placed_photo_has_one_place(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0))
    _choose(dialog, "/r/a.heic")
    dialog.strip.slotClicked.emit(1)
    _choose(dialog, "/r/a.heic")
    assert dialog._armed == "" and dialog._sel_photo == "/r/a.heic"
    dialog.move_btn.click()
    dialog.strip.slotClicked.emit(-1)
    assert dialog.plan.placed == {"/r/a.heic": -1}
    dialog.remove_btn.click()
    assert dialog.plan.placed == {}


def test_selected_frames_limit_what_apply_writes(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0), _photo("/r/b.heic", 12, 0))
    dialog.plan.place("/r/a.heic", 0)
    dialog.plan.place("/r/b.heic", 2)
    dialog.scope_choice.setCurrentIndex(dialog.scope_choice.findData("selection"))
    assert dialog.plan.placed == {"/r/a.heic": 0, "/r/b.heic": 2}
    assert [f.muted for f in dialog.strip._frames] == [False, False, True, True]
    dialog.apply_btn.click()
    assert set(dialog.changes()) == {"h0", "h1"}


def test_a_day_set_on_a_frame_applies_to_the_frames_after_it(dialog) -> None:
    _ready(dialog)
    dialog.strip.frameClicked.emit(2)
    dialog._set_day(date(2026, 10, 26))
    assert [dialog.table.item(r, TIME).text()[:10] for r in range(4)] == ["2026-10-25", "2026-10-25", "2026-10-26", "2026-10-26"]
    dialog.strip.frameClicked.emit(3)
    assert dialog.day_edit.text() == "" and dialog.day_edit.placeholderText() == "From F03: 2026-10-26"
    assert not dialog.day_clear_btn.isEnabled()
    dialog.strip.frameClicked.emit(2)
    dialog.day_clear_btn.click()
    assert dialog.plan.values_at(3).day == date(2026, 10, 25)


def test_a_photo_dropped_on_the_strip_is_placed(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0))
    dialog.strip.photoDropped.emit("/r/a.heic", 3)
    assert dialog.plan.placed == {"/r/a.heic": 3}


def test_a_cut_click_splits_and_joins(dialog) -> None:
    _ready(dialog)
    dialog.strip.cutClicked.emit(1, 0)
    assert dialog.plan.bounds() == [(0, 1), (2, 3)]
    assert [h.tooltip for h in dialog.strip.hits() if h.kind == "segment"][1].startswith("Segment 2 · 2026-10-25")
    dialog.strip.cutClicked.emit(1, 0)
    assert dialog.plan.bounds() == [(0, 3)]


def test_a_cut_after_a_photo_keeps_it_in_the_first_segment(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0))
    dialog.plan.place("/r/a.heic", 1)
    dialog._refresh()
    assert [(h.index, h.sub) for h in dialog.strip.hits() if h.kind == "cut"] == [(0, 0), (1, 0), (1, 1), (2, 0)]
    dialog.strip.cutClicked.emit(1, 1)
    assert dialog.plan.segment_of_photo("/r/a.heic") == 0
    assert _times(dialog)[2:] == ["12:00:00", "12:01:00"]


def test_a_click_on_a_segment_bar_selects_it(dialog) -> None:
    _ready(dialog)
    dialog.strip.cutClicked.emit(1, 0)
    dialog.strip.segmentClicked.emit(1)
    assert dialog._sel_frame == 2 and dialog.frame_header.text() == "Frame F03"


def test_reference_cards_show_time_date_place_and_their_place_in_the_strip(dialog) -> None:
    _ready(dialog, _photo("/r/IMG_2514.HEIC", 11, 9))
    dialog.days_choice.setCurrentIndex(dialog.days_choice.findData("all"))
    dialog.plan.place("/r/IMG_2514.HEIC", 0)
    dialog._refresh()
    card = dialog.references.item(0).data(module._CARD_ROLE)
    assert card == {
        "time": "11:09:00",
        "date": "2026-10-25 · UTC+03:00",
        "place": "27.14000, 33.88000",
        "name": "IMG_2514.HEIC",
        "placed": "Between F01 and F02",
    }


def test_long_frame_names_show_their_frame_number(qapp) -> None:
    frames = [PlanFrame(key=f"h{i}", name=f"26-11-14 R01 F{37 + i:03d}") for i in range(2)]
    dlg = AssignCaptureTimesDialog(None, {"roll": frames}, {})
    assert [f.name for f in dlg.strip._frames] == ["F037", "F038"]
    dlg.done(QDialog.DialogCode.Rejected)


def test_a_typed_time_pins_the_frame(dialog) -> None:
    _ready(dialog)
    dialog.strip.frameClicked.emit(2)
    assert dialog.time_edit.placeholderText() == "12:02:00"
    dialog.time_edit.setText("25:99")
    dialog.time_edit.editingFinished.emit()
    assert dialog.time_edit.styleSheet() and dialog.plan.typed == {}
    dialog.time_edit.setText("15:30")
    dialog.time_edit.editingFinished.emit()
    assert _times(dialog)[2] == "15:30:00"
    assert dialog.table.item(2, SET_BY).text() == "Typed time"
    assert dialog.time_clear_btn.isEnabled()
    dialog.time_clear_btn.click()
    assert dialog.plan.typed == {}


def test_reverse_shows_the_strip_in_film_order(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0))
    dialog.plan.place("/r/a.heic", 0)
    dialog.direction_choice.setCurrentIndex(dialog.direction_choice.findData("reverse"))
    assert [f.name for f in dialog.strip._frames] == ["F04", "F03", "F02", "F01"]
    assert [dialog.table.item(r, 0).text() for r in range(4)] == ["F04", "F03", "F02", "F01"]
    assert dialog.plan.gap_of("/r/a.heic") == 2 and dialog.action_text.text().startswith("F")


def test_the_preview_is_folded_at_first_and_lists_phone_photos(dialog) -> None:
    _ready(
        dialog,
        _photo(
            "/r/a.heic",
            12,
            1,
        ),
    )
    assert not dialog.preview_fold.is_open()
    dialog.preview_choice.setCurrentIndex(dialog.preview_choice.findData("with"))
    labels = [dialog.table.item(r, 0).text() for r in range(dialog.table.rowCount())]
    assert labels == ["F01", "F02", "a.heic", "F03", "F04"]


def test_the_preview_fold_remembers_it_was_open(qapp) -> None:
    repo = FakeRepo()
    first = _open(qapp, repo=repo)
    first.preview_fold.set_open(True)
    first.done(QDialog.DialogCode.Rejected)
    second = _open(qapp, repo=repo)
    assert second.preview_fold.is_open()
    second.done(QDialog.DialogCode.Rejected)


def test_frames_far_between_two_photos_ask_for_a_split(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 13, 30), _photo("/r/b.heic", 15, 20, day=26))
    dialog.plan.place("/r/a.heic", 0)
    dialog.plan.place("/r/b.heic", 3)
    dialog._refresh()
    assert "F02 to F04 lie between a.heic and b.heic, 26 h apart." in dialog.status.text()
    assert dialog.table.item(1, WARNINGS).text() == "Long gap"


def test_escape_stops_placing_and_keeps_the_dialog_open(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0))
    dialog.show()
    _choose(dialog, "/r/a.heic")
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    assert dialog._armed == "" and dialog.isVisible()


def test_apply_returns_the_changes_and_the_plan(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0))
    dialog.plan.place("/r/a.heic", 1)
    dialog._refresh()
    dialog.apply_btn.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    changes = dialog.changes()
    assert set(changes) == {"h0", "h1", "h2", "h3"}
    assert changes["h2"]["capture_date"] == "2026-10-25 11:02:00"
    assert json.loads(changes["h2"]["capture_plan"])["place"] == "segment"


def test_reopening_shows_the_last_times_and_no_photos(qapp) -> None:
    first = _open(qapp)
    _ready(first, _photo("/r/a.heic", 11, 0))
    first.plan.place("/r/a.heic", 1)
    first._refresh()
    first._on_apply()
    stored = [replace(f, metadata=replace(MetadataConfig(), **first.changes()[f.key])) for f in _frames()]
    again = AssignCaptureTimesDialog(None, {"roll": stored}, {})
    again.preview_choice.setCurrentIndex(again.preview_choice.findData("frames"))
    assert again.references.count() == 0 and again.plan.placed == {}
    assert _times(again) == ["10:56:00", "10:58:00", "11:02:00", "11:04:00"]
    assert {again.table.item(r, SET_BY).text() for r in range(4)} == {"From the last Apply"}
    again.done(QDialog.DialogCode.Rejected)


def test_remove_photo_takes_it_out_of_the_list_and_the_strip(dialog) -> None:
    _ready(dialog, _photo("/r/a.heic", 11, 0))
    dialog.days_choice.setCurrentIndex(dialog.days_choice.findData("all"))
    assert not dialog.remove_photo_btn.isEnabled()
    _choose(dialog, "/r/a.heic")
    dialog.strip.slotClicked.emit(1)
    assert dialog.remove_photo_btn.isEnabled()
    dialog.remove_photo_btn.click()
    assert dialog.references.count() == 0 and dialog.plan.placed == {}


def test_a_time_after_the_scan_is_marked(qapp) -> None:
    scanned = datetime(2026, 10, 25, 9, 0, 30, tzinfo=timezone.utc)
    dlg = _open(qapp, n=2, scan_times={"h0": scanned, "h1": scanned})
    _ready(dlg)
    assert [dlg.table.item(r, WARNINGS).text() for r in range(2)] == ["", "After the scan"]
    dlg.done(QDialog.DialogCode.Rejected)


def _session() -> DesktopSessionManager:
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.get_max_history_index.return_value = 0
    mgr = DesktopSessionManager(repo)
    scanned = datetime(2026, 11, 14, 18, 0, tzinfo=timezone.utc).timestamp()
    mgr.state.uploaded_files = [{"name": f"f{i}.tif", "path": f"/tmp/f{i}.tif", "hash": f"h{i}", "mtime": scanned} for i in range(3)]
    mgr.asset_model.refresh()
    mgr.state.selected_file_idx = 0
    mgr.state.selected_indices = [0]
    mgr.state.current_file_hash = "h0"
    mgr.state.current_file_path = "/tmp/f0.tif"
    return mgr


def test_the_opener_writes_what_the_dialog_applies(qapp, monkeypatch) -> None:
    mgr = _session()
    opened: list[AssignCaptureTimesDialog] = []

    def run(dlg: AssignCaptureTimesDialog) -> int:
        opened.append(dlg)
        _set_day(dlg)
        dlg._on_apply()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(module.AssignCaptureTimesDialog, "exec", run)
    assert open_assign_capture_times_dialog(None, mgr) == 3
    assert [f.name for f in opened[0].plan.frames] == ["f0", "f1", "f2"]
    assert opened[0]._scan_times["h0"] == datetime(2026, 11, 14, 18, 0, tzinfo=timezone.utc)
    assert mgr.state.config.metadata.capture_date == "2026-10-25 12:00:00"
    saved = {c.args[0]: c.args[1] for c in mgr.repo.save_file_settings.call_args_list}
    assert saved["h2"].metadata.capture_date == "2026-10-25 12:02:00"
    assert saved["h2"].metadata.capture_time_source == "start"
    assert json.loads(saved["h0"].metadata.capture_plan)["own"]["day"] == "2026-10-25"
