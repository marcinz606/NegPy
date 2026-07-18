"""Hardware-free interaction tests for the roll candidate-slot picker."""

import pytest
from PyQt6.QtCore import QPersistentModelIndex, QPoint, QRect, Qt
from PyQt6.QtGui import (
    QColor,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QStyleOptionViewItem

from negpy.desktop.view.widgets.roll_slot_model import RollPreviewSlot, RollSlotModel
from negpy.desktop.view.widgets.roll_slot_selector import RollSlotSelector
from negpy.desktop.view.shortcut_registry import default_bindings
from negpy.services.scanning.roll_preview_controls import ScanMaterial, boundary_offset_range, validate_boundary_offset


def _slots(count: int, *, warning_slots: set[int] | None = None) -> list[RollPreviewSlot]:
    warning_slots = warning_slots or set()
    thumbnail = QPixmap(120, 80)
    thumbnail.fill(QColor("#36454F"))
    return [
        RollPreviewSlot(
            slot_id,
            thumbnail,
            ("Likely blank or partial tail slot",) if slot_id in warning_slots else (),
        )
        for slot_id in range(1, count + 1)
    ]


def _shown_selector(qapp: QApplication, count: int = 40, *, warning_slots: set[int] | None = None) -> RollSlotSelector:
    selector = RollSlotSelector()
    selector.resize(660, 520)
    selector.set_slots(_slots(count, warning_slots=warning_slots))
    selector.show()
    qapp.processEvents()
    return selector


def _click_slot(
    selector: RollSlotSelector,
    slot_id: int,
    modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    index = selector.model.index(slot_id - 1, 0)
    QTest.mouseClick(
        selector.grid.viewport(),
        Qt.MouseButton.LeftButton,
        modifier,
        selector.grid.visualRect(index).center(),
    )
    QApplication.processEvents()


def test_model_requires_complete_ordered_one_based_slot_index() -> None:
    model = RollSlotModel()

    with pytest.raises(ValueError, match="contiguous and ordered"):
        model.replace_slots([RollPreviewSlot(1), RollPreviewSlot(3)])
    with pytest.raises(ValueError, match="positive 1-based"):
        RollPreviewSlot(0)
    with pytest.raises(ValueError, match="non-empty"):
        RollPreviewSlot(1, warnings=("",))


def test_boundary_offset_policy_is_slot_aware_and_strictly_integer() -> None:
    assert boundary_offset_range(1) == (0, 144)
    assert boundary_offset_range(2) == (-144, 144)
    assert validate_boundary_offset(1, 0) == 0
    assert validate_boundary_offset(1, 144) == 144
    assert validate_boundary_offset(12, -144) == -144
    assert validate_boundary_offset(12, 144) == 144

    for slot_id, offset in ((1, -1), (1, 145), (2, -145), (2, 145)):
        with pytest.raises(ValueError, match="boundary_offset"):
            validate_boundary_offset(slot_id, offset)
    with pytest.raises(ValueError, match="integer"):
        validate_boundary_offset(2, True)


def test_model_persists_per_slot_boundary_offsets_and_emits_the_role(qapp: QApplication) -> None:
    model = RollSlotModel([RollPreviewSlot(1, boundary_offset=17), RollPreviewSlot(2, boundary_offset=-32)])
    changed_roles: list[list[int]] = []
    model.dataChanged.connect(lambda _first, _last, roles: changed_roles.append(roles))

    second = model.index(1, 0)
    assert model.data(second, RollSlotModel.BOUNDARY_OFFSET_ROLE) == -32
    assert model.boundary_offset_for_slot_id(2) == -32

    model.set_boundary_offset(2, 41)

    assert model.data(second, RollSlotModel.BOUNDARY_OFFSET_ROLE) == 41
    assert model.boundary_offset_for_slot_id(2) == 41
    assert changed_roles == [[RollSlotModel.BOUNDARY_OFFSET_ROLE, int(Qt.ItemDataRole.ToolTipRole)]]


def test_model_exposes_one_meter_inset_for_every_thumbnail_and_repaints_all_rows(
    qapp: QApplication,
) -> None:
    model = RollSlotModel(_slots(3))
    changes: list[tuple[int, int, list[int]]] = []
    model.dataChanged.connect(lambda first, last, roles: changes.append((first.row(), last.row(), roles)))

    assert all(model.data(model.index(row, 0), RollSlotModel.METER_INSET_PERCENT_ROLE) == 10 for row in range(3))

    model.set_meter_inset_percent(24)

    assert all(model.data(model.index(row, 0), RollSlotModel.METER_INSET_PERCENT_ROLE) == 24 for row in range(3))
    assert changes == [(0, 2, [RollSlotModel.METER_INSET_PERCENT_ROLE])]


def test_thumbnail_delegate_dims_only_outside_the_metering_region(
    qapp: QApplication,
) -> None:
    selector = RollSlotSelector()
    thumbnail = QPixmap(100, 60)
    thumbnail.fill(QColor("white"))
    selector.set_slots([RollPreviewSlot(1, thumbnail)])
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 142, 112)

    def painted_values(inset: int) -> tuple[int, int]:
        selector.set_meter_inset_percent(inset)
        canvas = QImage(142, 112, QImage.Format.Format_ARGB32)
        canvas.fill(QColor("black"))
        painter = QPainter(canvas)
        selector.grid.itemDelegate().paint(
            painter,
            option,
            selector.model.index(0, 0),
        )
        painter.end()
        outside = canvas.pixelColor(25, 45).value()
        center = canvas.pixelColor(71, 45).value()
        return outside, center

    try:
        whole_outside, whole_center = painted_values(0)
        inset_outside, inset_center = painted_values(20)

        assert abs(whole_outside - whole_center) <= 2
        assert inset_outside < inset_center - 30
        assert inset_center >= whole_center - 2
        assert thumbnail.toImage().pixelColor(5, 30) == QColor("white")
    finally:
        selector.close()


def test_all_addressable_slots_remain_visible_selectable_and_unrenumbered(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, warning_slots={38, 39, 40})
    try:
        assert selector.model.rowCount() == 40
        assert selector.summary_label.text() == "40 scanner slots"
        assert selector.model.data(selector.model.index(39, 0), RollSlotModel.SLOT_ID_ROLE) == 40

        warning_index = selector.model.index(39, 0)
        flags = selector.model.flags(warning_index)
        assert flags & Qt.ItemFlag.ItemIsEnabled
        assert flags & Qt.ItemFlag.ItemIsSelectable
        assert selector.model.data(warning_index, RollSlotModel.HAS_WARNINGS_ROLE) is True
        tooltip = selector.model.data(warning_index, Qt.ItemDataRole.ToolTipRole)
        assert "remains selectable" in tooltip

        selector.set_selected_slot_ids([40])
        assert selector.selected_slot_ids() == [40]
    finally:
        selector.close()


def test_transport_warning_codes_are_explained_in_plain_language() -> None:
    model = RollSlotModel(
        [
            RollPreviewSlot(
                1,
                warnings=(
                    "start-broad-clear-region",
                    "broad-clear-region",
                    "transport-origin-inferred",
                ),
            )
        ]
    )

    tooltip = model.data(model.index(0, 0), Qt.ItemDataRole.ToolTipRole)

    assert "starting boundary" in tooltip
    assert "wider than a normal frame gap" in tooltip
    assert tooltip.count("wider than a normal frame gap") == 1
    assert "position was inferred" in tooltip
    assert "live safety recheck" in tooltip
    assert "before full-quality capture" in tooltip
    assert "leader" not in tooltip
    assert "start-broad-clear-region" not in tooltip
    assert "broad-clear-region" not in tooltip
    assert "transport-origin-inferred" not in tooltip


def test_model_distinguishes_advisory_from_blocking_review_warnings() -> None:
    model = RollSlotModel(
        [
            RollPreviewSlot(1, warnings=("beyond-advisory-content-end",)),
            RollPreviewSlot(2, warnings=("transport-origin-inferred",)),
        ]
    )

    advisory = model.index(0, 0)
    blocking = model.index(1, 0)

    assert model.data(advisory, RollSlotModel.WARNING_STATE_ROLE) == "advisory"
    assert model.data(blocking, RollSlotModel.WARNING_STATE_ROLE) == "blocking"
    assert "Advisory preview warning" in model.data(advisory, Qt.ItemDataRole.ToolTipRole)
    assert "Blocking review" in model.data(blocking, Qt.ItemDataRole.ToolTipRole)
    assert "blocking review required" in model.data(blocking, Qt.ItemDataRole.AccessibleTextRole)


@pytest.mark.parametrize(
    ("warning", "expected"),
    (
        ("start-outside-index-raster", "starting boundary falls outside"),
        ("start-broad-clear-region", "starting boundary is in a clear region"),
        ("start-narrow-gap-evidence", "starting boundary has clear-gap evidence"),
        ("start-no-local-gap-run", "starting boundary has no clear film-gap evidence"),
        ("start-low-gap-evidence", "starting boundary has weak film-gap evidence"),
        ("end-outside-index-raster", "ending boundary falls outside"),
        ("end-broad-clear-region", "ending boundary is in a clear region"),
        ("end-narrow-gap-evidence", "ending boundary has clear-gap evidence"),
        ("end-no-local-gap-run", "ending boundary has no clear film-gap evidence"),
        ("end-low-gap-evidence", "ending boundary has weak film-gap evidence"),
        ("outside-index-raster", "boundary used to position the scanner falls outside"),
        ("broad-clear-region", "boundary used to position the scanner is in a clear region"),
        ("narrow-gap-evidence", "boundary used to position the scanner has clear-gap evidence"),
        ("no-local-gap-run", "boundary used to position the scanner has no clear film-gap evidence"),
        ("low-gap-evidence", "boundary used to position the scanner has weak film-gap evidence"),
        ("low-content-support", "too little image content"),
        ("partial-index-coverage", "Only part of this slot"),
        ("spacing-outlier", "differs from the roll's regular frame spacing"),
        ("transport-origin-inferred", "position was inferred"),
        ("ambiguous-content-tail-boundary", "more than one plausible point"),
        ("beyond-advisory-content-end", "past the last likely photographed frame"),
    ),
)
def test_every_roll_warning_code_has_context_safe_plain_language(
    warning: str,
    expected: str,
) -> None:
    model = RollSlotModel([RollPreviewSlot(1, warnings=(warning,))])

    tooltip = model.data(model.index(0, 0), Qt.ItemDataRole.ToolTipRole)

    assert expected in tooltip
    assert warning not in tooltip


def test_ctrl_and_command_click_toggle_individual_slots(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 12)
    try:
        _click_slot(selector, 3)
        _click_slot(selector, 8, Qt.KeyboardModifier.ControlModifier)
        _click_slot(selector, 2, Qt.KeyboardModifier.MetaModifier)
        assert selector.selected_slot_ids() == [2, 3, 8]

        _click_slot(selector, 8, Qt.KeyboardModifier.ControlModifier)
        assert selector.selected_slot_ids() == [2, 3]
    finally:
        selector.close()


def test_shift_click_selects_a_contiguous_display_range(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 12)
    try:
        _click_slot(selector, 3)
        _click_slot(selector, 7, Qt.KeyboardModifier.ShiftModifier)
        assert selector.selected_slot_ids() == [3, 4, 5, 6, 7]
    finally:
        selector.close()


def test_drag_across_slots_selects_the_whole_display_range(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 12)
    try:
        start = selector.grid.visualRect(selector.model.index(1, 0)).center()
        end = selector.grid.visualRect(selector.model.index(5, 0)).center()

        QTest.mousePress(selector.grid.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(selector.grid.viewport(), end, 20)
        QTest.mouseRelease(selector.grid.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
        qapp.processEvents()

        assert selector.selected_slot_ids() == [2, 3, 4, 5, 6]
    finally:
        selector.close()


def test_empty_space_marquee_selects_every_intersected_cell(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 12)
    try:
        first = selector.grid.visualRect(selector.model.index(0, 0))
        start = QPoint(selector.grid.viewport().width() - 2, first.bottom() + 2)
        end = QPoint(1, 1)
        assert not selector.grid.indexAt(start).isValid()

        rectangle = QRect(start, end).normalized()
        expected = [
            row + 1
            for row in range(selector.model.rowCount())
            if selector.grid.visualRect(selector.model.index(row, 0)).intersects(rectangle)
        ]
        assert len(expected) >= 2

        QTest.mousePress(selector.grid.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(selector.grid.viewport(), end, 20)
        QTest.mouseRelease(selector.grid.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
        qapp.processEvents()

        assert selector.selected_slot_ids() == expected
    finally:
        selector.close()


def test_select_all_clear_and_selected_count(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 40)
    try:
        assert selector.selected_count_label.text() == "0 of 40 selected"
        assert not selector.scan_button.isEnabled()

        QTest.mouseClick(selector.select_all_button, Qt.MouseButton.LeftButton)
        assert selector.selected_slot_ids() == list(range(1, 41))
        assert selector.selected_count_label.text() == "40 of 40 selected"
        assert selector.scan_button.isEnabled()

        QTest.mouseClick(selector.clear_button, Qt.MouseButton.LeftButton)
        assert selector.selected_slot_ids() == []
        assert selector.selected_count_label.text() == "0 of 40 selected"
        assert not selector.scan_button.isEnabled()
    finally:
        selector.close()


def test_selected_frames_show_final_scratch_and_free_space_estimates(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 3)
    try:
        assert selector.storage_estimate_label.text() == "Select frames to estimate storage."

        selector.set_available_storage_bytes(8 * 1024**3)
        selector.set_selected_slot_ids([1, 3])

        estimate = selector.storage_estimate_label.text()
        assert "2 frames" in estimate
        assert "final budget 512 MiB" in estimate
        assert "scratch 591 MiB" in estimate
        assert "2.08 GiB working space needed" in estimate
        assert "8.00 GiB free" in estimate
        assert selector.scan_button.isEnabled()

        selector.set_available_storage_bytes(1024**3)

        assert not selector.scan_button.isEnabled()
        assert "only 1.00 GiB is free" in selector.scan_button.toolTip()
    finally:
        selector.close()


def test_storage_gate_includes_conservative_output_budget_for_every_selected_frame(
    qapp: QApplication,
) -> None:
    selector = _shown_selector(qapp, 3)
    required_for_two = 1_693_200_384 + 2 * 256 * 1024**2
    try:
        selector.set_available_storage_bytes(required_for_two)
        selector.set_selected_slot_ids([1, 2])

        assert selector.scan_button.isEnabled()

        selector.set_available_storage_bytes(required_for_two - 1)

        assert not selector.scan_button.isEnabled()
        assert "Full-quality capture needs" in selector.scan_button.toolTip()
    finally:
        selector.close()


def test_storage_gate_resyncs_immediately_for_color_and_black_and_white(
    qapp: QApplication,
) -> None:
    selector = _shown_selector(qapp, 3)
    try:
        selector.set_available_storage_bytes(1024**3)
        selector.set_selected_slot_ids([1, 2])

        assert selector.scan_material() is ScanMaterial.COLOR_NEGATIVE
        assert not selector.scan_button.isEnabled()
        assert "final budget 512 MiB" in selector.storage_estimate_label.text()
        assert "scratch 591 MiB" in selector.storage_estimate_label.text()

        selector.set_scan_material(ScanMaterial.BLACK_AND_WHITE_NEGATIVE)

        assert selector.scan_button.isEnabled()
        assert "final budget 384 MiB" in selector.storage_estimate_label.text()
        assert "scratch none" in selector.storage_estimate_label.text()
        assert "0.62 GiB working space needed" in selector.storage_estimate_label.text()

        selector.set_scan_material(ScanMaterial.COLOR_NEGATIVE)

        assert not selector.scan_button.isEnabled()
        assert "scratch 591 MiB" in selector.storage_estimate_label.text()
    finally:
        selector.close()


def test_scan_action_emits_explicit_slot_ids_in_display_order(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 12, warning_slots={9})
    requests: list[list[int]] = []
    selector.scan_requested.connect(requests.append)
    try:
        selector.request_scan()
        assert requests == []

        selector.set_selected_slot_ids([9, 2, 5])
        QTest.mouseClick(selector.scan_button, Qt.MouseButton.LeftButton)

        assert requests == [[2, 5, 9]]
    finally:
        selector.close()


def test_inferred_origin_requires_explicit_review_before_scan_request(
    qapp: QApplication,
) -> None:
    selector = RollSlotSelector()
    selector.set_slots(
        [
            RollPreviewSlot(1),
            RollPreviewSlot(2, warnings=("transport-origin-inferred",)),
        ]
    )
    requests: list[list[int]] = []
    selector.scan_requested.connect(requests.append)
    selector.show()
    qapp.processEvents()
    try:
        selector.set_selected_slot_ids([2])

        assert selector.review_approval_check.isVisible()
        assert selector.review_approval_check.text() == "I reviewed this inferred position"
        assert not selector.review_approval_check.isChecked()
        assert not selector.scan_button.isEnabled()
        assert "review slot 02" in selector.scan_button.toolTip().lower()

        selector.review_approval_check.setChecked(True)

        assert selector.model.data(selector.model.index(1, 0), RollSlotModel.WARNING_STATE_ROLE) == "reviewed"
        assert selector.approved_blocking_slot_ids() == [2]
        assert selector.scan_button.isEnabled()

        selector.request_thumbnail_reload()

        assert selector.approved_blocking_slot_ids() == []
        assert not selector.scan_button.isEnabled()
        selector.review_approval_check.setChecked(True)
        QTest.mouseClick(selector.scan_button, Qt.MouseButton.LeftButton)
        assert requests == [[2]]

        selector.confirm_slot_thumbnail(2, 0, None)

        assert selector.approved_blocking_slot_ids() == []
        assert selector.model.data(selector.model.index(1, 0), RollSlotModel.WARNING_STATE_ROLE) == "blocking"
        assert not selector.review_approval_check.isChecked()
        assert not selector.scan_button.isEnabled()

        selector.review_approval_check.setChecked(True)
        selector.boundary_offset_spin.setValue(-1)

        assert selector.approved_blocking_slot_ids() == []
        assert selector.model.data(selector.model.index(1, 0), RollSlotModel.WARNING_STATE_ROLE) == "blocking"
    finally:
        selector.close()


def test_boundary_offset_controls_follow_current_slot_and_reload_exactly_that_slot(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 4)
    reloads: list[tuple[int, int]] = []
    selector.thumbnail_reload_requested.connect(lambda slot_id, offset: reloads.append((slot_id, offset)))
    try:
        selector.set_selected_slot_ids([1])
        assert selector.boundary_offset_label.text() == "Film Spacing Offset: Slot 01"
        assert selector.boundary_offset_spin.minimum() == 0
        assert selector.boundary_offset_spin.maximum() == 144
        assert selector.boundary_offset_slider.minimum() == 0
        assert selector.boundary_offset_slider.maximum() == 144
        assert selector.boundary_offset_slider.value() == 0
        selector.boundary_offset_spin.setValue(37)
        assert selector.boundary_offset_slider.value() == 37
        assert selector.boundary_offset_spin.text() == "+37 rows"
        assert selector.model.boundary_offset_for_slot_id(1) == 37
        assert not selector.model.boundary_offset_is_confirmed(1)
        assert not selector.scan_button.isEnabled()

        QTest.mouseClick(selector.reload_thumbnail_button, Qt.MouseButton.LeftButton)
        assert reloads == [(1, 37)]
        selector.confirm_slot_thumbnail(1, 37, None)
        assert selector.model.boundary_offset_is_confirmed(1)
        assert selector.scan_button.isEnabled()

        selector.set_selected_slot_ids([1, 3])
        assert selector.boundary_offset_spin.minimum() == -144
        assert selector.boundary_offset_spin.maximum() == 144
        assert selector.boundary_offset_slider.minimum() == -144
        assert selector.boundary_offset_slider.maximum() == 144
        assert selector.boundary_offset_slider.value() == 0
        selector.boundary_offset_slider.setValue(-21)
        assert selector.boundary_offset_spin.value() == -21
        assert selector.model.boundary_offset_for_slot_id(1) == 37
        assert selector.model.boundary_offset_for_slot_id(3) == -21

        QTest.mouseClick(selector.reload_thumbnail_button, Qt.MouseButton.LeftButton)
        assert reloads == [(1, 37), (3, -21)]
        assert not selector.scan_button.isEnabled()
        selector.confirm_slot_thumbnail(3, -21, None)
        assert selector.scan_button.isEnabled()
        assert selector.selected_slot_ids() == [1, 3]
    finally:
        selector.close()


def test_film_spacing_controls_have_practical_hit_areas_and_discoverable_shortcuts(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 3)
    try:
        selector.set_selected_slot_ids([2])

        assert selector.boundary_offset_spin.minimumHeight() >= 40
        assert selector.boundary_offset_slider.minimumHeight() >= 40
        assert selector.reload_thumbnail_button.minimumHeight() >= 40
        assert selector.boundary_offset_spin.font().fixedPitch()
        assert "Alt+Left / Alt+Right" in selector.boundary_offset_slider.toolTip()
        assert selector.offset_shortcut_label.text() == "Alt+← / Alt+→"

        decrease = selector._offset_decrease_shortcut.key().toString(QKeySequence.SequenceFormat.PortableText)
        increase = selector._offset_increase_shortcut.key().toString(QKeySequence.SequenceFormat.PortableText)
        assert (decrease, increase) == ("Alt+Left", "Alt+Right")
        assigned = {QKeySequence(key).toString(QKeySequence.SequenceFormat.PortableText) for key in default_bindings().values() if key}
        assert decrease not in assigned
        assert increase not in assigned
    finally:
        selector.close()


def test_roll_selector_does_not_compete_for_the_window_escape_shortcut(
    qapp: QApplication,
) -> None:
    selector = _shown_selector(qapp, 3)
    try:
        escape = QKeySequence(Qt.Key.Key_Escape)
        assert all(
            shortcut.key() != escape
            for shortcut in selector.findChildren(QShortcut)
        )
    finally:
        selector.close()


def test_film_spacing_shortcuts_nudge_current_frame_within_slot_bounds(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 3)
    reloads: list[tuple[int, int]] = []
    selector.thumbnail_reload_requested.connect(lambda slot_id, offset: reloads.append((slot_id, offset)))
    try:
        selector.set_selected_slot_ids([2])
        selector.grid.setFocus()
        QTest.keyClick(selector.grid, Qt.Key.Key_Left, Qt.KeyboardModifier.AltModifier)
        qapp.processEvents()
        assert selector.boundary_offset_spin.value() == -1
        assert selector.boundary_offset_slider.value() == -1
        assert selector.model.boundary_offset_for_slot_id(2) == -1
        assert not selector.model.boundary_offset_is_confirmed(2)
        assert reloads == []

        selector.boundary_offset_spin.setValue(-144)
        selector._offset_decrease_shortcut.activated.emit()
        assert selector.boundary_offset_spin.value() == -144

        selector.boundary_offset_spin.setValue(144)
        selector._offset_increase_shortcut.activated.emit()
        assert selector.boundary_offset_spin.value() == 144

        selector.set_selected_slot_ids([1])
        selector._offset_decrease_shortcut.activated.emit()
        assert selector.boundary_offset_spin.value() == 0
        assert selector.model.boundary_offset_for_slot_id(1) == 0
    finally:
        selector.close()


def test_stale_thumbnail_response_cannot_confirm_a_newer_offset(
    qapp: QApplication,
) -> None:
    selector = _shown_selector(qapp, 3)
    try:
        selector.set_selected_slot_ids([2])
        selector.boundary_offset_spin.setValue(-12)
        selector.boundary_offset_spin.setValue(-18)

        with pytest.raises(ValueError, match="stale Film Spacing Offset"):
            selector.confirm_slot_thumbnail(2, -12, None)

        assert not selector.model.boundary_offset_is_confirmed(2)
        assert not selector.scan_button.isEnabled()
        selector.confirm_slot_thumbnail(2, -18, None)
        assert selector.scan_button.isEnabled()
    finally:
        selector.close()


def test_replacing_thumbnail_preserves_slot_selection_current_index_and_offset(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 10)
    try:
        selector.set_selected_slot_ids([2, 5, 9])
        selector.model.set_boundary_offset(5, -14)
        selection_model = selector.grid.selectionModel()
        assert selection_model is not None
        current_before = QPersistentModelIndex(selection_model.currentIndex())
        fifth = selector.model.index(4, 0)
        icon_before = selector.model.data(fifth, Qt.ItemDataRole.DecorationRole)
        assert isinstance(icon_before, QIcon)

        replacement = QPixmap(120, 80)
        replacement.fill(QColor("#AA281E"))
        selector.update_slot_thumbnail(5, replacement)

        icon_after = selector.model.data(fifth, Qt.ItemDataRole.DecorationRole)
        assert isinstance(icon_after, QIcon)
        assert icon_after.cacheKey() != icon_before.cacheKey()
        assert current_before.isValid()
        assert selection_model.currentIndex() == current_before
        assert selector.selected_slot_ids() == [2, 5, 9]
        assert selector.model.boundary_offset_for_slot_id(5) == -14
    finally:
        selector.close()


def test_scan_material_choice_exposes_capture_and_ir_repair_behavior(qapp: QApplication) -> None:
    selector = _shown_selector(qapp, 2)
    changes: list[ScanMaterial] = []
    selector.scan_material_changed.connect(changes.append)
    try:
        assert selector.scan_material() is ScanMaterial.COLOR_NEGATIVE
        assert selector.scan_material_combo.currentText() == "Colour negative (C-41)"
        assert "RGB 4× + IR" in selector.scan_material_status_label.text()
        assert "IR repair: On after import" in selector.ir_repair_status_label.text()
        assert "scanner-linear master stays unchanged" in selector.ir_repair_status_label.text()
        assert selector.ir_repair_status_label.property("irRepairState") == "on"

        selector.set_scan_material(ScanMaterial.BLACK_AND_WHITE_NEGATIVE)

        assert selector.scan_material() is ScanMaterial.BLACK_AND_WHITE_NEGATIVE
        assert selector.scan_material_combo.currentText() == "Black-and-white negative"
        assert "RGB only" in selector.scan_material_status_label.text()
        assert "IR repair: Off" in selector.ir_repair_status_label.text()
        assert "no infrared channel is captured" in selector.ir_repair_status_label.text()
        assert selector.ir_repair_status_label.property("irRepairState") == "off"
        assert changes == [ScanMaterial.BLACK_AND_WHITE_NEGATIVE]
    finally:
        selector.close()


def _installed_ice_summary():
    from negpy.infrastructure.scanners.portable_digital_ice_adapter import (
        PortableDigitalIceAvailability,
    )

    return PortableDigitalIceAvailability(
        engine_installed=True,
        engine_detail="/fake/portable_digital_ice/__init__.py",
        cpu_fast_installed=True,
        cpu_fast_detail="numba 0.65.1",
        cuda_installed=False,
        cuda_detail="CUDA backend is not usable: no cupy",
    )


def test_ice_controls_disable_with_an_install_hint_when_the_engine_is_missing(qapp, monkeypatch) -> None:
    import negpy.desktop.view.widgets.roll_slot_selector as module
    from negpy.infrastructure.scanners.portable_digital_ice_adapter import (
        PortableDigitalIceAvailability,
    )

    def missing_engine():
        detail = "portable Digital ICE is not installed"
        return PortableDigitalIceAvailability(
            engine_installed=False,
            engine_detail=detail,
            cpu_fast_installed=False,
            cpu_fast_detail=detail,
            cuda_installed=False,
            cuda_detail=detail,
        )

    monkeypatch.setattr(module, "availability_summary", missing_engine)
    selector = _shown_selector(qapp)

    assert selector.ice_check.isEnabled() is False
    assert "PORTABLE_DIGITAL_ICE" in selector.ice_check.toolTip()
    selector.ice_check.setChecked(True)
    assert selector.ice_capture_enabled() is False


def test_ice_toggle_switches_recipe_text_and_storage_budget(qapp, monkeypatch) -> None:
    import negpy.desktop.view.widgets.roll_slot_selector as module

    monkeypatch.setattr(module, "availability_summary", _installed_ice_summary)
    selector = _shown_selector(qapp)
    _click_slot(selector, 1)

    assert selector.ice_check.isEnabled() is True
    assert selector.ice_backend() == "cpu-fast"
    cuda_index = selector.ice_backend_combo.findData("cuda")
    assert selector.ice_backend_combo.model().item(cuda_index).isEnabled() is False
    baseline_estimate = selector.storage_estimate_label.text()
    assert "RGB 4× + IR" in selector.scan_material_status_label.text()

    selector.ice_check.setChecked(True)

    assert selector.ice_capture_enabled() is True
    assert "dual RGBI single-sample" in selector.scan_material_status_label.text()
    assert "Digital ICE" in selector.ir_repair_status_label.text()
    assert "file-based IR repair stays off" in selector.ir_repair_status_label.text()
    assert selector.storage_estimate_label.text() != baseline_estimate

    selector.ice_check.setChecked(False)
    assert "RGB 4× + IR" in selector.scan_material_status_label.text()
    assert selector.storage_estimate_label.text() == baseline_estimate


def test_ice_controls_hide_for_black_and_white_material(qapp, monkeypatch) -> None:
    import negpy.desktop.view.widgets.roll_slot_selector as module

    monkeypatch.setattr(module, "availability_summary", _installed_ice_summary)
    selector = _shown_selector(qapp)
    selector.ice_check.setChecked(True)
    assert selector.ice_capture_enabled() is True

    selector.set_scan_material(ScanMaterial.BLACK_AND_WHITE_NEGATIVE)

    assert selector.ice_check.isVisible() is False
    assert selector.ice_capture_enabled() is False
    assert "RGB only" in selector.scan_material_status_label.text()

    selector.set_scan_material(ScanMaterial.COLOR_NEGATIVE)
    assert selector.ice_check.isVisible() is True
    assert selector.ice_capture_enabled() is True
