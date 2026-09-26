from typing import Optional

from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from negpy.desktop.view.confirm import prompt_delete_scene
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, hint_label, header_row, section_subheader, set_hint_kind, wrap_tooltip
from negpy.desktop.view.styles.theme import scene_color
from negpy.kernel.system.text import count_of
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.services.assets import rolls

# Prefixes a roll's label once it has a saved baseline -- the field itself is the
# "analyzed" indicator, so the search text ignores it (built from the plain name).
_TICK = "✓ "

# Roll Analysis is the action, Roll Baseline the field it fills; the Library's own
# menu item runs the same action, so it reads these rather than restating them.
BATCH_ANALYSIS_TOOLTIP = (
    "Roll Analysis — measures the exposure of every loaded frame outside a scene and saves the average as this roll's baseline"
)
BATCH_ANALYSIS_DISABLED_TOOLTIP = "Open this roll first — Roll Analysis measures the files currently loaded."
FROM_FRAME_TOOLTIP = (
    "Use This Frame — save the current frame's bounds as this roll's baseline, in place of a measured "
    "average. Every frame outside a scene on Use Luma/Color Average follows it, including one loaded later."
)
FROM_FRAME_DISABLED_TOOLTIP = "Open this roll first — the baseline is written onto the files currently loaded."


class RollAnalysisSidebar(BaseSidebar):
    """
    Which roll's Roll Analysis baseline this frame's Use Luma/Color Average axes
    borrow: a searchable picker over every roll in your library, ticked once it has
    one. Picking a roll loads its baseline immediately -- there is no separate Apply.
    Reanalyze, beside the picker, runs Roll Analysis itself (the metering pass that
    fills the tick in) -- the same action the Library's own "Roll Analysis" offers,
    reachable here too since this is where you notice a roll has never been measured.
    Use This Frame, beside it, writes the current frame's own bounds there instead, for
    a roll that wants one chosen frame as its reference rather than an average. Both are
    enabled only for the loaded roll, since both act on the files currently open, not on
    whichever one this picker happens to show.
    """

    def _init_ui(self) -> None:
        self.roll_combo = SearchableGearCombo(placeholder="Search rolls…")
        self.roll_combo.setToolTip(wrap_tooltip("Picking a roll loads its saved baseline onto the loaded files."))
        self.reanalyze_btn = self._icon_action("fa5s.tachometer-alt", BATCH_ANALYSIS_TOOLTIP)
        self.from_frame_btn = self._icon_action("fa5s.crosshairs", FROM_FRAME_TOOLTIP)
        self.layout.addLayout(header_row(section_subheader("ROLLS"), self.reanalyze_btn, self.from_frame_btn))
        self.layout.addWidget(self.roll_combo)

        self.roll_status_hint = hint_label("", "muted")
        self.layout.addWidget(self.roll_status_hint)

        self.scenes_header = section_subheader("SCENES")
        self.layout.addWidget(self.scenes_header)
        self.scene_rows = QWidget()
        self._scene_rows_layout = QVBoxLayout(self.scene_rows)
        self._scene_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.scene_rows)
        self.scenes_hint = hint_label("Select frames in the Film Strip, then right-click › Scene › Group as Scene…", "muted")
        self.layout.addWidget(self.scenes_hint)

        self._roll_sync_key = None
        self._scene_sync_key = None
        self._refresh_rolls(force=True)
        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.roll_combo.selection_changed.connect(self._on_roll_picked)
        self.reanalyze_btn.clicked.connect(self.controller.request_batch_normalization)
        self.from_frame_btn.clicked.connect(self._on_from_frame_clicked)
        self.controller.session.files_changed.connect(self._refresh_scenes)
        self.sync_ui()

    def _on_roll_picked(self, roll_id: str) -> None:
        """Picking a roll is the whole action: it loads that roll's saved baseline
        onto the currently loaded files. A no-op if it has never been analyzed."""
        if roll_id:
            self.controller.apply_normalization_roll(roll_id)
        active_id = self.controller.state.active_roll_id
        self._update_roll_status_hint(active_id, roll_id)
        self._update_reanalyze_btn(active_id, roll_id)

    def _on_from_frame_clicked(self) -> None:
        active_id = self.controller.state.active_roll_id
        if active_id:
            self.controller.set_roll_baseline_from_frame(active_id)

    def _name_for_id(self, roll_id: str) -> str:
        entry = rolls.roll_for_id(self.controller.session.repo, roll_id)
        return entry["name"] if entry else roll_id

    def _refresh_rolls(self, *, force: bool = False) -> None:
        """
        Rebuilds the picker from every library roll, skipping a rebuild mid-search
        (SearchableGearCombo.is_editing) and one the roll set and selection don't need.
        The loaded roll is pinned first in the dropdown, ahead of the alphabetical
        rest, since it is the default choice.
        """
        if not force and self.roll_combo.is_editing():
            return
        repo = self.controller.session.repo
        active_id = self.controller.state.active_roll_id
        listed = rolls.all_rolls_sorted(repo)
        analyzed = {rid for rid, entry in listed if entry.get("normalization")}
        conf = self.state.config.process
        selected = active_id or ""
        if conf.roll_name:
            matching = next((rid for rid, entry in listed if entry.get("name") == conf.roll_name), None)
            if matching:
                selected = matching
        key = (tuple(rid for rid, _entry in listed), tuple(sorted(analyzed)), selected, active_id)
        if not force and key == self._roll_sync_key:
            return
        self._roll_sync_key = key
        ordered = [pair for pair in listed if pair[0] == active_id] + [pair for pair in listed if pair[0] != active_id]
        entries = [(f"{_TICK}{entry.get('name', '')}" if rid in analyzed else entry.get("name", ""), rid) for rid, entry in ordered]
        self.roll_combo.set_labeled_items(entries, selected, search_fn=lambda _label, item_id: self._name_for_id(item_id))
        self._update_roll_status_hint(active_id, selected)
        self._update_reanalyze_btn(active_id, selected)

    def _refresh_scenes(self) -> None:
        """One row per scene of the loaded roll: its mark, name, loaded frame count and
        whether it has a baseline. Hidden entirely outside a roll."""
        active_id = self.controller.state.active_roll_id
        scenes = rolls.roll_scenes(self.controller.session.repo, active_id)
        loaded = [f.get("scene") for f in self.controller.state.uploaded_files]
        key = (active_id, repr(scenes), repr(loaded))
        if key == self._scene_sync_key:
            return
        self._scene_sync_key = key
        while self._scene_rows_layout.count():
            item = self._scene_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for ordinal, (scene_id, entry) in enumerate(scenes, 1):
            count = sum(1 for s in loaded if s and s[1] == scene_id)
            tick = " · ✓" if entry.get("normalization") else ""
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            label = field_label(
                f'<span style="color:{scene_color(ordinal)}">●</span> {ordinal}  {entry["name"]} · {count_of(count, "frame")}{tick}'
            )
            row_layout.addWidget(label, 1)
            analyze = self._icon_action(
                "fa5s.tachometer-alt", f"Scene Analysis — measure “{entry['name']}” and give its frames their own baseline"
            )
            analyze.clicked.connect(lambda _=False, sid=scene_id: self.controller.request_scene_analysis(sid))
            select = self._icon_action("fa5s.object-group", f"Select the frames of “{entry['name']}” in the Film Strip")
            select.clicked.connect(lambda _=False, sid=scene_id: self.controller.select_scene_frames(sid))
            for btn in (analyze, select):
                btn.setEnabled(count > 0)
                row_layout.addWidget(btn)
            delete = self._icon_action("fa5s.trash", f"Delete “{entry['name']}” — its frames keep their edits and baseline")
            delete.clicked.connect(lambda _=False, sid=scene_id: prompt_delete_scene(self, self.controller, sid))
            row_layout.addWidget(delete)
            self._scene_rows_layout.addWidget(row)
        in_roll = active_id is not None
        self.scenes_header.setVisible(in_roll)
        self.scene_rows.setVisible(bool(scenes))
        self.scenes_hint.setVisible(in_roll and not scenes)

    def _update_roll_status_hint(self, active_id: Optional[str], selected_id: str) -> None:
        """Flags a baseline picked from a roll other than the one loaded."""
        if active_id and selected_id and selected_id != active_id:
            set_hint_kind(self.roll_status_hint, "warning")
            self.roll_status_hint.setText(f"Using “{self._name_for_id(selected_id)}”, a baseline saved for a different roll")
        else:
            self.roll_status_hint.setText("")

    def _update_reanalyze_btn(self, active_id: Optional[str], selected_id: str) -> None:
        """Both ways of filling a baseline write onto the loaded roll, same as the
        Library's own "Roll Analysis" -- grayed out otherwise, with the same
        explanation."""
        is_active = bool(selected_id) and selected_id == active_id
        for btn, on, off in (
            (self.reanalyze_btn, BATCH_ANALYSIS_TOOLTIP, BATCH_ANALYSIS_DISABLED_TOOLTIP),
            (self.from_frame_btn, FROM_FRAME_TOOLTIP, FROM_FRAME_DISABLED_TOOLTIP),
        ):
            btn.setEnabled(is_active)
            btn.setToolTip(wrap_tooltip(on if is_active else off))

    def sync_ui(self) -> None:
        self.block_signals(True)
        try:
            self._refresh_rolls()
            self._refresh_scenes()
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all buttons.
        """
        self.roll_combo.blockSignals(blocked)
