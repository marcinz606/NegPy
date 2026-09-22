from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.view.confirm import confirm_assembly_mode, confirm_undiptych
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, wrap_tooltip

_NO_ROLL_HINT = "Half Frame is a roll-wide setting, and this isn't one roll. Open the roll itself to split its scans."


class HalfFrameSidebar(BaseSidebar):
    """
    Half-frame splitting for this roll: whether each scan becomes two frames, and the
    split geometry for the odd frame the batch detection gets wrong. Each roll
    remembers its own state.
    """

    def _init_ui(self) -> None:
        self.enable_btn = self._small_toggle(
            "mdi.view-split-vertical",
            "Half Frame Mode",
            self.controller.half_frame_mode_for_roll(self.state.active_roll_id),
            "",
        )
        self.layout.addWidget(self.enable_btn)

        self.adjust_btn = self._labeled_action(
            "mdi.tune-variant",
            " Adjust…",
            wrap_tooltip("Set this scan's crop and the gutter it splits at; its Apply picks what the result saves to"),
        )
        self.auto_btn = self._labeled_action(
            "fa5s.layer-group",
            " Detect All",
            wrap_tooltip("Re-find the crop and the gutter on every loaded scan"),
        )
        self.unsplit_btn = self._labeled_action(
            "fa5s.object-ungroup",
            " Unsplit",
            wrap_tooltip("Turn this diptych back into one plain scan, deleting both halves' edits"),
        )
        btn_row = QHBoxLayout()
        for btn in (self.adjust_btn, self.auto_btn, self.unsplit_btn):
            btn_row.addWidget(btn, 1)
        self.layout.addLayout(btn_row)

        self.hint = hint_label("", "warning")
        self.layout.addWidget(self.hint)

    def _connect_signals(self) -> None:
        self.enable_btn.toggled.connect(self._on_toggled)
        self.controller.half_frame_mode_changed.connect(self._follow_mode)
        self.adjust_btn.clicked.connect(self._on_adjust)
        self.auto_btn.clicked.connect(self.controller.auto_detect_all_half_frame_splits)
        self.unsplit_btn.clicked.connect(self._on_unsplit)
        self.sync_ui()

    def _on_toggled(self, checked: bool) -> None:
        """No editor pops up: past the confirmation, turning it on splits every loaded
        scan at its auto-detected gutter directly, and the odd frame it gets wrong is
        fixed afterward with Adjust. With nothing loaded there is nothing to split and
        nothing to confirm."""
        loaded = len(self.state.uploaded_files)
        if checked and loaded and not confirm_assembly_mode(self, "Half Frame", loaded):
            self._follow_mode(False)
            return
        self.controller.set_half_frame_mode(checked)
        if checked and loaded:
            self.controller.auto_detect_all_half_frame_splits()

    def _follow_mode(self, enabled: bool) -> None:
        """Follow the active roll's own toggle state. Signals are blocked because
        request_asset_discovery already applied it for this roll; letting toggled
        through would ask for it a second time and re-run discovery."""
        self.enable_btn.blockSignals(True)
        self.enable_btn.setChecked(enabled)
        self.enable_btn.blockSignals(False)
        self.sync_ui()

    def _on_adjust(self) -> None:
        """Open the half-frame rectangle editor on the current image; its own
        Apply ▾ picks what the result gets saved to."""
        path, file_hash = self.controller.current_base_file()
        if not path or not file_hash:
            return
        result = self.controller.open_half_frame_dialog(path, file_hash, selected_hashes=self.controller.selected_base_hashes())
        if result is not None:
            self.controller.reload_after_half_frame_change()

    def _on_unsplit(self) -> None:
        if confirm_undiptych(self):
            self.controller.request_undiptych()

    def sync_ui(self) -> None:
        """The toggle is a roll-wide fact -- one film type, split or not -- so it has
        nothing to apply to a batch with no single active roll (a library-wide search's
        mixed results, a restored session with no shared roll). Nothing in such a batch
        splits: the roll itself is where a scan becomes two frames."""
        has_roll = bool(self.state.active_roll_id)
        has_files = bool(self.state.uploaded_files)
        self.block_signals(True)
        try:
            self.enable_btn.setChecked(self.controller.half_frame_mode_for_roll(self.state.active_roll_id))
            self.enable_btn.setEnabled(has_roll)
            self.enable_btn.setToolTip(
                wrap_tooltip("Split each scan into two frames, edited and measured separately") if has_roll else wrap_tooltip(_NO_ROLL_HINT)
            )
            self.adjust_btn.setEnabled(has_roll and has_files)
            self.auto_btn.setEnabled(has_roll and has_files)
            self.unsplit_btn.setEnabled(self.controller.active_diptych() is not None)
            self.hint.setText("" if has_roll else _NO_ROLL_HINT)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.enable_btn, self.adjust_btn, self.auto_btn, self.unsplit_btn):
            w.blockSignals(blocked)
