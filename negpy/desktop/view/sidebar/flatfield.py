import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
)

from negpy.desktop.view.confirm import confirm_delete_named
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, hint_label
from negpy.desktop.view.widgets.file_dialogs import last_open_folder, pick_start_dir

_NONE_LABEL = "— None —"
_FILE_FILTER = "Reference images (*.dng *.tif *.tiff *.cr2 *.cr3 *.nef *.arw *.raf *.rw2 *.jpg *.jpeg *.png);;All files (*)"


class FlatFieldSidebar(BaseSidebar):
    """
    Flat-field / falloff correction. Manages named reference profiles (the bare
    light-source scan) and a per-image enable toggle.
    """

    def _init_ui(self) -> None:
        row = QHBoxLayout()
        row.addWidget(field_label("Profile"))
        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip("Saved flat-field reference profiles (scan of the bare light source)")
        row.addWidget(self.profile_combo, 1)

        self.add_btn = self._icon_action("fa5s.plus", "Pick a reference image and save it as a named profile")
        self.delete_btn = self._icon_action("fa5s.trash", "Remove the selected profile")
        row.addWidget(self.add_btn)
        row.addWidget(self.delete_btn)
        self.layout.addLayout(row)

        self.hint = hint_label("Add a scan of the bare light source to enable.")
        self.layout.addWidget(self.hint)

        self.enable_btn = self._small_toggle(
            "fa5s.lightbulb",
            "Apply Flat Field",
            False,
            "Apply the active flat-field reference to this image",
        )
        self.layout.addWidget(self.enable_btn)

        self.layout.addStretch()
        self._refresh_profiles()

    def _connect_signals(self) -> None:
        self.enable_btn.toggled.connect(self.controller.set_flatfield_enabled)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        self.add_btn.clicked.connect(self._on_add)
        self.delete_btn.clicked.connect(self._on_delete)
        self.sync_ui()

    def _refresh_profiles(self) -> None:
        # Preserve the caller's block state: unblocking here would let sync_ui's setCurrentIndex
        # re-fire _on_profile_selected and loop into update_config.
        from negpy.services.assets.flatfield import FlatFieldProfiles

        prev = self.profile_combo.signalsBlocked()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem(_NONE_LABEL, "")
        for profile_id, name in FlatFieldProfiles.list_profiles():
            self.profile_combo.addItem(name, profile_id)
        self.profile_combo.blockSignals(prev)

    def _on_profile_selected(self, _idx: int) -> None:
        profile_id = self.profile_combo.currentData() or ""
        active = self.controller.session.repo.get_global_setting("flatfield_active_profile") or ""
        if profile_id == active:
            return
        self.controller.set_active_flatfield_profile(profile_id)
        self.sync_ui()

    def _on_add(self) -> None:
        start = pick_start_dir(last_open_folder(self.controller.session.repo))
        path, _ = QFileDialog.getOpenFileName(self, "Select flat-field reference", start, _FILE_FILTER)
        if not path:
            return
        default_name = os.path.splitext(os.path.basename(path))[0]
        name, ok = QInputDialog.getText(self, "Save Flat-Field Profile", "Profile name:", text=default_name)
        if ok and name:
            # save_flatfield_profile decodes the reference RAW to bake the gain, a brief blocking beat
            # on the GUI thread, so show a wait cursor.
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.controller.save_flatfield_profile(name, path)
            finally:
                QApplication.restoreOverrideCursor()
            self._refresh_profiles()
            self.sync_ui()

    def _on_delete(self) -> None:
        profile_id = self.profile_combo.currentData()
        if profile_id and confirm_delete_named(
            self,
            "Flat-Field Profile",
            self.profile_combo.currentText(),
            informative="Every frame using it loses its correction; the baked gain map cannot be recovered.",
        ):
            self.controller.delete_flatfield_profile(profile_id)
            self._refresh_profiles()
            self.sync_ui()

    def sync_ui(self) -> None:
        conf = self.state.config.flatfield
        active = self.controller.session.repo.get_global_setting("flatfield_active_profile") or ""

        self.block_signals(True)
        try:
            self._refresh_profiles()
            idx = self.profile_combo.findData(active)
            self.profile_combo.setCurrentIndex(idx if idx >= 0 else 0)

            self.enable_btn.setChecked(conf.apply)
            self.enable_btn.setEnabled(bool(conf.profile_id))
            self.hint.setVisible(not conf.profile_id)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.enable_btn, self.profile_combo, self.add_btn, self.delete_btn):
            w.blockSignals(blocked)
