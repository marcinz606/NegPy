from negpy.kernel.system.i18n import tr
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
)

from negpy.desktop.view.confirm import confirm_delete_named
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, field_label, hint_label
from negpy.desktop.view.widgets.file_dialogs import last_open_folder, pick_start_dir
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.infrastructure.loaders.helpers import get_supported_raw_wildcards

_NONE_LABEL = tr("None")
_FILE_FILTER = tr("Reference images ({wildcards});;All files (*)", wildcards=get_supported_raw_wildcards())


class FlatFieldSidebar(BaseSidebar):
    """
    Flat-field / falloff correction. Manages named reference profiles (the bare
    light-source scan); the profile and the toggle are roll-wide, since one roll is
    scanned under one light.
    """

    def _init_ui(self) -> None:
        row = QHBoxLayout()
        row.addWidget(field_label(tr("Profile")))
        self.profile_combo = StepCombo()
        self.profile_combo.setToolTip(tr("Saved shots of the bare light, used to even out the light."))
        row.addWidget(self.profile_combo, 1)

        self.enable_btn = self._small_toggle(
            "fa5s.lightbulb",
            "",
            False,
            tr("Apply Flat Field: even out the light on this roll with the selected profile."),
        )
        self.enable_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        row.addWidget(self.enable_btn)

        self.check_btn = self._small_toggle(
            "fa5s.eye",
            "",
            False,
            tr("Check Flat Field: show how well the profile evens out its own reference, where even gray is good."),
        )
        self.check_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        row.addWidget(self.check_btn)

        self.add_btn = self._icon_action("fa5s.plus", tr("Pick a reference image and save it as a named profile"))
        self.delete_btn = self._icon_action("fa5s.trash", tr("Remove the selected profile"))
        # add_btn and delete_btn go on the card's FLAT FIELD CORRECTION header (ControlsPanel).
        self.layout.addLayout(row)

        self.hint = hint_label(tr("Add a scan of the bare light source to enable."))
        self.layout.addWidget(self.hint)
        self._refresh_profiles()

    def _connect_signals(self) -> None:
        self.enable_btn.toggled.connect(self.controller.set_flatfield_enabled)
        self.check_btn.toggled.connect(lambda checked: self.controller.toggle_flatfield_peek(force=checked))
        self.controller.flatfield_peek_changed.connect(self._on_peek_changed)
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
        if profile_id == self.state.config.flatfield.profile_id:
            return
        self.controller.set_active_flatfield_profile(profile_id)
        self.sync_ui()

    def _on_add(self) -> None:
        start = pick_start_dir(last_open_folder(self.controller.session.repo))
        path, _ = QFileDialog.getOpenFileName(self, tr("Select Flat Field Reference"), start, _FILE_FILTER)
        if not path:
            return
        default_name = os.path.splitext(os.path.basename(path))[0]
        name, ok = QInputDialog.getText(self, tr("Save Flat Field Profile"), tr("Profile name:"), text=default_name)
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
            tr("Flat Field Profile"),
            self.profile_combo.currentText(),
            informative=tr("Every frame using it loses its correction, and the profile cannot be recovered."),
        ):
            self.controller.delete_flatfield_profile(profile_id)
            self._refresh_profiles()
            self.sync_ui()

    def sync_ui(self) -> None:
        conf = self.state.config.flatfield

        self.block_signals(True)
        try:
            self._refresh_profiles()
            # A profile this machine does not have reads as none; nothing renders from it.
            idx = self.profile_combo.findData(conf.profile_id) if conf.profile_id else -1
            self.profile_combo.setCurrentIndex(max(idx, 0))

            self.enable_btn.setChecked(conf.apply and idx > 0)
            self.enable_btn.setEnabled(idx > 0)
            self.hint.setVisible(idx <= 0)
        finally:
            self.block_signals(False)

    def _on_peek_changed(self, active: bool) -> None:
        self.check_btn.blockSignals(True)
        self.check_btn.setChecked(active)
        self.check_btn.blockSignals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.enable_btn, self.check_btn, self.profile_combo, self.add_btn, self.delete_btn):
            w.blockSignals(blocked)
