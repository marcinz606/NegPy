from PyQt6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QLineEdit, QVBoxLayout

from negpy.desktop.view.styles.templates import field_label, hint_label, pin_button_box, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME


class RenameRollDialog(QDialog):
    """Renames a roll: its display name always, and -- for a folder roll, only if
    asked -- the actual folder on disk too. The checkbox starts unchecked every time;
    it is never remembered, since a disk rename is a bigger deal than a library-only
    one. A virtual roll has no folder, so it gets the same dialog without that row.
    """

    def __init__(self, current_name: str, parent=None, *, folder_backed: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Rename Roll")
        self.setMinimumWidth(340)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, THEME.space_xl)
        root.setSpacing(THEME.space_lg)

        root.addWidget(field_label("Name"))
        self.name_edit = QLineEdit(current_name)
        self.name_edit.setToolTip(wrap_tooltip("The roll's name in the library"))
        self.name_edit.selectAll()
        root.addWidget(self.name_edit)

        self.rename_folder_check = QCheckBox("Also rename the folder on disk")
        self.rename_folder_check.setToolTip(wrap_tooltip("Rename the actual folder as well, in place"))
        self.rename_folder_check.setVisible(folder_backed)
        root.addWidget(self.rename_folder_check)
        if folder_backed:
            root.addWidget(
                hint_label(
                    "Renames the actual folder, in place. Inside a cloud-sync folder (Dropbox, "
                    "iCloud, OneDrive), a sync client may see this as a delete and re-upload "
                    "rather than a rename.",
                    kind="warning",
                )
            )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        pin_button_box(buttons)
        root.addWidget(buttons)

        self.name_edit.setFocus()

    def name(self) -> str:
        return self.name_edit.text().strip()

    def rename_folder(self) -> bool:
        return not self.rename_folder_check.isHidden() and self.rename_folder_check.isChecked()
