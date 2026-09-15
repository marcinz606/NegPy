from PyQt6.QtWidgets import QCheckBox, QMessageBox


def confirm_load_roll(parent, repo, image_count: int, label: str) -> bool:
    """Ask before hashing and thumbnailing a folder's images into the session.

    Skippable via "Always load without asking", persisted so importing a library
    full of rolls one at a time does not re-prompt for each.
    """
    if repo.get_global_setting("library_autoload_folders", False):
        return True

    n = image_count
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Load Roll")
    box.setText(f"Load {n} image{'s' if n != 1 else ''} from “{label}”?")
    box.setInformativeText("They are hashed and thumbnailed on load, which takes a moment on a large roll.")
    remember = QCheckBox("Always load without asking")
    box.setCheckBox(remember)
    load = box.addButton("Load", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.exec()
    if box.clickedButton() is not load:
        return False
    if remember.isChecked():
        repo.save_global_setting("library_autoload_folders", True)
    return True


def confirm_unload(parent, *, clear_all: bool = False, count: int = 1) -> bool:
    """Ask the user to confirm removing image(s) from the session.

    Unloading only drops the frames from the current list — saved edits stay in the
    database keyed by content hash — but re-adding a large roll is tedious, and an
    accidental Clear All is destructive to the working set, so we gate it behind a
    prompt. Enter confirms (default button); Esc cancels.
    """
    if clear_all:
        title = "Clear All"
        text = "Remove all loaded images from the session?"
    elif count > 1:
        title = "Unload Selected"
        text = f"Unload the {count} selected images from the session?"
    else:
        title = "Unload"
        text = "Unload this image from the session?"

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    box.setInformativeText("Your saved edits stay in the database — this only removes the frames from the list.")
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return box.exec() == QMessageBox.StandardButton.Yes


def confirm_delete_named(parent, kind: str, name: str, *, informative: str = "") -> bool:
    """Ask before deleting a named, user-created item — a work print, a roll, a
    flat-field profile. None of them are undoable and none can be re-derived from the
    frame, so each one is gated like Clear All. Enter confirms; Esc cancels.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(f"Delete {kind}")
    box.setText(f"Delete the {kind.lower()} “{name}”?")
    if informative:
        box.setInformativeText(informative)
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return box.exec() == QMessageBox.StandardButton.Yes


def confirm_delete_several(parent, kind: str, names: list, *, informative: str = "") -> bool:
    """Ask before deleting several named items at once, selected together. Enter
    confirms; Esc cancels."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(f"Delete {len(names)} {kind}s")
    box.setText(f"Delete these {len(names)} {kind.lower()}s?")
    box.setInformativeText(informative or "\n".join(names))
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return box.exec() == QMessageBox.StandardButton.Yes


def confirm_delete_mask(parent) -> bool:
    """Ask before deleting a single dodge/burn mask. Enter confirms; Esc cancels."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Delete Mask")
    box.setText("Delete this dodge/burn mask?")
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return box.exec() == QMessageBox.StandardButton.Yes


def confirm_clear_heals(parent, count: int) -> bool:
    """Ask before wiping every manual heal/scratch on the frame.

    Unlike single-heal undo this is not step-recoverable, so gate it like the
    session Clear All. Enter confirms (default button); Esc cancels.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Clear All Heals")
    box.setText(f"Remove all {count} manual heal{'s' if count != 1 else ''} from this image?")
    box.setInformativeText("Every heal and scratch repair placed on this frame will be removed.")
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return box.exec() == QMessageBox.StandardButton.Yes
