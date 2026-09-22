from PyQt6.QtWidgets import QCheckBox, QMessageBox
from negpy.kernel.system.text import count_of
from negpy.services.assets import rolls


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
    box.setText(f"Load {count_of(n, 'image')} from “{label}”?")
    box.setInformativeText("They are hashed and thumbnailed on load, which takes a moment on a large roll.")
    remember = QCheckBox("Always load without asking")
    box.setCheckBox(remember)
    load = box.addButton("Load", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(load)
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
        text = f"Unload the {count_of(count, 'selected image')} from the session?"
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


def prompt_delete_scene(parent, controller, scene_id: str) -> None:
    """The Film Strip menu and the Roll tab ask the same question before forgetting a scene."""
    entry = dict(rolls.roll_scenes(controller.session.repo, controller.state.active_roll_id)).get(scene_id)
    if entry and confirm_delete_named(
        parent, "Scene", entry["name"], informative="Its frames keep their edits and the baseline they carry."
    ):
        controller.request_delete_scene(scene_id)


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


def confirm_reset_frames(parent, count: int, *, roll: bool = False) -> bool:
    """Ask before resetting several frames to their defaults at once.

    Each frame's reset is still an ordinary undo step, but reverting many frames at
    once in the wrong roll or selection is disruptive, so it is gated like Clear All.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    if roll:
        box.setWindowTitle("Reset Roll to Defaults")
        box.setText(f"Reset all {count_of(count, 'frame')} in this roll to their default settings?")
    else:
        box.setWindowTitle("Reset Settings")
        box.setText(f"Reset {count_of(count, 'frame')} to their default settings?")
    box.setInformativeText("Each frame's reset is a normal undo step, so you can revert it after.")
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return box.exec() == QMessageBox.StandardButton.Yes


def confirm_reset_tab(parent, tab: str, count: int) -> bool:
    """Ask before one tab's whole set of cards goes back to defaults. One card's reset
    button is a single undo step and asks nothing; a tab's is every card at once."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(f"Reset {tab}")
    box.setText(f"Reset {count_of(count, 'card')} on {tab} to their default settings?")
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
    box.setText(f"Remove all {count_of(count, 'manual heal')} from this image?")
    box.setInformativeText("Every heal and scratch repair placed on this frame will be removed.")
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return box.exec() == QMessageBox.StandardButton.Yes


def confirm_assembly_mode(parent, mode: str, count: int) -> bool:
    """Ask before Trichrome or Half Frame mode goes on.

    Turning one on regroups or splits every loaded scan, so the whole roll is read and
    thumbnailed again. Nothing is lost, but it is a long beat to start by accident.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(f"{mode} Mode")
    box.setText(f"Turn {mode} Mode on for {count_of(count, 'loaded frame')}?")
    box.setInformativeText("Every loaded frame is read and thumbnailed again. Your saved edits stay.")
    turn_on = box.addButton("Turn On", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(turn_on)
    box.exec()
    return box.clickedButton() is turn_on


def _confirm_with_verb(parent, title: str, text: str, informative: str, verb: str) -> bool:
    """A destructive confirmation whose accept button is named after the act, not Yes.
    Enter confirms; Esc cancels."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    box.setInformativeText(informative)
    accept = box.addButton(verb, QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(accept)
    box.exec()
    return box.clickedButton() is accept


def confirm_unfork_edit(parent) -> bool:
    """Ask before this roll's own edit for a frame goes and the frame returns to the
    edit every other roll it belongs to already shares."""
    return _confirm_with_verb(
        parent,
        "Use the Shared Edit Again",
        "Drop this roll's own edit for this frame?",
        "Its independent edit is deleted. The frame goes back to the edit shared with every other roll.",
        "Use Shared Edit",
    )


def confirm_undiptych(parent) -> bool:
    """Ask before a half-frame diptych's two halves, and both their edits, go."""
    return _confirm_with_verb(
        parent,
        "Unsplit Diptych",
        "Turn this diptych back into one plain frame?",
        "Both halves' edits are deleted. Splitting the scan again starts from defaults.",
        "Unsplit",
    )


def warn_invalid_roll_name(parent, title: str) -> None:
    """The one wording for a roll name the library cannot store, wherever it is typed."""
    QMessageBox.warning(parent, title, 'A roll name cannot contain / \\ : * ? " < > | or start or end with a dot.')
