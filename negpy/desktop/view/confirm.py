from negpy.kernel.system.i18n import tr
from typing import Optional

from PyQt6.QtWidgets import QMessageBox

from negpy.desktop.view.styles.templates import box_toggle
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
    box.setWindowTitle(tr("Load Roll"))
    box.setText(tr("Load {value} from “{label}”?", value=count_of(n, "image"), label=label))
    box.setInformativeText(tr("They are hashed and thumbnailed on load, which takes a moment on a large roll."))
    remember = box_toggle(tr("Always Load Without Asking"))
    box.setCheckBox(remember)
    load = box.addButton(tr("Load"), QMessageBox.ButtonRole.AcceptRole)
    box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(load)
    box.exec()
    if box.clickedButton() is not load:
        return False
    if remember.isChecked():
        repo.save_global_setting("library_autoload_folders", True)
    return True


def confirm_unmetered_scan(parent, repo) -> bool:
    """Ask before a scan shoots at the camera's own exposure with no Meter reading and no exposure
    set in NegPy. Skippable via "Do Not Ask Again", persisted."""
    if repo.get_global_setting("scan_unmetered_confirm_dismissed", False):
        return True
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(tr("Scan"))
    box.setText(tr("Scan without metering?"))
    box.setInformativeText(
        tr(
            "The scan uses the exposure set on the camera. If the film base clips, the shadows of the print turn solid black. "
            "Click Cancel, then click Meter."
        )
    )
    remember = box_toggle(tr("Do Not Ask Again"))
    box.setCheckBox(remember)
    scan = box.addButton(tr("Scan"), QMessageBox.ButtonRole.AcceptRole)
    cancel = box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel)
    box.exec()
    if box.clickedButton() is not scan:
        return False
    if remember.isChecked():
        repo.save_global_setting("scan_unmetered_confirm_dismissed", True)
    return True


def warn_targets_affect_unbaked(parent, repo) -> None:
    """On a confirmed target change, advise that unbaked frames move with the new targets, and
    that baking a frame first keeps it fixed. Skippable via "Don't show again", persisted so a
    one-time retune does not re-prompt for every frame after."""
    if repo.get_global_setting("exposure_targets_warn_dismissed", False):
        return
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle(tr("Auto Density and Grade Targets"))
    box.setText(tr("Changing the targets affects every frame, including ones not yet baked."))
    box.setInformativeText(tr("Bake Auto Density and Grade on a frame before you change the targets to keep its density and grade fixed."))
    dismiss = box_toggle(tr("Do Not Show Again"))
    box.setCheckBox(dismiss)
    ok = box.addButton(tr("OK"), QMessageBox.ButtonRole.AcceptRole)
    box.setDefaultButton(ok)
    box.exec()
    if dismiss.isChecked():
        repo.save_global_setting("exposure_targets_warn_dismissed", True)


def confirm_unload_all(parent, roll_name: str | None) -> bool:
    """Ask before the Film Strip empties; *roll_name* is the open roll, which stays in the Library."""
    if roll_name is None:
        text = tr("Unload all frames from the Film Strip?")
        informative = tr("Your saved edits stay in the database.")
    else:
        text = tr("Unload all frames of “{roll_name}” from the Film Strip?", roll_name=roll_name)
        informative = tr("Your saved edits stay in the database. The roll stays in the Library.")
    return confirm_action(parent, tr("Unload All"), text, tr("Unload"), informative, destructive=False)


def confirm_unload(parent, *, count: int = 1) -> bool:
    """Ask the user to confirm removing image(s) from the session.

    Unloading only drops the frames from the current list — saved edits stay in the
    database keyed by content hash — but re-adding a large roll is tedious, so we gate
    it behind a prompt. Enter confirms (default button); Esc cancels.
    """
    verb = tr("Unload")
    if count > 1:
        title = tr("Unload Selected")
        text = tr("Unload the {value} from the session?", value=count_of(count, tr("selected image")))
    else:
        title = tr("Unload")
        text = tr("Unload this image from the session?")
    return confirm_action(
        parent, title, text, verb, tr("Your saved edits stay in the database. This only removes the frames from the list.")
    )


def confirm_delete_named(parent, kind: str, name: str, *, informative: str = "") -> bool:
    """Ask before deleting a named, user-created item — a work print, a roll, a
    flat-field profile. None of them are undoable and none can be re-derived from the
    frame, so each one asks first. Enter confirms; Esc cancels.
    """
    return confirm_action(
        parent,
        tr("Delete {kind}", kind=kind),
        tr("Delete the {value} “{name}”?", value=kind.lower(), name=name),
        tr("Delete"),
        informative,
    )


def prompt_delete_scene(parent, controller, scene_id: str) -> None:
    """The Film Strip menu and the Roll tab ask the same question before forgetting a scene."""
    entry = dict(rolls.roll_scenes(controller.session.repo, controller.state.active_roll_id)).get(scene_id)
    if entry and confirm_delete_named(
        parent, tr("Scene"), entry["name"], informative=tr("Its frames keep their edits and the baseline they carry.")
    ):
        controller.request_delete_scene(scene_id)


def confirm_delete_several(parent, kinds: str, names: list, *, informative: str = "") -> bool:
    """Ask before deleting several named items at once, selected together. Enter
    confirms; Esc cancels."""
    return confirm_action(
        parent,
        tr("Delete {value} {kinds}", value=len(names), kinds=kinds),
        tr("Delete these {value} {kinds}?", value=len(names), kinds=kinds.lower()),
        tr("Delete"),
        informative or "\n".join(names),
    )


def confirm_delete_mask(parent) -> bool:
    """Ask before deleting a single dodge/burn mask. Enter confirms; Esc cancels."""
    return confirm_action(parent, tr("Delete Mask"), tr("Delete this dodge/burn mask?"), tr("Delete"))


def confirm_reset_frames(parent, count: int, *, roll: bool = False) -> bool:
    """Ask before resetting several frames to their defaults at once.

    Each frame's reset is still an ordinary undo step, but reverting many frames at
    once in the wrong roll or selection is disruptive, so it asks first.
    """
    if roll:
        title = tr("Reset Roll to Defaults")
        text = tr("Reset all {value} in this roll to their default settings?", value=count_of(count, "frame"))
    else:
        title = tr("Reset Settings")
        text = tr("Reset {value} to their default settings?", value=count_of(count, "frame"))
    return confirm_action(parent, title, text, tr("Reset"), tr("Each frame's reset is a normal undo step, so you can revert it after."))


def confirm_reset_tab(parent, tab: str, count: int) -> bool:
    """Ask before one tab's whole set of cards goes back to defaults. One card's reset
    button is a single undo step and asks nothing; a tab's is every card at once."""
    return confirm_action(
        parent,
        tr("Reset {tab}", tab=tab),
        tr("Reset {value} on {tab} to their default settings?", value=count_of(count, "card"), tab=tab),
        tr("Reset"),
    )


def confirm_clear_heals(parent, count: int) -> bool:
    """Ask before wiping every manual heal/scratch on the frame.

    Unlike single-heal undo this is not step-recoverable, so it asks first.
    Enter confirms (default button); Esc cancels.
    """
    return confirm_action(
        parent,
        tr("Clear All Heals"),
        tr("Remove all {value} from this image?", value=count_of(count, tr("manual heal"))),
        tr("Clear"),
        tr("This removes every heal and scratch repair on this frame."),
    )


def confirm_clear_clones(parent, count: int) -> bool:
    return confirm_action(
        parent, tr("Clear All Clones"), tr("Remove all {value} from this image?", value=count_of(count, tr("clone stroke"))), tr("Clear")
    )


def confirm_delete_leak(parent) -> bool:
    """Ask before deleting a single light leak shape. Enter confirms; Esc cancels."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(tr("Delete Leak"))
    box.setText(tr("Delete this light leak?"))
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return box.exec() == QMessageBox.StandardButton.Yes


def confirm_clear_leaks(parent, count: int) -> bool:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(tr("Clear All Leaks"))
    box.setText(tr("Remove all {value} from this image?", value=count_of(count, tr("leak"))))
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
    box.setWindowTitle(tr("{mode} Mode", mode=mode))
    box.setText(tr("Turn {mode} Mode on for {value}?", mode=mode, value=count_of(count, tr("loaded frame"))))
    box.setInformativeText(tr("Every loaded frame is read and thumbnailed again. Your saved edits stay."))
    turn_on = box.addButton(tr("Turn On"), QMessageBox.ButtonRole.AcceptRole)
    box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(turn_on)
    box.exec()
    return box.clickedButton() is turn_on


def confirm_sensor_capture(parent) -> bool:
    """Ask before Capture from Camera fires the shutter. Film left in the holder still
    measures and saves a profile, a wrong one, so the rig is checked here."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(tr("Calibrate Sensor"))
    box.setText(tr("Shoot the red, green and blue exposures now?"))
    box.setInformativeText(
        tr(
            "Take the film out of the holder. Set the camera to Manual (M), at the ISO and aperture you scan with.\n\n"
            "NegPy lights each LED in turn and sets the shutter itself."
        )
    )
    capture = box.addButton(tr("Capture"), QMessageBox.ButtonRole.AcceptRole)
    box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(capture)
    box.exec()
    return box.clickedButton() is capture


def confirm_frame_merge(parent, title: str, counts: dict, skipped: list) -> Optional[bool]:
    """None on Cancel, else whether the sources go to the Trash. *counts* is frames per composite kind."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    one = sum(counts.values()) == 1
    box.setText(
        tr("Merge {value} into {value1}?", value=_merge_subject(counts), value1=tr("a TIFF negative") if one else tr("TIFF negatives"))
    )
    lines = [tr("Each frame becomes a linear TIFF negative beside its primary source and keeps its edit. The demosaic is baked in.")]
    if counts.get("stitch"):
        lines.append(tr("A stitch also bakes in its flat field and sensor correction; the TIFF cannot be unstitched."))
    if skipped:
        shown = skipped[:8]
        if len(skipped) > len(shown):
            shown.append(tr("…and {value} more", value=len(skipped) - len(shown)))
        lines.append(tr("Skipped:\n") + "\n".join(shown))
    box.setInformativeText("\n\n".join(lines))
    trash = box_toggle(
        tr("Move Source Files to the Trash"),
        True,
        tr("Move each merged frame's source files to the Trash, or keep the frame in the Film Strip beside its TIFF"),
    )
    box.setCheckBox(trash)
    merge = box.addButton(tr("Merge"), QMessageBox.ButtonRole.AcceptRole)
    box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(merge)
    box.exec()
    if box.clickedButton() is not merge:
        return None
    return trash.isChecked()


def _merge_subject(counts: dict) -> str:
    names = {"rgb": ("triplet", ""), "stitch": ("stitch", "stitches")}
    parts = [count_of(counts[k], *names[k]) for k in ("rgb", "stitch") if counts.get(k)]
    return " and ".join(parts) if parts else count_of(0, "frame")


def confirm_action(
    parent, title: str, text: str, verb: str, informative: str = "", *, destructive: bool = True, default_cancel: bool = False
) -> bool:
    """A confirmation whose accept button is named after the act, not Yes. Esc cancels; Enter
    confirms, or cancels with default_cancel for an act that wipes data for good."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning if destructive else QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    box.setInformativeText(informative)
    accept = box.addButton(verb, QMessageBox.ButtonRole.AcceptRole)
    cancel = box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel if default_cancel else accept)
    box.exec()
    return box.clickedButton() is accept


def confirm_remove_from_roll(parent, count: int, roll_name: str) -> bool:
    """Ask before frames leave a roll. Their files and edits stay."""
    text = (
        tr("Remove this frame from “{roll_name}”?", roll_name=roll_name)
        if count == 1
        else tr("Remove the {value} from “{roll_name}”?", value=count_of(count, tr("selected frame")), roll_name=roll_name)
    )
    return confirm_action(
        parent,
        tr("Remove from Roll"),
        text,
        tr("Remove"),
        tr("The files and their edits stay. Their scene, card locks and own edit in this roll are deleted."),
    )


def confirm_restore_removed(parent, count: int, roll_name: str) -> bool:
    """Ask before the frames removed from a folder roll come back."""
    return confirm_action(
        parent,
        tr("Restore Removed Frames"),
        tr("Bring back the {value} removed from “{roll_name}”?", value=count_of(count, "frame"), roll_name=roll_name),
        tr("Restore"),
        tr("They return to the roll with their edits."),
    )


def confirm_unfork_edit(parent) -> bool:
    """Ask before this roll's own edit for a frame goes and the frame returns to the
    edit every other roll it belongs to already shares."""
    return confirm_action(
        parent,
        tr("Use the Shared Edit Again"),
        tr("Drop this roll's own edit for this frame?"),
        tr("Use Shared Edit"),
        tr("Its independent edit is deleted. The frame goes back to the edit shared with every other roll."),
    )


def confirm_undiptych(parent) -> bool:
    """Ask before a half-frame diptych's two halves, and both their edits, go."""
    return confirm_action(
        parent,
        tr("Unsplit Diptych"),
        tr("Turn this diptych back into one plain frame?"),
        tr("Unsplit"),
        tr("Both halves' edits are deleted. Splitting the scan again starts from defaults."),
    )


def warn_invalid_roll_name(parent, title: str) -> None:
    """The one wording for a roll name the library cannot store, wherever it is typed."""
    QMessageBox.warning(parent, title, tr('A roll name cannot contain / \\ : * ? " < > | or start or end with a dot.'))


def confirm_move_to_trash(parent, count: int) -> bool:
    """Ask before frames' files leave the disk for the OS Trash."""
    text = (
        tr("Move the {value} to the Trash?", value=count_of(count, tr("selected frame")))
        if count > 1
        else tr("Move this frame to the Trash?")
    )
    return confirm_action(
        parent,
        tr("Move to Trash"),
        text,
        tr("Move to Trash"),
        tr("The source files and their sidecars go to the Trash. The edits stay in the database, so a restored file keeps its edit."),
    )


def warn_trash_composite(parent) -> None:
    """A frame made from several files, or one file split in two, does not go to the Trash."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle(tr("Move to Trash"))
    box.setText(tr("Stitches, HDR merges, trichromes and half-frame scans cannot go to the Trash."))
    box.setInformativeText(
        tr(
            "Dissolve them first: Unstitch, Unmerge Exposures, Unsplit Diptych, "
            "or turn off Trichrome Mode or Half Frame Mode on the Roll tab. Then move their files to the Trash."
        )
    )
    box.setStandardButtons(QMessageBox.StandardButton.Close)
    box.exec()
