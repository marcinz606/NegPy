"""Move to Trash and Show in Folder, the one action behind every surface that offers each."""

from negpy.kernel.system.i18n import tr

from negpy.desktop.file_manager import show_in_folder
from negpy.desktop.session import composite_kind
from negpy.desktop.view.confirm import confirm_move_to_trash, warn_trash_composite

TRASH_LABEL = tr("Move to Trash…")
SHOW_LABEL = tr("Show in Folder")


def _targets(state) -> list[int]:
    return [i for i in (state.selected_indices or [state.selected_file_idx]) if 0 <= i < len(state.uploaded_files)]


def move_selected_to_trash(parent, controller) -> None:
    """Every frame in the selection must be a plain file: one composite blocks the whole action."""
    state = controller.session.state
    targets = _targets(state)
    if not targets:
        return
    if any(composite_kind(state.uploaded_files[i]) for i in targets):
        warn_trash_composite(parent)
        return
    if confirm_move_to_trash(parent, len(targets)):
        controller.move_frames_to_trash(targets)


def show_current_in_folder(controller) -> None:
    """The active frame only: not every file manager can select several files."""
    state = controller.session.state
    idx = state.selected_file_idx
    if not 0 <= idx < len(state.uploaded_files):
        return
    if not show_in_folder(state.uploaded_files[idx]["path"]):
        controller.set_status(tr("The frame's folder is gone"), 4000, kind="warning")
