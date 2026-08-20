"""A render whose frame is no longer open must not be filed under whatever is selected now.

`_update_thumbnail_from_state` runs on file switch and after an export — not only
from the render that produced the buffer. So `state.last_metrics` can hold a render whose
frame has since left `uploaded_files`: opening a different folder replaces the list, and a
merge or unmerge swaps frames for a composite. Resolving that by falling back to
`selected_file_idx` files one frame's picture under another frame's key, and the write is
persisted, so it survives restarts until that frame is rendered again.
"""

from unittest.mock import MagicMock

from negpy.desktop.controller import AppController


def _controller(files, selected):
    ctrl = MagicMock()
    ctrl.state.uploaded_files = files
    ctrl.state.selected_file_idx = selected
    return ctrl


A = {"hash": "hash-a", "path": "/x/a.nef"}
B = {"hash": "hash-b", "path": "/x/b.nef"}


def test_a_render_is_attributed_to_its_own_frame():
    ctrl = _controller([A, B], 1)
    assert AppController._asset_for_render(ctrl, {"source_hash": "hash-a"}) is A


def test_a_render_whose_frame_has_closed_is_not_attributed_to_anyone():
    """The reported bug: a slide-08 frame's render cached under a slide-07 frame's key,
    correlation 0.10 against its own file, persisted to disk."""
    ctrl = _controller([A, B], 1)
    assert AppController._asset_for_render(ctrl, {"source_hash": "hash-of-a-closed-folder"}) is None


def test_metrics_without_an_identity_are_not_attributed():
    ctrl = _controller([A, B], 1)
    assert AppController._asset_for_render(ctrl, {}) is None


def test_a_composite_is_matched_by_its_own_suffixed_hash():
    """Merges and half-frames carry suffixed hashes; source_hash is current_file_hash, so
    they match directly and must not be mistaken for a miss."""
    composite = {"hash": "digest#hdr", "path": "/x/a.nef", "hdr_paths": ("/x/b.nef",)}
    ctrl = _controller([composite], 0)
    assert AppController._asset_for_render(ctrl, {"source_hash": "digest#hdr"}) is composite
