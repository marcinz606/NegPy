"""A finished render's thumbnail belongs to the frame it was rendered from.

Renders are asynchronous and a full-res decode takes seconds, so one can land after the
user has clicked on to the next frame. Keying those pixels by the current selection files
one frame's picture under another frame's thumbnail — the filmstrip then shows the wrong
image until that cell is clicked and re-rendered.

The render memo already guards this way (`source_hash == current_file_hash`); the
thumbnail path did not.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.controller import AppController
from negpy.services.assets.thumbnails import asset_thumbnail_key

A = {"name": "a.nef", "path": "/f/a.nef", "hash": "hash-a"}
B = {"name": "b.nef", "path": "/f/b.nef", "hash": "hash-b"}


def _controller(selected_idx, metrics):
    c = MagicMock()
    c.state.uploaded_files = [A, B]
    c.state.selected_file_idx = selected_idx
    c.state.current_file_path = "/f/b.nef"
    c.state.current_file_hash = "hash-b"
    c.state.last_metrics = metrics
    c.display_transform_params.return_value = ("sRGB", None, False)
    # The selector is the thing under test too — bind the real one to the stub.
    c._asset_for_render = lambda metrics: AppController._asset_for_render(c, metrics)
    return c


def _emitted_key(controller):
    controller.thumbnail_update_requested.emit.assert_called_once()
    return controller.thumbnail_update_requested.emit.call_args.args[0].file_hash


class Attribution(unittest.TestCase):
    def test_a_late_render_updates_the_frame_it_came_from(self):
        """A is rendering; the user clicks B; A's render lands. Its pixels are A's."""
        metrics = {"base_positive": np.zeros((4, 4, 3), np.float32), "source_hash": "hash-a"}
        controller = _controller(selected_idx=1, metrics=metrics)  # B is selected now

        AppController._update_thumbnail_from_state(controller)

        self.assertEqual(_emitted_key(controller), asset_thumbnail_key(A))

    def test_the_ordinary_case_is_unchanged(self):
        metrics = {"base_positive": np.zeros((4, 4, 3), np.float32), "source_hash": "hash-b"}
        controller = _controller(selected_idx=1, metrics=metrics)

        AppController._update_thumbnail_from_state(controller)

        self.assertEqual(_emitted_key(controller), asset_thumbnail_key(B))

    def test_without_a_source_hash_nothing_is_written(self):
        """Originally this fell back to the selection, on the grounds that the
        active_file_changing caller snapshots the outgoing frame, which is still selected.
        That cannot tell "outgoing frame still selected" from "a different folder is open
        now", and the second case files one frame's picture under another's — found on
        disk, correlating 0.10 with its own file. The render worker always sets
        source_hash, so refusing here costs nothing real."""
        metrics = {"base_positive": np.zeros((4, 4, 3), np.float32)}
        controller = _controller(selected_idx=0, metrics=metrics)

        AppController._update_thumbnail_from_state(controller)

        controller.thumbnail_update_requested.emit.assert_not_called()

    def test_a_render_of_an_unloaded_frame_updates_nothing(self):
        """Its asset is gone from the session — better to skip than to key by the
        selection, which is how the wrong picture got cached in the first place."""
        metrics = {"base_positive": np.zeros((4, 4, 3), np.float32), "source_hash": "hash-gone"}
        controller = _controller(selected_idx=1, metrics=metrics)
        controller.state.uploaded_files = []

        AppController._update_thumbnail_from_state(controller)

        controller.thumbnail_update_requested.emit.assert_not_called()


class Selector(unittest.TestCase):
    def test_matches_on_hash_not_position(self):
        controller = _controller(selected_idx=0, metrics={})
        self.assertIs(AppController._asset_for_render(controller, {"source_hash": "hash-b"}), B)
        self.assertIs(AppController._asset_for_render(controller, {"source_hash": "hash-a"}), A)
