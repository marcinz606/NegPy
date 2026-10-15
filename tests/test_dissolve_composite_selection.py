from unittest.mock import MagicMock

import pytest

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState


@pytest.mark.parametrize(
    "method,composite",
    [
        ("request_unmerge_hdr", {"hdr_paths": ["/r/b.tif"]}),
        ("request_unstitch", {"stitch_paths": ["/r/b.tif"]}),
    ],
)
def test_dissolving_the_active_composite_leaves_no_frame_selected(method, composite) -> None:
    controller = MagicMock()
    controller.state = AppState()
    controller.session.state = controller.state
    controller.state.uploaded_files = [
        {"name": "c", "path": "/r/a.tif", "hash": "c", **composite},
        {"name": "z", "path": "/r/z.tif", "hash": "z"},
    ]
    controller.state.selected_file_idx = 0
    controller.state.selected_indices = [0, 1]
    controller._drop_dissolved_composite = lambda idx: AppController._drop_dissolved_composite(controller, idx)

    getattr(AppController, method)(controller)

    assert controller.state.selected_file_idx == -1
    assert controller.state.selected_indices == [0]  # z moved up a row
