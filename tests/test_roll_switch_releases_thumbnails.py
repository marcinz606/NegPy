from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.services.assets.thumbnails import asset_thumbnail_key


def test_opening_another_roll_releases_the_previous_rolls_frame_caches() -> None:
    controller = MagicMock()
    controller.state = AppState()
    kept, gone = {"name": "a", "path": "/r2/a", "hash": "a"}, {"name": "b", "path": "/r1/b", "hash": "b"}
    controller.state.uploaded_files = [kept]
    for f in (kept, gone):
        controller.state.thumbnails[asset_thumbnail_key(f)] = object()
        controller.state.stale_thumbnails.add(asset_thumbnail_key(f))
        controller.state.embeddings[f["hash"]] = object()
        controller.state.source_exif[f["hash"]] = {}

    AppController._forget_unloaded_frames(controller)

    assert set(controller.state.thumbnails) == {asset_thumbnail_key(kept)}
    assert controller.state.stale_thumbnails == {asset_thumbnail_key(kept)}
    assert set(controller.state.embeddings) == {"a"}
    assert set(controller.state.source_exif) == {"a"}
