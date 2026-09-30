"""The Contact Sheet dialog's worker: file facts before it opens, preview tiles while it is open."""

from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from negpy.desktop.workers.contact_sheet import ContactSheetPrepTask, ContactSheetPreviewTask, ContactSheetPreviewWorker
from negpy.domain.models import WorkspaceConfig
from negpy.services.export.contact_sheet_roll import FrameFacts, SheetFrame


def test_prepare_reads_every_frame_and_keeps_asset_order(tmp_path):
    paths = []
    for i, size in enumerate(((300, 200), (200, 300))):
        path = str(tmp_path / f"{i}.jpg")
        Image.new("RGB", size).save(path)
        paths.append(path)
    worker = ContactSheetPreviewWorker()
    results = []
    worker.prepared.connect(lambda generation, facts: results.append((generation, facts)))
    worker.prepare(ContactSheetPrepTask(4, ({"path": paths[0]}, {"path": paths[1]}, {"path": str(tmp_path / "missing.jpg")})))
    generation, facts = results[0]
    assert generation == 4
    assert [f.scan_size for f in facts] == [(300, 200), (200, 300), None]
    assert facts[2] == FrameFacts(birth_time=0.0)


def test_a_cancelled_prepare_stays_silent():
    worker = ContactSheetPreviewWorker()
    results = []
    worker.prepared.connect(lambda *args: results.append(args))
    worker.cancel(5)
    worker.prepare(ContactSheetPrepTask(5, ({"path": "/nowhere.jpg"},)))
    assert results == []


def test_render_emits_one_tile_per_frame_and_stops_when_cancelled():
    worker = ContactSheetPreviewWorker()
    worker._processor = MagicMock()
    tile = np.zeros((20, 30, 3), np.uint8)
    frames = tuple(SheetFrame({"path": "/s/a.tif", "name": f"a{i}", "hash": f"a{i}", "half": i + 1}, WorkspaceConfig()) for i in range(2))
    worker._processor.render_display_array.return_value = tile
    tiles = []
    worker.tile_ready.connect(lambda generation, index, t: tiles.append((generation, index)))
    worker.render(ContactSheetPreviewTask(8, frames))
    assert tiles == [(8, 0), (8, 1)]
    assert [c.kwargs["keep_source"] for c in worker._processor.render_display_array.call_args_list] == [True, False]
    assert all(c.kwargs["prefer_gpu"] is False for c in worker._processor.render_display_array.call_args_list)

    tiles.clear()
    worker._processor.render_display_array.reset_mock()
    worker.cancel(9)
    worker.render(ContactSheetPreviewTask(9, frames))
    assert tiles == []
