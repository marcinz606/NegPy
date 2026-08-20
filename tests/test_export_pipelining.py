"""Batch-export pipelining: the finisher thread encodes and writes the previous
render while the next file renders, and the prefetcher prepares the next source
into the ImageProcessor handoff slot."""

import threading
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import tifffile

from negpy.desktop.workers.export import ExportTask, ExportWorker
from negpy.domain.models import WorkspaceConfig, preset_from_export_config
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _preset(tmp_path):
    preset = preset_from_export_config(DEFAULT_WORKSPACE_CONFIG.export)
    return replace(preset, output_path=str(tmp_path))


def _task(tmp_path, name: str, **kwargs) -> ExportTask:
    return ExportTask(
        file_info={"name": name, "path": str(tmp_path / name), "hash": name},
        params=DEFAULT_WORKSPACE_CONFIG,
        export_settings=_preset(tmp_path),
        **kwargs,
    )


def _worker(render=(np.zeros((4, 4, 3), np.float32), "sRGB"), encode=(b"JPGDATA", "jpg")):
    worker = ExportWorker()
    proc = MagicMock()
    proc.render_export.return_value = render
    proc.encode_export.return_value = encode
    worker._processor = proc
    return worker, proc


def test_finisher_writes_rendered_file_before_finished(tmp_path) -> None:
    worker, proc = _worker()
    written_at_finish: list[list] = []
    worker.finished.connect(lambda: written_at_finish.append(list(tmp_path.glob("*.jpg"))))

    worker.run_batch([_task(tmp_path, "a.cr2")])

    assert len(written_at_finish) == 1
    assert len(written_at_finish[0]) == 1
    assert written_at_finish[0][0].read_bytes() == b"JPGDATA"


def test_encode_runs_off_the_render_thread(tmp_path) -> None:
    worker, proc = _worker()
    idents: list[int] = []
    proc.encode_export.side_effect = lambda *a, **k: (idents.append(threading.get_ident()), (b"X", "jpg"))[1]

    worker.run_batch([_task(tmp_path, "a.cr2")])

    assert idents and idents[0] != threading.get_ident()


def test_encode_error_surfaces_and_batch_continues(tmp_path) -> None:
    worker, proc = _worker(encode=(None, "encode boom"))
    errors: list[str] = []
    finished: list[bool] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run_batch([_task(tmp_path, "a.cr2"), _task(tmp_path, "b.cr2")])

    assert errors == ["encode boom", "encode boom"]
    assert finished == [True]
    assert list(tmp_path.glob("*.jpg")) == []


def test_cancel_still_writes_the_rendered_file(tmp_path) -> None:
    worker, proc = _worker()

    def _render(*_a, **_k):
        worker.cancel()
        return (np.zeros((4, 4, 3), np.float32), "sRGB")

    proc.render_export.side_effect = _render
    cancelled: list[bool] = []
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.run_batch([_task(tmp_path, "a.cr2"), _task(tmp_path, "b.cr2")])

    assert cancelled == [True]
    assert proc.render_export.call_count == 1
    assert len(list(tmp_path.glob("*.jpg"))) == 1


def test_prefetch_skips_a_diptych_next_task(tmp_path) -> None:
    worker, proc = _worker(render=(None, "render failed"))
    cfg = DEFAULT_WORKSPACE_CONFIG
    diptych = _task(tmp_path, "b.cr2", diptych=(cfg, cfg))

    worker.run_batch([_task(tmp_path, "a.cr2"), diptych])

    proc.prefetch_export_source.assert_not_called()


def _real_processor_and_source(tmp_path):
    from negpy.services.rendering.image_processor import ImageProcessor

    arr = (np.random.default_rng(0).random((64, 96, 3)) * 40000 + 8000).astype(np.uint16)
    path = tmp_path / "s.tif"
    tifffile.imwrite(path, arr, photometric="rgb")
    return ImageProcessor(), str(path)


def test_prefetch_fills_slot_and_prepare_serves_from_it(tmp_path) -> None:
    proc, path = _real_processor_and_source(tmp_path)
    cfg = WorkspaceConfig()

    proc.prefetch_export_source(path, cfg, "hash1")
    assert proc._prepare_slot is not None

    calls: list[int] = []
    orig = proc._prepare_export_source_locked
    proc._prepare_export_source_locked = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]

    buf, _cs, _token = proc._prepare_export_source(path, cfg, "hash1")
    assert calls == []
    assert buf is proc._prepare_slot[1][0]

    # A different source hash misses the slot and computes the same pixels.
    buf2, _cs2, _token2 = proc._prepare_export_source(path, cfg, "hash2")
    assert calls == [1]
    assert np.array_equal(buf, buf2)

    proc.release_source_cache()
    assert proc._prepare_slot is None


def test_prefetch_failure_is_swallowed(tmp_path) -> None:
    from negpy.services.rendering.image_processor import ImageProcessor

    proc = ImageProcessor()
    proc.prefetch_export_source(str(tmp_path / "missing.tif"), WorkspaceConfig(), "h")
    assert proc._prepare_slot is None
