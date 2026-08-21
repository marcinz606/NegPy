"""Batch-export pipelining: the finisher thread encodes and writes the previous
render while the next file renders, and the prefetcher prepares the next source
into the ImageProcessor handoff slot."""

import threading
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import tifffile

from negpy.desktop.workers.export import ExportTask, ExportWorker
from negpy.domain.models import ExportFormat, WorkspaceConfig, preset_from_export_config
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _preset(tmp_path, **overrides):
    preset = preset_from_export_config(DEFAULT_WORKSPACE_CONFIG.export)
    return replace(preset, output_path=str(tmp_path), **overrides)


def _task(tmp_path, name: str, preset=None, **kwargs) -> ExportTask:
    return ExportTask(
        file_info={"name": name, "path": str(tmp_path / name), "hash": name},
        params=DEFAULT_WORKSPACE_CONFIG,
        export_settings=preset if preset is not None else _preset(tmp_path),
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


def test_empty_batch_finishes_cleanly() -> None:
    worker, proc = _worker()
    finished: list[bool] = []
    errors: list[str] = []
    cancelled: list[bool] = []
    worker.finished.connect(lambda: finished.append(True))
    worker.error.connect(errors.append)
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.run_batch([])

    assert finished == [True]
    assert errors == []
    assert cancelled == []


def test_progress_emitted_per_task_in_order(tmp_path) -> None:
    worker, proc = _worker(render=(None, "skip"))
    seen: list[tuple] = []
    worker.progress.connect(lambda c, t, n: seen.append((c, t, n)))

    worker.run_batch([_task(tmp_path, "a.cr2"), _task(tmp_path, "b.cr2")])

    assert seen == [(1, 2, "a"), (2, 2, "b")]


def test_render_error_skips_file_but_batch_continues(tmp_path) -> None:
    worker, proc = _worker()
    proc.render_export.side_effect = [(None, "render boom"), (np.zeros((4, 4, 3), np.float32), "sRGB")]
    errors: list[str] = []
    finished: list[bool] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run_batch([_task(tmp_path, "a.cr2"), _task(tmp_path, "b.cr2")])

    assert errors == ["render boom"]
    assert finished == [True]
    assert len(list(tmp_path.glob("*.jpg"))) == 1


def test_write_error_surfaces_and_batch_continues(tmp_path, monkeypatch) -> None:
    import negpy.desktop.workers.export as export_mod

    worker, proc = _worker()

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(export_mod.os, "replace", _boom)
    errors: list[str] = []
    finished: list[bool] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run_batch([_task(tmp_path, "a.cr2"), _task(tmp_path, "b.cr2")])

    assert errors == ["disk full", "disk full"]
    assert finished == [True]
    assert list(tmp_path.glob("*.jpg")) == []
    assert list(tmp_path.glob("*.part")) == []  # tmp file cleaned up on failure


def test_unexpected_exception_aborts_batch_with_error(tmp_path) -> None:
    worker, proc = _worker()
    proc.render_export.side_effect = RuntimeError("kaboom")
    errors: list[str] = []
    finished: list[bool] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run_batch([_task(tmp_path, "a.cr2"), _task(tmp_path, "b.cr2")])

    assert errors == ["kaboom"]
    assert finished == []


def test_no_overwrite_numbers_a_conflicting_target(tmp_path) -> None:
    worker, _ = _worker()
    task = _task(tmp_path, "a.cr2")  # overwrite defaults to False

    worker.run_batch([task])
    worker.run_batch([task])

    files = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert len(files) == 2
    assert any("_2" in n for n in files)


def test_overwrite_replaces_the_existing_target(tmp_path) -> None:
    preset = _preset(tmp_path, overwrite=True)
    worker, _ = _worker()
    task = _task(tmp_path, "a.cr2", preset=preset)

    worker.run_batch([task])
    worker.run_batch([task])

    assert len(list(tmp_path.glob("*.jpg"))) == 1


def test_metadata_embeds_on_the_finisher_thread(tmp_path, monkeypatch) -> None:
    from negpy.features.metadata.models import MetadataConfig

    embed = MagicMock(return_value=b"WITHMETA")
    monkeypatch.setattr("negpy.desktop.workers.export.embed_metadata", embed)
    worker, proc = _worker()
    task = _task(tmp_path, "a.cr2", metadata_config=MetadataConfig(film="Portra 400"))

    worker.run_batch([task])

    embed.assert_called_once()
    assert embed.call_args.args[0] == b"JPGDATA"
    assert list(tmp_path.glob("*.jpg"))[0].read_bytes() == b"WITHMETA"


def test_protect_original_metadata_uses_preserve(tmp_path, monkeypatch) -> None:
    from negpy.features.metadata.models import MetadataConfig

    preserve = MagicMock(return_value=b"PRESERVED")
    embed = MagicMock()
    monkeypatch.setattr("negpy.desktop.workers.export.preserve_source_metadata", preserve)
    monkeypatch.setattr("negpy.desktop.workers.export.embed_metadata", embed)
    worker, proc = _worker()
    task = _task(tmp_path, "a.cr2", metadata_config=MetadataConfig(protect_original_metadata=True))

    worker.run_batch([task])

    preserve.assert_called_once()
    embed.assert_not_called()
    assert list(tmp_path.glob("*.jpg"))[0].read_bytes() == b"PRESERVED"


def test_tiff_metadata_goes_through_the_embed_plan(tmp_path, monkeypatch) -> None:
    from negpy.features.metadata.models import MetadataConfig

    plan = (b"exif", b"xmp", False)
    monkeypatch.setattr("negpy.desktop.workers.export.export_embed_plan", MagicMock(return_value=plan))
    embed = MagicMock()
    monkeypatch.setattr("negpy.desktop.workers.export.embed_metadata", embed)
    worker, proc = _worker(encode=(b"TIFFDATA", "tiff"))
    preset = _preset(tmp_path, export_fmt=ExportFormat.TIFF)
    task = _task(tmp_path, "a.cr2", preset=preset, metadata_config=MetadataConfig())

    worker.run_batch([task])

    assert proc.encode_export.call_args.kwargs["embed_plan"] == plan
    embed.assert_not_called()
    assert list(tmp_path.glob("*.tiff"))[0].read_bytes() == b"TIFFDATA"


def test_completed_batch_never_emits_cancelled(tmp_path) -> None:
    worker, _ = _worker()
    cancelled: list[bool] = []
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.run_batch([_task(tmp_path, "a.cr2")])

    assert cancelled == []


def test_slot_key_discriminates_every_field(tmp_path) -> None:
    proc, path = _real_processor_and_source(tmp_path)
    cfg = WorkspaceConfig()

    calls: list[int] = []
    orig = proc._prepare_export_source_locked
    proc._prepare_export_source_locked = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]

    proc.prefetch_export_source(path, cfg, "base")
    assert len(calls) == 1

    other_cfg = replace(cfg, retouch=replace(cfg.retouch, dust_threshold=cfg.retouch.dust_threshold + 0.1))
    misses = [
        (path, cfg, "other-hash", {}),
        (path, other_cfg, "base", {}),
        (path, cfg, "base", {"half": 1}),
        (path, cfg, "base", {"half": 1, "split_x": 0.4}),
        (path, cfg, "base", {"gutter_thickness": 0.1}),
    ]
    for n, (p, c, h, kw) in enumerate(misses, start=2):
        proc._prepare_export_source(p, c, h, **kw)
        assert len(calls) == n, f"variation {kw or h} must miss the slot"

    proc._prepare_export_source(path, cfg, "base")
    assert len(calls) == len(misses) + 1  # exact key still served from the slot


def test_prepare_miss_keeps_the_slot(tmp_path) -> None:
    proc, path = _real_processor_and_source(tmp_path)
    cfg = WorkspaceConfig()

    proc.prefetch_export_source(path, cfg, "keep")
    slot = proc._prepare_slot
    proc._prepare_export_source(path, cfg, "miss")
    assert proc._prepare_slot is slot


def test_prefetch_same_key_twice_computes_once(tmp_path) -> None:
    proc, path = _real_processor_and_source(tmp_path)
    cfg = WorkspaceConfig()

    calls: list[int] = []
    orig = proc._prepare_export_source_locked
    proc._prepare_export_source_locked = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]

    proc.prefetch_export_source(path, cfg, "twice")
    proc.prefetch_export_source(path, cfg, "twice")
    assert calls == [1]


def test_render_export_uses_the_prefetched_source(tmp_path) -> None:
    proc, path = _real_processor_and_source(tmp_path)
    proc.engine_gpu = None
    cfg = WorkspaceConfig()

    proc.prefetch_export_source(path, cfg, "hx")
    calls: list[int] = []
    orig = proc._prepare_export_source_locked
    proc._prepare_export_source_locked = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]

    buf, cs = proc.render_export(path, cfg, cfg.export, "hx")

    assert calls == []
    assert isinstance(buf, np.ndarray)
    assert cs


def test_process_export_equals_render_plus_encode(tmp_path) -> None:
    proc, path = _real_processor_and_source(tmp_path)
    proc.engine_gpu = None
    cfg = WorkspaceConfig()

    bits1, fmt1 = proc.process_export(path, cfg, cfg.export, "pe")
    buf, cs = proc.render_export(path, cfg, cfg.export, "pe")
    bits2, fmt2 = proc.encode_export(buf, cfg.export, cs)

    assert bits1 == bits2
    assert fmt1 == fmt2 == "jpg"


def test_render_export_error_returns_none_and_message(tmp_path) -> None:
    from negpy.services.rendering.image_processor import ImageProcessor

    proc = ImageProcessor()
    bits, msg = proc.render_export(str(tmp_path / "missing.tif"), WorkspaceConfig(), WorkspaceConfig().export, "h")
    assert bits is None
    assert msg


def test_encode_export_error_returns_none_and_message() -> None:
    from negpy.services.rendering.image_processor import ImageProcessor

    proc = ImageProcessor.__new__(ImageProcessor)  # encode path touches no state
    cfg = WorkspaceConfig()
    # JXL cannot encode the working space; the raised ValueError must come back as (None, msg).
    export = replace(cfg.export, export_fmt=ExportFormat.JXL, export_color_space="No Such Space")
    bits, msg = proc.encode_export(np.zeros((4, 4, 3), np.float32), export, "No Such Space")
    assert bits is None
    assert "JPEG XL" in msg


def test_concurrent_prefetch_and_prepare_agree(tmp_path) -> None:
    proc, path = _real_processor_and_source(tmp_path)
    cfg = WorkspaceConfig()

    ref, _cs, _tok = proc._prepare_export_source(path, cfg, "cA")

    t = threading.Thread(target=proc.prefetch_export_source, args=(path, cfg, "cB"))
    t.start()
    got, _cs2, _tok2 = proc._prepare_export_source(path, cfg, "cC")
    t.join()

    assert np.array_equal(ref, got)
    assert proc._prepare_slot is not None
    assert np.array_equal(ref, proc._prepare_slot[1][0])
