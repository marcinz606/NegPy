import threading

from negpy.kernel.system import timing


def _fake_clock(monkeypatch, ticks):
    it = iter(ticks)
    monkeypatch.setattr(timing, "perf_counter", lambda: next(it))


def test_laps_and_steps_add_up_to_the_total(monkeypatch):
    # collect 0 · lap a 0.010 · step enter 0.010 · inner lap 0.040 · step exit 0.050 · lap b 0.052 · collect exit 0.060
    _fake_clock(monkeypatch, [0.0, 0.010, 0.010, 0.040, 0.050, 0.052, 0.060])
    with timing.collect() as times:
        timing.lap("decode")
        with timing.step("GPU"):
            timing.lap("upload")
        timing.lap("encode")
    assert timing.summary(times) == "60 ms = decode 10 + GPU 40 [upload 30, other 10] + encode 2 + other 8"


def test_steps_under_half_a_millisecond_are_left_out(monkeypatch):
    _fake_clock(monkeypatch, [0.0, 0.0002, 0.0102, 0.0102])
    with timing.collect() as times:
        timing.lap("cached stage")
        timing.lap("lab")
    assert timing.summary(times) == "10 ms = lab 10"


def test_nothing_is_recorded_without_a_collector():
    timing.lap("orphan")
    with timing.step("orphan step"):
        timing.lap("inner")


def test_another_thread_does_not_record_into_this_collector():
    with timing.collect() as times:
        worker = threading.Thread(target=lambda: timing.lap("prefetch"))
        worker.start()
        worker.join()
    assert [c.name for c in times.children] == []


def test_repeated_steps_merge_into_one_entry(monkeypatch):
    # collect 0 · step enter 0 · lap 0.002 · exit 0.003 · step enter 0.003 · lap 0.006 · exit 0.007 · collect exit 0.007
    _fake_clock(monkeypatch, [0.0, 0.0, 0.002, 0.003, 0.003, 0.006, 0.007, 0.007])
    with timing.collect() as times:
        for _ in range(2):
            with timing.step("tile"):
                timing.lap("upload")
    assert timing.summary(times) == "7 ms = tile 7 [upload 5, other 2]"


def test_a_cpu_render_reports_its_stages():
    import numpy as np

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    img = np.random.default_rng(1).random((90, 60, 3), dtype=np.float32)
    with timing.collect() as times:
        ImageProcessor(use_gpu=False).run_pipeline(img, WorkspaceConfig(), "t", render_size_ref=90, prefer_gpu=False)
    cpu = next(c for c in times.children if c.name == "CPU")
    names = [c.name for c in cpu.children]
    for stage in ("geometry/normalize", "exposure", "lab", "finish/output encode"):
        assert stage in names


def test_an_export_logs_its_render_steps(tmp_path, caplog):
    import numpy as np
    import tifffile

    from negpy.domain.models import ExportConfig, WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    path = tmp_path / "frame.tif"
    tifffile.imwrite(path, (np.random.default_rng(1).random((60, 90, 3)) * 40000 + 8000).astype(np.uint16), photometric="rgb")
    with caplog.at_level("INFO", logger="negpy"):
        ImageProcessor(use_gpu=False).render_export(str(path), WorkspaceConfig(), ExportConfig(), "h", prefer_gpu=False)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("export-timing render"))
    assert "decode/flat-field/unmix" in line and "CPU" in line and line.endswith("| frame.tif")
