import leak_sim as sim
import numpy as np

from negpy.desktop.workers.leak_roll import LeakAutoTask, LeakAutoWorker, LeakRollTask, LeakRollWorker
from negpy.domain.models import WorkspaceConfig
from negpy.features.leak.roll import picture_box
from negpy.features.leak.shapes import new_band


def _frames():
    leaked = sim.leaked(sim.gray_scene(), sim.edge_leak(3.0), sim.BACK)
    clean = sim.leaked(sim.gray_scene(seed=2), np.zeros((sim.H, sim.W), np.float32), sim.BACK)
    config = WorkspaceConfig(geometry=leaked.geometry)
    return {"/roll/a.tif": leaked.neg, "/roll/b.tif": clean.neg}, config


def _roll_worker(monkeypatch, decoded):
    worker = LeakRollWorker()
    monkeypatch.setattr(worker._processor, "_load_source_f32", lambda path, params, **k: (decoded[path], None, "Adobe RGB"))
    released = []
    monkeypatch.setattr(worker._processor, "release_source_cache", lambda: released.append(True))
    out = []
    worker.finished.connect(out.append)
    worker.error.connect(lambda message: out.append({"error": message}))
    return worker, out, released


def test_auto_runs_on_the_frame_it_was_sent():
    decoded, config = _frames()
    worker = LeakAutoWorker()
    out = []
    worker.finished.connect(out.append)
    worker.run(LeakAutoTask("ha", decoded["/roll/a.tif"], config))
    assert out[0]["hash"] == "ha" and out[0]["shape"] == (sim.H, sim.W)
    assert [s.kind.value for s in out[0]["found"]] == ["edge"]


def test_from_roll_selects_the_frames_auto_finds_a_leak_on(monkeypatch):
    decoded, config = _frames()
    worker, out, released = _roll_worker(monkeypatch, decoded)
    files = tuple({"path": p, "hash": p[-5]} for p in decoded)
    worker.run(LeakRollTask(files, {p: config for p in decoded}))
    assert out == [{"found": ["a"]}]
    assert released == [True]


def test_from_roll_carries_the_drawn_leak_by_each_frames_picture(monkeypatch):
    decoded, config = _frames()
    worker, out, released = _roll_worker(monkeypatch, decoded)
    shapes = (new_band((0.5, 0.2), (0.5, 0.8), 0.05, 0.05),)
    source = picture_box(decoded["/roll/a.tif"], config.geometry)
    files = tuple({"path": p, "hash": p[-5]} for p in decoded)
    worker.run(LeakRollTask(files, {p: config for p in decoded}, shapes=shapes, source=source))
    assert set(out[0]["shapes"]) == {"a", "b"}
    np.testing.assert_allclose(np.array(out[0]["shapes"]["b"][0].points), np.array(shapes[0].points), atol=1e-6)
    assert released == [True]


def test_auto_reads_the_preview_flat_fielded_and_unmixed_as_the_bake_does(monkeypatch):
    from negpy.services.rendering import image_processor

    decoded, config = _frames()
    seen = []

    def precorrect(img, settings, gain_slices=(), orientation=1):
        seen.append((gain_slices, orientation))
        return img

    monkeypatch.setattr(image_processor, "precorrect_source", precorrect)
    worker = LeakAutoWorker()
    out = []
    worker.finished.connect(out.append)
    half = ((1, 0.5, None, 0.0, "x"),)
    worker.run(LeakAutoTask("ha", decoded["/roll/a.tif"], config, gain_slices=half, orientation=6))
    assert seen == [(half, 6)]
    assert out[0]["found"]
