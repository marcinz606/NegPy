"""CPU renders must publish a float-domain histogram_raw off the pipeline output."""

import numpy as np

from negpy.desktop.workers.render import RenderTask, RenderWorker
from negpy.features.exposure.analysis import output_histogram
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


class _StubProcessor:
    def __init__(self, result: np.ndarray):
        self._result = result

    def run_pipeline(self, *args, **kwargs):
        return self._result, {}


def _make_worker(result: np.ndarray) -> RenderWorker:
    worker = RenderWorker.__new__(RenderWorker)
    super(RenderWorker, worker).__init__()
    worker._processor = _StubProcessor(result)
    return worker


def _run(worker: RenderWorker, task: RenderTask) -> dict:
    got: dict = {}
    worker.finished.connect(lambda _res, metrics: got.update(metrics))
    worker.process(task)
    return got


def test_histogram_raw_binned_from_the_float_pipeline_output():
    rng = np.random.default_rng(0)
    float_result = rng.uniform(0.2, 0.8, (32, 32, 3)).astype(np.float32)
    worker = _make_worker(float_result)
    metrics = _run(
        worker,
        RenderTask(
            buffer=float_result,
            config=DEFAULT_WORKSPACE_CONFIG,
            source_hash="h",
            preview_size=32.0,
        ),
    )
    assert np.array_equal(metrics["histogram_raw"], output_histogram(float_result))
    # The worker hands the render on untouched — no proof is baked into it, so the
    # histogram and the published buffer describe the same pixels.
    assert metrics["base_positive"] is float_result


def test_histogram_raw_not_recomputed_when_pipeline_provides_it():
    float_result = np.full((8, 8, 3), 0.5, dtype=np.float32)
    worker = _make_worker(float_result)
    gpu_bins = np.ones((4, 256), dtype=np.float64)
    worker._processor.run_pipeline = lambda *a, **k: (float_result, {"histogram_raw": gpu_bins})
    metrics = _run(
        worker,
        RenderTask(
            buffer=float_result,
            config=DEFAULT_WORKSPACE_CONFIG,
            source_hash="h",
            preview_size=8.0,
        ),
    )
    assert metrics["histogram_raw"] is gpu_bins
