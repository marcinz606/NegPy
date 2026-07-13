"""CPU renders must publish a float-domain histogram_raw before soft-proof quantization."""

import numpy as np
from PIL import Image

from negpy.desktop.workers.render import RenderTask, RenderWorker
from negpy.features.exposure.analysis import output_histogram
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


class _StubProcessor:
    def __init__(self, result: np.ndarray):
        self._result = result

    def run_pipeline(self, *args, **kwargs):
        return self._result, {}

    def buffer_to_pil(self, result, config):
        return Image.fromarray((np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8))

    def soft_proof_preview(self, pil_img, *args, **kwargs):
        return pil_img


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


def test_histogram_raw_binned_from_float_output_before_soft_proof():
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
            icc_output_path="/fake/display.icc",
        ),
    )
    assert np.array_equal(metrics["histogram_raw"], output_histogram(float_result))
    # base_positive is the quantized proof buffer; the histogram must not come from it.
    assert not np.array_equal(metrics["histogram_raw"], output_histogram(metrics["base_positive"]))


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
