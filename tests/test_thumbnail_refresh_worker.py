from dataclasses import replace

import numpy as np

import negpy.desktop.workers.render as render_workers
from negpy.desktop.workers.render import (
    ThumbnailRenderInput,
    ThumbnailRenderTask,
    ThumbnailRenderWorker,
)
from negpy.domain.models import WorkspaceConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.features.stitch.models import StitchConfig


def _meta(tag: str) -> dict:
    return {
        "cam_xyz": [[tag, 0, 0], [0, 1, 0], [0, 0, 1]],
        "camera_wb": [1.1, 1.0, 1.3, 1.0],
        "ir_preview": np.full((2, 2), 0.9, dtype=np.float32),
        "detect_preview": np.full((2, 2), 0.8, dtype=np.float32),
    }


class _PreviewService:
    def __init__(self) -> None:
        self.linear_calls: list[dict] = []
        self.rgb_calls: list[dict] = []
        self.stitch_calls: list[dict] = []

    def load_linear_preview(
        self,
        file_path,
        color_space,
        use_camera_wb,
        full_resolution,
        file_hash,
        demosaic="Auto",
        positive_source=False,  # noqa: ARG002 — accepted, not asserted on
        highlight_mode=0,  # noqa: ARG002 — accepted, not asserted on
        bake_camera_wb=False,  # noqa: ARG002 — accepted, not asserted on
    ):
        self.linear_calls.append({"file_path": file_path, "file_hash": file_hash})
        raw = np.full((4, 6, 3), 0.5, dtype=np.float32)
        return raw, (6, 4), _meta("plain")

    def load_linear_preview_rgb(
        self,
        red_path,
        rgbscan,
        color_space,
        use_camera_wb,
        full_resolution,
        file_hash,
        demosaic="Auto",  # noqa: ARG002 — accepted, not asserted on
    ):
        self.rgb_calls.append({"red_path": red_path, "file_hash": file_hash})
        raw = np.full((4, 6, 3), 0.25, dtype=np.float32)
        return raw, (6, 4), _meta("rgb")

    def load_linear_preview_stitch(
        self,
        primary_path,
        stitch,
        color_space,
        use_camera_wb,
        full_resolution,
        file_hash,
        flatfield_profile_id="",
        demosaic="Auto",  # noqa: ARG002 — accepted, not asserted on
    ):
        self.stitch_calls.append({"primary_path": primary_path, "file_hash": file_hash, "flatfield_profile_id": flatfield_profile_id})
        raw = np.full((4, 6, 3), 0.35, dtype=np.float32)
        return raw, (6, 4), _meta("stitch")


class _StubProcessor:
    """Stands in for ImageProcessor: records run_pipeline calls, returns a fixed
    positive without running the real pipeline."""

    def __init__(self, use_gpu: bool = True) -> None:
        self.use_gpu = use_gpu
        self.run_calls: list[dict] = []
        self.cleanup_calls: list[dict] = []
        self.fail_hashes: set[str] = set()
        self.result_value = np.full((4, 6, 4), 0.75, dtype=np.float32)

    def run_pipeline(self, buffer, config, source_hash, **kwargs):
        if source_hash in self.fail_hashes:
            raise RuntimeError(f"broken pipeline for {source_hash}")
        self.run_calls.append({"buffer": buffer, "config": config, "source_hash": source_hash, **kwargs})
        return self.result_value, {}

    def cleanup(self, **kwargs):
        self.cleanup_calls.append(kwargs)


def _worker(monkeypatch, preview=None) -> tuple[ThumbnailRenderWorker, _StubProcessor]:
    processor = _StubProcessor()
    monkeypatch.setattr(render_workers, "ImageProcessor", lambda use_gpu=True: processor)
    worker = ThumbnailRenderWorker(preview or _PreviewService())
    return worker, processor


def _input(name: str, config: WorkspaceConfig, icc_input_active: bool = False) -> ThumbnailRenderInput:
    return ThumbnailRenderInput(
        file_info={"name": name, "path": f"/{name}.dng", "hash": f"hash-{name}"},
        config=config,
        thumbnail_key=f"key-{name}",
        icc_input_active=icc_input_active,
    )


def _task(*frames: ThumbnailRenderInput, generation: int = 0) -> ThumbnailRenderTask:
    return ThumbnailRenderTask(frames=list(frames), workspace_color_space="Display P3", generation=generation)


def test_thumbnail_render_constructs_a_cpu_only_processor(qapp, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(render_workers, "ImageProcessor", lambda use_gpu=True: captured.setdefault("use_gpu", use_gpu) or _StubProcessor())
    ThumbnailRenderWorker(_PreviewService())
    assert captured["use_gpu"] is False


def test_thumbnail_render_decodes_and_runs_pipeline_per_frame(qapp, monkeypatch) -> None:
    worker, processor = _worker(monkeypatch)
    base = WorkspaceConfig()
    rendered: list[tuple[str, np.ndarray]] = []
    worker.rendered.connect(lambda frame, buf: rendered.append((frame.thumbnail_key, buf)))

    worker.process(_task(_input("a", base), _input("b", base)))

    assert [call["source_hash"] for call in processor.run_calls] == ["hash-a", "hash-b"]
    assert all(call["prefer_gpu"] is False for call in processor.run_calls)
    assert all(call["cache_stages"] is False for call in processor.run_calls)
    assert all(call["readback_metrics"] is False for call in processor.run_calls)
    assert [key for key, _buf in rendered] == ["key-a", "key-b"]
    assert all(buf.shape[2] == 3 for _key, buf in rendered)
    assert processor.cleanup_calls == [{"release_source_cache": True, "collect": False}]


def test_thumbnail_render_passes_camera_transform_and_repair_planes_through(qapp, monkeypatch) -> None:
    """A background render must reach the same camera-matrix/IR inputs the live render
    would, or a transparency (E-6) thumbnail renders with the wrong colour."""
    worker, processor = _worker(monkeypatch)
    base = WorkspaceConfig()

    worker.process(_task(_input("frame", base)))

    call = processor.run_calls[0]
    assert call["cam_xyz"] == [["plain", 0, 0], [0, 1, 0], [0, 0, 1]]
    assert call["camera_wb"] == [1.1, 1.0, 1.3, 1.0]
    assert np.array_equal(call["ir_buffer"], np.full((2, 2), 0.9, dtype=np.float32))
    assert np.array_equal(call["detect_buffer"], np.full((2, 2), 0.8, dtype=np.float32))


def test_thumbnail_render_substitutes_wb_only_cam_xyz_when_input_icc_active(qapp, monkeypatch) -> None:
    worker, processor = _worker(monkeypatch)
    base = WorkspaceConfig()

    worker.process(_task(_input("frame", base, icc_input_active=True)))

    call = processor.run_calls[0]
    # wb_only_cam_xyz replaces the decoder's own matrix with the fixed working-space
    # identity stand-in; it must not still be the raw ["plain", 0, 0] row from decode.
    assert call["cam_xyz"] != [["plain", 0, 0], [0, 1, 0], [0, 0, 1]]
    assert call["cam_xyz"] is not None


def test_thumbnail_render_routes_composite_decode_by_asset_type(qapp, monkeypatch) -> None:
    preview = _PreviewService()
    worker, _processor = _worker(monkeypatch, preview)
    rgb = RgbScanConfig(enabled=True, green_path="/g.dng", blue_path="/b.dng", align=False)
    config = replace(WorkspaceConfig(), rgbscan=rgb, process=replace(WorkspaceConfig().process, linear_raw=False))

    worker.process(_task(_input("triplet", config)))

    assert preview.linear_calls == []
    assert preview.stitch_calls == []
    assert [call["file_hash"] for call in preview.rgb_calls] == ["hash-triplet"]


def test_thumbnail_render_routes_stitch_composites_through_the_stitch_decode(qapp, monkeypatch) -> None:
    preview = _PreviewService()
    worker, processor = _worker(monkeypatch, preview)
    stitch = StitchConfig(stitch_enabled=True, stitch_paths=("/a.dng", "/b.dng"))
    config = replace(WorkspaceConfig(), stitch=stitch)

    worker.process(_task(_input("panorama", config)))

    assert preview.linear_calls == []
    assert preview.rgb_calls == []
    assert [call["file_hash"] for call in preview.stitch_calls] == ["hash-panorama"]
    # A stitch composite must not be persisted as its first tile alone.
    assert processor.run_calls[0]["buffer"].shape == (4, 6, 3)


def test_thumbnail_render_skips_failed_frame_and_continues(qapp, monkeypatch) -> None:
    worker, processor = _worker(monkeypatch)
    processor.fail_hashes = {"hash-bad"}
    base = WorkspaceConfig()
    rendered: list[str] = []
    progress: list[tuple[int, int, str]] = []
    finished: list[int] = []
    worker.rendered.connect(lambda frame, _buf: rendered.append(frame.thumbnail_key))
    worker.progress.connect(lambda current, total, name: progress.append((current, total, name)))
    worker.finished.connect(finished.append)

    worker.process(_task(_input("bad", base), _input("good", base)))

    assert rendered == ["key-good"]
    assert [count for count, _total, _name in progress] == [1, 2]
    assert finished == [1]


def test_thumbnail_render_cancel_stops_before_further_frames(qapp, monkeypatch) -> None:
    worker, processor = _worker(monkeypatch)
    base = WorkspaceConfig()
    rendered: list[str] = []
    cancelled: list[bool] = []
    finished: list[int] = []

    # Cancelling as soon as the first frame's result lands must stop the second
    # frame's pipeline call from ever happening.
    worker.rendered.connect(lambda frame, _buf: (rendered.append(frame.thumbnail_key), worker.cancel()))
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.finished.connect(finished.append)

    worker.process(_task(_input("first", base), _input("second", base)))

    assert rendered == ["key-first"]
    assert len(processor.run_calls) == 1
    assert cancelled == [True]
    assert finished == []


def test_thumbnail_render_cancel_racing_the_last_frame_still_emits_cancelled_not_finished(qapp, monkeypatch) -> None:
    """A cancel() arriving after the loop's own cancel check but before the terminal
    signal must still win — the atomic _emit_finished_unless_cancelled is what this
    guards, since a naive check-then-emit has a window for exactly this race."""
    worker, processor = _worker(monkeypatch)
    base = WorkspaceConfig()
    finished: list[int] = []
    cancelled: list[bool] = []
    worker.finished.connect(finished.append)
    worker.cancelled.connect(lambda: cancelled.append(True))

    orig_cleanup = processor.cleanup

    def _cleanup_and_cancel(**kwargs):
        worker.cancel()  # races the terminal decision, after the loop already finished
        return orig_cleanup(**kwargs)

    processor.cleanup = _cleanup_and_cancel

    worker.process(_task(_input("frame", base)))

    assert cancelled == [True]
    assert finished == []


def test_thumbnail_render_cancel_before_process_starts_emits_no_frames(qapp, monkeypatch) -> None:
    worker, processor = _worker(monkeypatch)
    base = WorkspaceConfig()
    rendered: list[str] = []
    cancelled: list[bool] = []
    worker.rendered.connect(lambda frame, _buf: rendered.append(frame.thumbnail_key))
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.cancel(42)
    worker.process(_task(_input("frame", base), generation=42))

    assert rendered == []
    assert cancelled == [True]
    assert processor.run_calls == []


def test_thumbnail_render_cancel_does_not_poison_next_generation(qapp, monkeypatch) -> None:
    worker, processor = _worker(monkeypatch)
    base = WorkspaceConfig()
    rendered: list[str] = []
    finished: list[int] = []
    worker.rendered.connect(lambda frame, _buf: rendered.append(frame.thumbnail_key))
    worker.finished.connect(finished.append)

    worker.cancel(1)
    worker.process(_task(_input("skipped", base), generation=1))
    worker.process(_task(_input("kept", base), generation=2))

    assert rendered == ["key-kept"]
    assert finished == [1]
