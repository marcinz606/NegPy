from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import negpy.desktop.workers.render as render_workers
from negpy.desktop.workers.render import (
    BatchAutoCropInput,
    BatchAutoCropResult,
    BatchAutoCropTask,
    BatchAutoCropWorker,
)
from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.batch_autocrop import ResolvedCrop
from negpy.features.rgbscan.models import RgbScanConfig


class _PreviewService:
    def __init__(self) -> None:
        self.linear_calls: list[dict] = []
        self.rgb_calls: list[dict] = []

    def load_linear_preview(
        self,
        file_path,
        color_space,
        use_camera_wb,
        full_resolution,
        file_hash,
    ):
        self.linear_calls.append(
            {
                "file_path": file_path,
                "color_space": color_space,
                "use_camera_wb": use_camera_wb,
                "full_resolution": full_resolution,
                "file_hash": file_hash,
            }
        )
        raw = np.full((8, 12, 3), 0.5, dtype=np.float32)
        return raw, (12, 8), {}

    def load_linear_preview_rgb(
        self,
        red_path,
        green_path,
        blue_path,
        color_space,
        use_camera_wb,
        full_resolution,
        file_hash,
        align,
    ):
        self.rgb_calls.append(
            {
                "red_path": red_path,
                "green_path": green_path,
                "blue_path": blue_path,
                "color_space": color_space,
                "use_camera_wb": use_camera_wb,
                "full_resolution": full_resolution,
                "file_hash": file_hash,
                "align": align,
            }
        )
        raw = np.full((8, 12, 3), 0.25, dtype=np.float32)
        return raw, (12, 8), {}


def _input(name: str, config: WorkspaceConfig, fingerprint: tuple = ()) -> BatchAutoCropInput:
    return BatchAutoCropInput(
        file_info={"name": name, "path": f"/{name}.dng", "hash": f"hash-{name}"},
        config=config,
        fingerprint=fingerprint,
    )


def _task(*frames: BatchAutoCropInput, generation: int = 0) -> BatchAutoCropTask:
    return BatchAutoCropTask(frames=list(frames), workspace_color_space="Display P3", generation=generation)


def _stub_detection(monkeypatch) -> None:
    monkeypatch.setattr(
        render_workers,
        "detect_crop_candidate",
        lambda key, _image, *, target_ratio: SimpleNamespace(key=key, target_ratio=target_ratio),
    )
    monkeypatch.setattr(render_workers, "resolve_roll_crops", lambda _evidence: [])


def test_batch_autocrop_decodes_with_render_white_balance(qapp, monkeypatch) -> None:
    _stub_detection(monkeypatch)
    base = WorkspaceConfig()
    camera_wb = replace(base, process=replace(base.process, linear_raw=False))
    flat_wb = replace(base, process=replace(base.process, linear_raw=True))
    preview = _PreviewService()
    worker = BatchAutoCropWorker(preview)

    worker.process(_task(_input("camera", camera_wb), _input("flat", flat_wb)))

    assert [call["use_camera_wb"] for call in preview.linear_calls] == [True, False]
    assert all(call["color_space"] == "Display P3" for call in preview.linear_calls)
    assert all(call["full_resolution"] is False for call in preview.linear_calls)


def test_batch_autocrop_uses_asset_rgb_triplet_paths(qapp, monkeypatch) -> None:
    _stub_detection(monkeypatch)
    base = WorkspaceConfig()
    rgb = RgbScanConfig(enabled=True, green_path="/green.dng", blue_path="/blue.dng", align=False)
    config = replace(base, rgbscan=rgb, process=replace(base.process, linear_raw=False))
    preview = _PreviewService()
    worker = BatchAutoCropWorker(preview)

    worker.process(_task(_input("red", config)))

    assert preview.linear_calls == []
    assert preview.rgb_calls == [
        {
            "red_path": "/red.dng",
            "green_path": "/green.dng",
            "blue_path": "/blue.dng",
            "color_space": "Display P3",
            "use_camera_wb": True,
            "full_resolution": False,
            "file_hash": "hash-red",
            "align": False,
        }
    ]


@pytest.mark.parametrize(("flatfield_enabled", "expected_k1"), [(True, 0.23), (False, 0.0)])
def test_batch_autocrop_applies_flatfield_and_crop_free_geometry(
    qapp,
    monkeypatch,
    flatfield_enabled: bool,
    expected_k1: float,
) -> None:
    base = WorkspaceConfig()
    geometry = replace(
        base.geometry,
        rotation=1,
        fine_rotation=2.5,
        flip_horizontal=True,
        manual_crop_rect=(0.1, 0.2, 0.8, 0.9),
        auto_crop_enabled=True,
        autocrop_offset=17,
        autocrop_ratio="4:3",
    )
    flatfield = replace(
        base.flatfield,
        apply=flatfield_enabled,
        reference_path="/flat.dng",
        k1=0.23,
    )
    config = replace(base, geometry=geometry, flatfield=flatfield)
    preview = _PreviewService()
    worker = BatchAutoCropWorker(preview)
    captured: dict = {}

    def _apply_flatfield(image, received_config):
        captured["flatfield_config"] = received_config
        return image + 1.0

    class _GeometryProcessor:
        def __init__(self, received_geometry, distortion_k1):
            captured["geometry"] = received_geometry
            captured["distortion_k1"] = distortion_k1

        def process(self, image, context):
            captured["geometry_input"] = image.copy()
            captured["context"] = context
            return image + 2.0

    def _detect(key, transformed, *, target_ratio):
        captured["detected_key"] = key
        captured["detected_image"] = transformed.copy()
        captured["target_ratio"] = target_ratio
        return SimpleNamespace(key=key)

    monkeypatch.setattr(render_workers, "apply_flatfield", _apply_flatfield)
    monkeypatch.setattr(render_workers, "GeometryProcessor", _GeometryProcessor)
    monkeypatch.setattr(render_workers, "detect_crop_candidate", _detect)
    monkeypatch.setattr(render_workers, "resolve_roll_crops", lambda _evidence: [])

    worker.process(_task(_input("frame", config)))

    assert captured["flatfield_config"] == flatfield
    assert captured["geometry"].manual_crop_rect is None
    assert captured["geometry"].auto_crop_enabled is False
    assert captured["geometry"].autocrop_offset == 0
    assert captured["geometry"].rotation == 1
    assert captured["geometry"].fine_rotation == 2.5
    assert captured["geometry"].flip_horizontal is True
    assert captured["distortion_k1"] == expected_k1
    assert np.allclose(captured["geometry_input"], 1.5)
    assert np.allclose(captured["detected_image"], 3.5)
    assert captured["target_ratio"] == "4:3"
    assert captured["context"].original_size == (12, 8)
    assert captured["context"].scale_factor == 1.0


def test_batch_autocrop_per_file_failure_does_not_abort_roll(qapp, monkeypatch) -> None:
    base = WorkspaceConfig()

    class _FailFirstPreview(_PreviewService):
        def load_linear_preview(self, file_path, color_space, use_camera_wb, full_resolution, file_hash):
            if file_hash == "hash-bad":
                raise RuntimeError("broken preview")
            return super().load_linear_preview(file_path, color_space, use_camera_wb, full_resolution, file_hash)

    preview = _FailFirstPreview()
    worker = BatchAutoCropWorker(preview)
    detected: list[str] = []

    def _detect(key, _transformed, *, target_ratio):
        detected.append(key)
        return SimpleNamespace(key=key)

    def _resolve(evidence):
        assert len(evidence) == 1
        return [ResolvedCrop(evidence[0].key, (0.1, 0.2, 0.8, 0.9), 1.25, 0.7, True)]

    monkeypatch.setattr(render_workers, "detect_crop_candidate", _detect)
    monkeypatch.setattr(render_workers, "resolve_roll_crops", _resolve)
    progress: list[tuple[int, int, str]] = []
    finished: list[list[BatchAutoCropResult]] = []
    errors: list[str] = []
    worker.progress.connect(lambda current, total, name: progress.append((current, total, name)))
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.process(_task(_input("bad", base), _input("good", base, ("good-fingerprint",))))

    assert len(detected) == 1
    assert progress == [(1, 2, "bad"), (2, 2, "good")]
    assert errors == []
    assert len(finished) == 1
    assert [result.file_info["name"] for result in finished[0]] == ["good"]


def test_batch_autocrop_cancellation_after_decode_emits_no_results(qapp, monkeypatch) -> None:
    base = WorkspaceConfig()
    preview = _PreviewService()
    worker = BatchAutoCropWorker(preview)
    original_load = preview.load_linear_preview

    def _load_and_cancel(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        worker.cancel()
        return loaded

    preview.load_linear_preview = _load_and_cancel
    detect_calls: list[bool] = []
    resolve_calls: list[bool] = []
    monkeypatch.setattr(render_workers, "detect_crop_candidate", lambda *_a, **_k: detect_calls.append(True))
    monkeypatch.setattr(render_workers, "resolve_roll_crops", lambda _e: resolve_calls.append(True))
    finished: list[list] = []
    cancelled: list[bool] = []
    worker.finished.connect(finished.append)
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.process(_task(_input("frame", base)))

    assert cancelled == [True]
    assert finished == []
    assert detect_calls == []
    assert resolve_calls == []


def test_batch_autocrop_honors_cancel_before_queued_process_starts(qapp, monkeypatch) -> None:
    _stub_detection(monkeypatch)
    preview = _PreviewService()
    worker = BatchAutoCropWorker(preview)
    finished: list[list] = []
    cancelled: list[bool] = []
    worker.finished.connect(finished.append)
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.cancel(42)
    worker.process(_task(_input("frame", WorkspaceConfig()), generation=42))

    assert cancelled == [True]
    assert finished == []
    assert preview.linear_calls == []


def test_batch_autocrop_honors_cancel_during_final_resolution(qapp, monkeypatch) -> None:
    base = WorkspaceConfig()
    worker = BatchAutoCropWorker(_PreviewService())
    monkeypatch.setattr(
        render_workers,
        "detect_crop_candidate",
        lambda key, _image, *, target_ratio: SimpleNamespace(key=key),
    )

    def _resolve(evidence):
        worker.cancel(77)
        return [ResolvedCrop(evidence[0].key, (0.1, 0.1, 0.9, 0.9), 0.0, 0.8, False)]

    monkeypatch.setattr(render_workers, "resolve_roll_crops", _resolve)
    finished: list[list] = []
    cancelled: list[bool] = []
    worker.finished.connect(finished.append)
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.process(_task(_input("frame", base), generation=77))

    assert cancelled == [True]
    assert finished == []


def test_batch_autocrop_propagates_resolved_payload_to_source_frames(qapp, monkeypatch) -> None:
    base = WorkspaceConfig()
    first = _input("first", base, ("first", 1))
    second = _input("second", base, ("second", 2))
    worker = BatchAutoCropWorker(_PreviewService())

    monkeypatch.setattr(
        render_workers,
        "detect_crop_candidate",
        lambda key, _image, *, target_ratio: SimpleNamespace(key=key),
    )

    def _resolve(evidence):
        return [
            ResolvedCrop(evidence[1].key, (0.2, 0.3, 0.7, 0.8), -0.75, 0.82, True),
            ResolvedCrop(evidence[0].key, (0.1, 0.15, 0.9, 0.95), 1.5, 0.91, False),
        ]

    monkeypatch.setattr(render_workers, "resolve_roll_crops", _resolve)
    finished: list[list[BatchAutoCropResult]] = []
    worker.finished.connect(finished.append)

    worker.process(_task(first, second))

    assert len(finished) == 1
    assert finished[0] == [
        BatchAutoCropResult(second.file_info, second.fingerprint, (0.2, 0.3, 0.7, 0.8), -0.75, 0.82, True),
        BatchAutoCropResult(first.file_info, first.fingerprint, (0.1, 0.15, 0.9, 0.95), 1.5, 0.91, False),
    ]


def test_batch_autocrop_resolve_failure_emits_error_not_finished(qapp, monkeypatch) -> None:
    _stub_detection(monkeypatch)
    worker = BatchAutoCropWorker(_PreviewService())
    monkeypatch.setattr(render_workers, "resolve_roll_crops", lambda _evidence: (_ for _ in ()).throw(RuntimeError("resolve failed")))
    finished: list[list] = []
    errors: list[str] = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.process(_task(_input("frame", WorkspaceConfig())))

    assert finished == []
    assert errors == ["resolve failed"]
