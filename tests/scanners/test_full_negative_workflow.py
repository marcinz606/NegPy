from __future__ import annotations

import json
import threading
from dataclasses import replace

import numpy as np
import pytest

from negpy.infrastructure.scanners.params import (
    RegisteredScanGeometry,
    ScannerCaptureState,
    ScanParams,
)
from negpy.infrastructure.scanners.result import (
    ScanResult,
    SplitIrAlignment,
    SplitSourceCapture,
)
from negpy.services.scanning.full_negative_workflow import FullNegativeWorkflow
from negpy.services.scanning.forward_roll import validate_full_negative_manifest
from negpy.services.scanning.provenance import PlanIdentity
from negpy.services.scanning.roll_service import FrameRegistration


PITCH = 8
PHASE = 3
APPROVED_PLAN_BINDING = "a" * 64


def _identity() -> PlanIdentity:
    return PlanIdentity(
        roll_id="11111111-1111-4111-8111-111111111111",
        insertion_id="22222222-2222-4222-8222-222222222222",
        session_id="33333333-3333-4333-8333-333333333333",
    )


def _state() -> ScannerCaptureState:
    return ScannerCaptureState(216, 1.0, 1370.0, 1290.0, 1120.0)


def _source(rows: np.ndarray) -> ScanResult:
    height = len(rows)
    rgb = np.broadcast_to(rows[:, None, None], (height, 5, 3)).astype(np.uint16, copy=True)
    ir = np.broadcast_to((rows + 100)[:, None], (height, 5)).astype(np.uint16, copy=True)
    proxy = rgb.copy()
    return ScanResult(
        rgb=rgb,
        ir=ir,
        dpi=4000,
        device_model="Nikon LS-5000 ED",
        ir_alignment=SplitIrAlignment(mode="identity", dx_px=0.0, dy_px=0.0),
        ir_valid_mask=np.ones((height, 5), dtype=np.bool_),
        capture_state=_state(),
        split_source=SplitSourceCapture(rgb4x=rgb, rgb1x_proxy=proxy, ir1x=ir),
    )


def _multiscale_alignment() -> SplitIrAlignment:
    return SplitIrAlignment(
        mode="multiscale-global",
        dx_px=0.10,
        dy_px=0.0,
        phase_responses=(0.45, 0.46, 0.44),
        channel_spread_px=0.02,
        estimator_version=2,
        multiscale_max_dimensions=(1024, 2048),
        multiscale_channel_shifts_px=(
            ((0.18, -0.06), (0.16, -0.04), (0.17, -0.05)),
            ((0.10, -0.05), (0.11, -0.04), (0.09, -0.06)),
        ),
        multiscale_responses=((0.35, 0.36, 0.34), (0.45, 0.46, 0.44)),
    )


def _registration() -> FrameRegistration:
    return FrameRegistration(
        frame=37,
        target_start_row=107,
        usable_tail_row=595.0,
        confidence="medium",
        geometry=RegisteredScanGeometry(
            frame=37,
            subframe_mm=(PHASE + 0.25) * 25.4 / 4000,
            br_y_device_px=4949,
        ),
    )


def _recipe() -> ScanParams:
    return ScanParams(
        dpi=4000,
        depth=16,
        capture_ir=True,
        samples_per_scan=4,
        autofocus=True,
        auto_exposure=True,
    )


class _Scanner:
    def __init__(self, results: list[ScanResult], *, output_dir=None) -> None:
        self.results = list(results)
        self.calls: list[ScanParams] = []
        self.output_dir = output_dir

    def run_scan(self, _device_id, params, _progress, _cancel):
        if len(self.calls) == 1 and self.output_dir is not None:
            assert list((self.output_dir / "sources" / "frame037").glob("split-source-*/manifest.json"))
        self.calls.append(params)
        result = self.results.pop(0)
        if params.capture_state is not None:
            result = replace(result, capture_state=params.capture_state)
        return result

    def probe_device(self, device_id):
        return type(
            "Device",
            (),
            {"id": device_id, "vendor": "Nikon", "model": "LS-5000 ED"},
        )()


def test_workflow_commits_each_raw_source_then_final_triplet_and_manifest(tmp_path) -> None:
    scanner = _Scanner(
        [
            _source(np.arange(PITCH, dtype=np.uint16)),
            _source(np.arange(PITCH, PITCH + PHASE, dtype=np.uint16)),
        ],
        output_dir=tmp_path,
    )

    completed = FullNegativeWorkflow(
        scanner=scanner,
        output_dir=tmp_path,
        identity=_identity(),
        approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
        pitch_rows=PITCH,
    ).capture_frame(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=threading.Event(),
    )

    assert completed.manifest_path == tmp_path / "manifests" / "frame037.json"
    manifest = json.loads(completed.manifest_path.read_text(encoding="utf-8"))
    assert manifest["identity"] == _identity().to_dict()
    assert manifest["source_pair"]["logical_frame"] == 37
    assert manifest["captures"][0]["physical_frame"] == 37
    assert manifest["captures"][1]["physical_frame"] == 38
    assert manifest["captures"][1]["geometry"]["br_y_device_px"] == PHASE - 1
    assert manifest["captures"][1]["capture_state"]["mode"] == "replayed"
    assert all(capture["recipe"]["preserve_full_canvas"] is True for capture in manifest["captures"])
    assert manifest["reconstruction"]["crop_start_row"] == PHASE
    assert len(manifest["binding_sha256"]) == 64
    assert (tmp_path / "final" / "frame037.tif").is_file()
    assert (tmp_path / "final" / "frame037_IR.tif").is_file()
    assert (tmp_path / "final" / "frame037_IR_VALID.tif").is_file()
    assert len(scanner.calls) == 2


def test_forward_roll_validator_reparses_manifest_and_rechecks_every_live_artifact(tmp_path) -> None:
    completed = FullNegativeWorkflow(
        scanner=_Scanner(
            [
                _source(np.arange(PITCH, dtype=np.uint16)),
                _source(np.arange(PITCH, PITCH + PHASE, dtype=np.uint16)),
            ]
        ),
        output_dir=tmp_path,
        identity=_identity(),
        approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
        pitch_rows=PITCH,
    ).capture_frame(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=threading.Event(),
    )

    digest = validate_full_negative_manifest(
        manifest_path=completed.manifest_path,
        identity=_identity(),
        frame=37,
    )
    assert len(digest) == 64

    rgb_path = tmp_path / "final" / "frame037.tif"
    rgb_path.write_bytes(rgb_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="live artifact differs"):
        validate_full_negative_manifest(
            manifest_path=completed.manifest_path,
            identity=_identity(),
            frame=37,
        )


def test_second_transfer_failure_preserves_first_raw_master_without_final_manifest(tmp_path) -> None:
    class FailingSecond(_Scanner):
        def run_scan(self, *args, **kwargs):
            if self.calls:
                raise OSError("USB stall")
            return super().run_scan(*args, **kwargs)

    scanner = FailingSecond([_source(np.arange(PITCH, dtype=np.uint16))])
    workflow = FullNegativeWorkflow(
        scanner=scanner,
        output_dir=tmp_path,
        identity=_identity(),
        approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
        pitch_rows=PITCH,
    )

    with pytest.raises(OSError, match="USB stall"):
        workflow.capture_frame(
            device_id="coolscan3:usb:test",
            registration=_registration(),
            recipe=_recipe(),
            stop=threading.Event(),
        )

    assert list((tmp_path / "sources" / "frame037").glob("split-source-*/manifest.json"))
    assert (tmp_path / "checkpoints" / "frame037-first-source.json").is_file()
    assert not (tmp_path / "manifests" / "frame037.json").exists()
    assert not (tmp_path / "final" / "frame037.tif").exists()

    resume_scanner = _Scanner([_source(np.arange(PITCH, PITCH + PHASE, dtype=np.uint16))])
    completed = FullNegativeWorkflow(
        scanner=resume_scanner,
        output_dir=tmp_path,
        identity=_identity(),
        approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
        pitch_rows=PITCH,
    ).capture_frame(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=threading.Event(),
    )

    assert completed.manifest_path.is_file()
    assert len(resume_scanner.calls) == 1
    assert resume_scanner.calls[0].frame == 38


def test_first_source_checkpoint_round_trips_multiscale_alignment_telemetry(tmp_path) -> None:
    class FailingSecond(_Scanner):
        def run_scan(self, *args, **kwargs):
            if self.calls:
                raise OSError("USB stall")
            return super().run_scan(*args, **kwargs)

    first_source = replace(
        _source(np.arange(PITCH, dtype=np.uint16)),
        ir_alignment=_multiscale_alignment(),
    )
    workflow = FullNegativeWorkflow(
        scanner=FailingSecond([first_source]),
        output_dir=tmp_path,
        identity=_identity(),
        approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
        pitch_rows=PITCH,
    )
    with pytest.raises(OSError, match="USB stall"):
        workflow.capture_frame(
            device_id="coolscan3:usb:test",
            registration=_registration(),
            recipe=_recipe(),
            stop=threading.Event(),
        )

    second_source = replace(
        _source(np.arange(PITCH, PITCH + PHASE, dtype=np.uint16)),
        ir_alignment=_multiscale_alignment(),
    )
    completed = FullNegativeWorkflow(
        scanner=_Scanner([second_source]),
        output_dir=tmp_path,
        identity=_identity(),
        approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
        pitch_rows=PITCH,
    ).capture_frame(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=threading.Event(),
    )

    assert completed.manifest_path.is_file()
    manifest = json.loads(completed.manifest_path.read_text(encoding="utf-8"))
    restored = manifest["captures"][0]["split_alignment"]
    assert restored["estimator_version"] == 2
    assert restored["multiscale_max_dimensions"] == [1024, 2048]
    assert len(restored["multiscale_channel_shifts_px"]) == 2


def test_completed_manifest_resumes_offline_only_after_live_artifacts_revalidate(tmp_path) -> None:
    initial_scanner = _Scanner(
        [
            _source(np.arange(PITCH, dtype=np.uint16)),
            _source(np.arange(PITCH, PITCH + PHASE, dtype=np.uint16)),
        ]
    )
    first = FullNegativeWorkflow(
        scanner=initial_scanner,
        output_dir=tmp_path,
        identity=_identity(),
        approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
        pitch_rows=PITCH,
    ).capture_frame(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=threading.Event(),
    )
    resume_scanner = _Scanner([])

    resumed = FullNegativeWorkflow(
        scanner=resume_scanner,
        output_dir=tmp_path,
        identity=_identity(),
        approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
        pitch_rows=PITCH,
    ).capture_frame(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=threading.Event(),
    )

    assert resumed.manifest_path == first.manifest_path
    assert resumed.capture is None
    assert resume_scanner.calls == []

    (tmp_path / "final" / "frame037.tif").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="artifact evidence"):
        FullNegativeWorkflow(
            scanner=resume_scanner,
            output_dir=tmp_path,
            identity=_identity(),
            approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
            pitch_rows=PITCH,
        ).capture_frame(
            device_id="coolscan3:usb:test",
            registration=_registration(),
            recipe=_recipe(),
            stop=threading.Event(),
        )
    assert resume_scanner.calls == []


def test_completed_manifest_resume_rejects_any_capture_recipe_difference(tmp_path) -> None:
    workflow = FullNegativeWorkflow(
        scanner=_Scanner(
            [
                _source(np.arange(PITCH, dtype=np.uint16)),
                _source(np.arange(PITCH, PITCH + PHASE, dtype=np.uint16)),
            ]
        ),
        output_dir=tmp_path,
        identity=_identity(),
        approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
        pitch_rows=PITCH,
    )
    workflow.capture_frame(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=threading.Event(),
    )

    with pytest.raises(RuntimeError, match="request binding"):
        FullNegativeWorkflow(
            scanner=_Scanner([]),
            output_dir=tmp_path,
            identity=_identity(),
            approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
            pitch_rows=PITCH,
        ).capture_frame(
            device_id="coolscan3:usb:test",
            registration=_registration(),
            recipe=replace(_recipe(), auto_exposure=False),
            stop=threading.Event(),
        )


def test_concurrent_workflows_cannot_capture_into_the_same_output(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingScanner(_Scanner):
        def run_scan(self, *args, **kwargs):
            entered.set()
            assert release.wait(timeout=5)
            return super().run_scan(*args, **kwargs)

    first = FullNegativeWorkflow(
        scanner=BlockingScanner(
            [
                _source(np.arange(PITCH, dtype=np.uint16)),
                _source(np.arange(PITCH, PITCH + PHASE, dtype=np.uint16)),
            ]
        ),
        output_dir=tmp_path,
        identity=_identity(),
        approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
        pitch_rows=PITCH,
    )
    failure: list[BaseException] = []

    def run_first() -> None:
        try:
            first.capture_frame(
                device_id="coolscan3:usb:test",
                registration=_registration(),
                recipe=_recipe(),
                stop=threading.Event(),
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failure.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="already in progress"):
            FullNegativeWorkflow(
                scanner=_Scanner([]),
                output_dir=tmp_path,
                identity=_identity(),
                approved_plan_binding_sha256=APPROVED_PLAN_BINDING,
                pitch_rows=PITCH,
            ).capture_frame(
                device_id="coolscan3:usb:test",
                registration=_registration(),
                recipe=_recipe(),
                stop=threading.Event(),
            )
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert failure == []
