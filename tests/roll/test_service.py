"""Tests for RollScanningService: lifecycle orchestration and output writing.

Lifecycle tests use `fake_coolscanpy` (see tests/roll/conftest.py) the
same way test_coolscanpy_roll.py does. write_frame() tests construct a fake
Frame/Receipt directly -- writing to disk never touches coolscanpy itself,
so no module injection is needed for those.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import time
from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import tifffile
from PIL import Image

from negpy.infrastructure.roll import repair as roll_repair
from negpy.services.roll import service as roll_service
from negpy.services.roll import exact_color
from negpy.services.roll.portable_cms import PortableCMSOnEvaluator
from negpy.services.roll.service import RollScanningError, RollScanningService
from tests.roll._exact_fixtures import make_stage3_replay_receipt, production_cms_payload


class TestAvailable:
    def test_reexports_backend_availability(self, fake_coolscanpy) -> None:
        assert roll_service.available() is True


class TestRollLifecycle:
    def _open_service(self, fake_coolscanpy, roll=None):
        roll = roll if roll is not None else fake_coolscanpy.Roll()
        device = fake_coolscanpy.Device(roll)
        fake_coolscanpy.state["open_device"] = device
        service = RollScanningService()
        service.open_roll("ls5000-usb-001")
        return service, roll, device

    def test_open_then_close(self, fake_coolscanpy) -> None:
        service, roll, device = self._open_service(fake_coolscanpy)
        service.close()
        assert roll.closed is True
        assert device.closed is True

    def test_open_forwards_caller_owned_attempts_root(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        sentinel = object()
        calls: list[tuple[object, object, object]] = []

        def open_roll(device_id, *, material=None, attempts_root=None):
            calls.append((device_id, material, attempts_root))
            return sentinel

        monkeypatch.setattr(roll_service.coolscanpy_roll, "open_roll", open_roll)
        service = RollScanningService()
        evidence = tmp_path / "scanner-attempts"
        service.open_roll("ls5000-usb-001", attempts_root=evidence)

        assert calls == [("ls5000-usb-001", None, evidence)]

    def test_failed_close_retains_the_open_handle_for_a_safe_retry(self, fake_coolscanpy) -> None:
        ownership_error = RuntimeError("USB ownership is retained")

        class UncertainRoll(fake_coolscanpy.Roll):
            def close(self) -> None:
                raise ownership_error

        service, _roll, device = self._open_service(
            fake_coolscanpy,
            UncertainRoll(),
        )

        with pytest.raises(RuntimeError) as raised:
            service.close()

        assert raised.value is ownership_error
        assert service._roll is not None
        assert device.closed is False

    def test_double_open_raises(self, fake_coolscanpy) -> None:
        service, _roll, _device = self._open_service(fake_coolscanpy)
        with pytest.raises(RollScanningError, match="already open"):
            service.open_roll()

    def test_methods_before_open_raise(self) -> None:
        service = RollScanningService()
        with pytest.raises(RollScanningError, match="no roll is open"):
            service.preview()

    def test_close_without_open_is_a_no_op(self) -> None:
        RollScanningService().close()  # must not raise

    def test_preview_and_approve_delegate_to_handle(self, fake_coolscanpy) -> None:
        approval = object()

        class ReturningApprovalRoll(fake_coolscanpy.Roll):
            def approve(self, slot):
                super().approve(slot)
                return approval

        thumb = fake_coolscanpy.Thumbnail(
            slot=1,
            image=np.zeros((2, 2, 3)),
            boundary_rows=(0, 2),
            spacing_offset=0,
            needs_approval=True,
        )
        service, roll, _device = self._open_service(
            fake_coolscanpy,
            ReturningApprovalRoll(thumbnails=[thumb]),
        )

        assert service.preview() == [thumb]
        assert service.approve(1) is approval
        assert roll.approved == [1]

    def test_restore_preview_session_delegates_to_open_handle(self, fake_coolscanpy) -> None:
        thumbnails = [
            fake_coolscanpy.Thumbnail(
                slot=slot,
                image=np.zeros((2, 2, 3)),
                boundary_rows=(0, 2),
                spacing_offset=0,
                needs_approval=slot == 1,
            )
            for slot in (1, 2)
        ]
        service, roll, _device = self._open_service(
            fake_coolscanpy,
            fake_coolscanpy.Roll(thumbnails=thumbnails),
        )

        result = service.restore_preview_session("saved-session", [2])

        assert [thumbnail.slot for thumbnail in result] == [2]
        assert roll.restore_preview_session_calls == [("saved-session", (2,))]

    def test_restore_preview_session_requires_an_open_roll(self) -> None:
        service = RollScanningService()

        with pytest.raises(RollScanningError, match="no roll is open"):
            service.restore_preview_session("saved-session")

    def test_scan_many_delegates_to_handle(self, fake_coolscanpy) -> None:
        frame = fake_coolscanpy.Frame(slot=1, rgb=np.zeros((2, 2, 3), dtype=np.uint16), ir=None, ir_validity=None, receipt=None)
        service, _roll, _device = self._open_service(fake_coolscanpy, fake_coolscanpy.Roll(frames=[frame]))

        assert list(service.scan_many([1])) == [frame]

    def test_safe_stop_before_open_is_a_no_op(self) -> None:
        RollScanningService().safe_stop()  # must not raise

    def test_context_manager_closes(self, fake_coolscanpy) -> None:
        roll = fake_coolscanpy.Roll()
        device = fake_coolscanpy.Device(roll)
        fake_coolscanpy.state["open_device"] = device

        with RollScanningService() as service:
            service.open_roll()
        assert roll.closed is True


class TestWriteFrame:
    def _frame(self, fake_coolscanpy, *, slot=7, ir=None, meter_rgbi=None):
        rgb = np.random.randint(0, 65535, (40, 60, 3), dtype=np.uint16)
        receipt = fake_coolscanpy.Receipt(version=1, slot=slot, dpi=4000, depth=16, device_id="usb:1:2", transport_smear_verdict="clean")
        return fake_coolscanpy.Frame(slot=slot, rgb=rgb, ir=ir, ir_validity=None, receipt=receipt, meter_rgbi=meter_rgbi)

    def test_writes_rgb_tiff(self, fake_coolscanpy, tmp_path) -> None:
        frame = self._frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ date }}_slot{{ "%02d" % seq }}')

        assert os.path.exists(output.rgb_path)
        assert output.rgb_path.endswith(".tif")
        readback = tifffile.imread(output.rgb_path)
        assert readback.shape == (40, 60, 3)
        assert readback.dtype == np.uint16
        np.testing.assert_array_equal(readback, frame.rgb)

    def test_seq_seeded_from_slot_number(self, fake_coolscanpy, tmp_path) -> None:
        frame = self._frame(fake_coolscanpy, slot=23)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}')

        assert "023" in os.path.basename(output.rgb_path)

    def test_writes_ir_sidecar_when_present(self, fake_coolscanpy, tmp_path) -> None:
        ir = np.random.randint(0, 65535, (40, 60), dtype=np.uint16)
        frame = self._frame(fake_coolscanpy, ir=ir)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}')

        assert output.ir_path is not None
        assert output.ir_path.endswith("_IR.tif")
        assert os.path.exists(output.ir_path)
        readback = tifffile.imread(output.ir_path)
        np.testing.assert_array_equal(readback, ir)

    def test_no_ir_sidecar_when_absent(self, fake_coolscanpy, tmp_path) -> None:
        frame = self._frame(fake_coolscanpy, ir=None)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}')

        assert output.ir_path is None
        assert not any(name.endswith("_IR.tif") for name in os.listdir(tmp_path))

    def test_writes_receipt_json_sidecar(self, fake_coolscanpy, tmp_path) -> None:
        frame = self._frame(fake_coolscanpy, slot=9)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}')

        assert output.receipt_path.endswith("_receipt.json")
        with open(output.receipt_path) as fh:
            payload = json.load(fh)
        assert payload["slot"] == 9
        assert payload["dpi"] == 4000
        assert payload["transport_smear_verdict"] == "clean"

    def test_writes_current_coolscan_receipt_with_immutable_artifacts(self, tmp_path) -> None:
        from coolscanpy.types import (
            ArtifactEvidence,
            ClippingTelemetry,
            ExposureVector,
            FocusDetailTelemetry,
            Frame,
            Receipt,
            TransportSmearAssessment,
        )

        rgb = np.zeros((2, 3, 3), dtype=np.uint16)
        artifact = ArtifactEvidence(
            sha256=sha256(memoryview(rgb).cast("B")).hexdigest(),
            byte_length=rgb.nbytes,
            shape=rgb.shape,
            dtype=rgb.dtype.str,
        )
        receipt = Receipt(
            version=1,
            slot=1,
            spacing_offset=0,
            dpi=4000,
            depth=16,
            device_id="usb:2:7",
            device_model="LS-5000 ED",
            reviewed_fingerprint_sha256="a" * 64,
            fresh_fingerprint_sha256="a" * 64,
            manual_approval=None,
            exposure=ExposureVector(
                focus_position=1,
                exposure_multiplier=1.0,
                red_exposure_us=1.0,
                green_exposure_us=1.0,
                blue_exposure_us=1.0,
            ),
            split_alignment=None,
            clipping=ClippingTelemetry(
                fractions=(0.0, 0.0, 0.0),
                clip_level=65535.0,
                warning_fraction=0.01,
                warning=False,
            ),
            focus_detail=FocusDetailTelemetry(
                method="laplacian",
                verdict="measured",
                score=1.0,
                texture_span=1.0,
            ),
            transport_smear=TransportSmearAssessment(
                verdict="clean",
                start_row=None,
                suffix_rows=0,
                minimum_matches=0,
                tail_median_rms=None,
                tail_min_corr=None,
                pre_tail_median_rms=None,
                texture_span=None,
                reason="no repeated tail",
            ),
            artifacts={"rgb": artifact},
        )
        assert type(receipt.artifacts).__name__ == "_ImmutableArtifacts"
        frame = Frame(
            slot=1,
            rgb=rgb,
            ir=None,
            ir_validity=None,
            receipt=receipt,
        )

        output = RollScanningService().write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
        )

        with open(output.receipt_path, encoding="utf-8") as stream:
            payload = json.load(stream)
        assert payload["artifacts"] == {
            "rgb": {
                "byte_length": rgb.nbytes,
                "dtype": rgb.dtype.str,
                "sha256": artifact.sha256,
                "shape": list(rgb.shape),
            }
        }

    def test_rescanning_same_slot_overwrites(self, fake_coolscanpy, tmp_path) -> None:
        service = RollScanningService()
        pattern = '{{ "%03d" % seq }}'

        first = service.write_frame(self._frame(fake_coolscanpy, slot=4), str(tmp_path), pattern)
        second_frame = self._frame(fake_coolscanpy, slot=4)
        second = service.write_frame(second_frame, str(tmp_path), pattern)

        assert first.rgb_path == second.rgb_path
        readback = tifffile.imread(second.rgb_path)
        np.testing.assert_array_equal(readback, second_frame.rgb)

    def test_creates_output_folder(self, fake_coolscanpy, tmp_path) -> None:
        nested = tmp_path / "does" / "not" / "exist" / "yet"
        frame = self._frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(nested), '{{ "%03d" % seq }}')

        assert os.path.exists(output.rgb_path)


def _tier_frame(
    fake_coolscanpy,
    *,
    slot=11,
    ir=True,
    seed=0,
    meter_rgbi=None,
    attempt=1,
):
    """A frame with real per-pixel variance (not a flat fill) so a rendered
    Tier-3 positive has something non-degenerate to measure -- red-biased
    like a plausible C41 negative, matching tests/roll/test_positive.py's
    own synthetic data."""
    rng = np.random.default_rng(seed)
    shape = (24, 32)
    rgb = np.zeros((*shape, 3), dtype=np.uint16)
    rgb[..., 0] = rng.integers(40000, 60000, size=shape)
    rgb[..., 1] = rng.integers(25000, 45000, size=shape)
    rgb[..., 2] = rng.integers(15000, 35000, size=shape)
    ir_plane = rng.integers(0, 65535, size=shape, dtype=np.uint16) if ir else None
    acquisition = None
    validity = None
    if ir_plane is not None and 1 <= slot <= 40:
        if meter_rgbi is None:
            meter_rgbi = rng.integers(
                0,
                65535,
                size=(4, 5, 4),
                dtype=np.uint16,
            )
        storage_rgbi = np.dstack((rgb, ir_plane))
        native_rgbi = np.ascontiguousarray(np.rot90(storage_rgbi, k=-1, axes=(0, 1)))
        validity = np.ones(shape, dtype=np.bool_)
        native_validity = np.ascontiguousarray(np.rot90(validity, k=-1, axes=(0, 1)))
        reservation_id = f"reservation-{slot:03d}"
        capture_attempt_id = f"fine-slot-{slot}-attempt-{attempt:03d}"
        acquisition_id, evidence_sha256 = roll_service._derive_digital_ice_producer_binding(
            slot=slot,
            reservation_id=reservation_id,
            capture_attempt_id=capture_attempt_id,
            main_rgbi=native_rgbi,
            prepass_rgbi=meter_rgbi,
            ir_validity=native_validity,
        )
        acquisition = roll_repair.RepairAcquisition.from_arrays(
            acquisition_id=acquisition_id,
            slot=slot,
            reservation_id=reservation_id,
            capture_attempt_id=capture_attempt_id,
            storage_transform=roll_repair.DIGITAL_ICE_STORAGE_TRANSFORM,
            evidence_sha256=evidence_sha256,
            main_rgbi=native_rgbi,
            prepass_rgbi=meter_rgbi,
            ir_validity=native_validity,
        )
    receipt = fake_coolscanpy.Receipt(version=1, slot=slot, dpi=4000, depth=16, device_id="usb:1:2", transport_smear_verdict="clean")
    return fake_coolscanpy.Frame(
        slot=slot,
        rgb=rgb,
        ir=ir_plane,
        ir_validity=validity,
        receipt=receipt,
        meter_rgbi=meter_rgbi,
        digital_ice_acquisition=acquisition,
    )


def _valid_hybrid_result(
    acquisition: roll_repair.RepairAcquisition,
    *,
    routed_pixel_count: int = 1,
    at_floor_pixel_count: int = 1,
) -> roll_repair.RepairResult:
    native_rgb = np.ascontiguousarray(acquisition.main_rgbi[..., :3] + 3)
    routed_mask = np.zeros(acquisition.main_rgbi.shape[:2], dtype=np.bool_)
    routed_mask.reshape(-1)[:routed_pixel_count] = True
    applied_mask = np.ascontiguousarray(routed_mask & acquisition.ir_validity)

    def png(mask: np.ndarray) -> bytes:
        stream = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            stream,
            format="PNG",
        )
        return stream.getvalue()

    routed_png = png(routed_mask)
    applied_png = png(applied_mask)
    storage_mask = acquisition.storage_mask(applied_mask)
    storage_png = png(storage_mask)
    native_hash = sha256(native_rgb.astype("<u2").tobytes()).hexdigest()
    counts = {
        "at_floor_pixels": at_floor_pixel_count,
        "final_regions": 1 if routed_pixel_count else 0,
        "frame_pixels": routed_mask.size,
        "synthesis_pixels": routed_pixel_count,
    }
    routed_u8 = routed_mask.astype(np.uint8) * 255
    receipt_document = {
        "artifacts": [
            {
                "raw_sha256": native_hash,
                "role": "hybrid_output_rgb16",
            },
            {
                "dtype": "|u1",
                "file_sha256": sha256(routed_png).hexdigest(),
                "raw_sha256": sha256(routed_u8.tobytes()).hexdigest(),
                "role": "synthesis_mask_png",
                "shape": list(routed_mask.shape),
            },
        ],
        "composite": {"hybrid_rgb16_raw_sha256": native_hash},
        "routing": {"counts": counts},
        "schema": "fauxce-hybrid-receipt-v2",
        "synthesis": {
            "fraction": routed_pixel_count / routed_mask.size,
            "frame_pixel_count": routed_mask.size,
            "pixel_count": routed_pixel_count,
            "within_budget": True,
        },
    }
    receipt = (
        json.dumps(
            receipt_document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    storage_rgb = acquisition.storage_rgb(native_rgb)
    return roll_repair.RepairResult(
        rgb=storage_rgb,
        engine="digital-fauxice",
        engine_version="0.3.0",
        mode_requested=roll_repair.RepairMode.HYBRID,
        mode_resolved=roll_repair.RepairMode.HYBRID,
        reason="verified hybrid applied",
        acquisition_id=acquisition.acquisition_id,
        slot=acquisition.slot,
        reservation_id=acquisition.reservation_id,
        evidence_sha256=acquisition.evidence_sha256,
        backend_requested="auto",
        backend_used="cpu-fast",
        backend_selection_reason="test",
        native_output_rgb_sha256=native_hash,
        storage_output_rgb_sha256=sha256(storage_rgb.astype("<u2").tobytes()).hexdigest(),
        native_synthesis_mask_png=applied_png,
        native_synthesis_mask_sha256=sha256(applied_png).hexdigest(),
        native_synthesis_mask_shape=applied_mask.shape,
        routed_native_synthesis_mask_png=routed_png,
        routed_native_synthesis_mask_sha256=sha256(routed_png).hexdigest(),
        routed_native_synthesis_mask_shape=routed_mask.shape,
        storage_synthesis_mask_png=storage_png,
        storage_synthesis_mask_sha256=sha256(storage_png).hexdigest(),
        storage_synthesis_mask_shape=storage_mask.shape,
        synthesis_mask_transform=acquisition.storage_transform,
        synthesis_fraction=int(np.count_nonzero(applied_mask)) / applied_mask.size,
        routing_counts=counts,
        hybrid_receipt=receipt,
        hybrid_receipt_sha256=sha256(receipt).hexdigest(),
        hybrid_provenance_class="caller_asserted_bare_npy",
        hybrid_receipt_output_rgb_sha256=native_hash,
    )


def _receipt_blob(receipt_type, payload: dict, *, attested: bool):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return receipt_type(payload=encoded, sha256=sha256(encoded).hexdigest(), attested=attested)


def _replace_cms_payload(
    receipt: exact_color.VerifiedCMSReceipt,
    payload: dict | bytes,
) -> exact_color.VerifiedCMSReceipt:
    encoded = payload if type(payload) is bytes else json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return dataclasses.replace(receipt, payload=encoded, sha256=sha256(encoded).hexdigest())


def _self_attested_cms_receipt(payload: dict) -> exact_color.VerifiedCMSReceipt:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return exact_color.VerifiedCMSReceipt(
        payload=encoded,
        sha256=sha256(encoded).hexdigest(),
        _factory_token=object(),
    )


def _builder_receipt() -> exact_color.ValidatedBuilderReceipt:
    identity = np.arange(65_536, dtype=np.uint16)
    return make_stage3_replay_receipt((identity, identity, identity))


class _ExactStage1Builder:
    def __init__(self) -> None:
        self.calls = []

    def apply(self, rgb, *, builder_receipt):
        self.calls.append((rgb, builder_receipt))
        output = rgb.copy()
        source_hash = exact_color.rgb16_content_sha256(rgb)
        stage1_hash = exact_color.rgb16_content_sha256(output)
        return exact_color.Stage1BuilderResult(
            rgb=output,
            source_rgb_sha256=source_hash,
            stage1_input_rgb_sha256=stage1_hash,
            builder_receipt=builder_receipt,
            application_receipt=_receipt_blob(
                exact_color.VerifiedBuilderApplicationReceipt,
                {
                    "builder_receipt_sha256": builder_receipt.sha256,
                    "fixed_composition": {
                        "lut_sha256": exact_color.FIXED_COMPOSITION_SHA256,
                        "order": "F[B_c(i)]",
                    },
                    "kind": "negpy.verified-stage1-builder-application",
                    "native_per_acquisition_builder": False,
                    "pre_f_lut_sha256": dict(zip(("r", "g", "b"), builder_receipt.pre_f_lut_sha256, strict=True)),
                    "source_rgb_sha256": source_hash,
                    "scope": exact_color.STAGE3_REPLAY_SCOPE,
                    "stage1_input_rgb_sha256": stage1_hash,
                    "stage3_receipt_sha256": builder_receipt.stage3_receipt_sha256,
                    "version": 1,
                },
                attested=True,
            ),
        )


class _ExactColorEvaluator:
    def __init__(self) -> None:
        self.calls = []
        self.results = []
        self._evaluator = PortableCMSOnEvaluator(chunk_pixels=17)

    def evaluate(self, rgb, *, builder_receipt):
        self.calls.append((rgb, builder_receipt))
        result = self._evaluator.evaluate(rgb, builder_receipt=builder_receipt)
        self.results.append(result)
        return result


class TestThreeTierWriting:
    """`write_frame`'s three independently-selectable output tiers: unrepaired
    (Tier 1), repaired (Tier 2, needs a registered repair engine), and
    positive (Tier 3, always derived from Tier 2's in-memory result). Naming,
    receipt provenance, and every degrade path get covered here; the plain
    default-settings behavior (Tier 1 only) is already covered above by
    TestWriteFrame, unchanged."""

    def _receipt(self, path: str) -> dict:
        with open(path) as fh:
            return json.load(fh)

    # -- selecting nothing --------------------------------------------------

    def test_no_tier_selected_writes_only_the_receipt(self, fake_coolscanpy, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=False, write_repaired=False, write_positive=False
        )

        assert (output.rgb_path, output.ir_path, output.repaired_rgb_path, output.repaired_ir_path, output.positive_path) == (
            None,
            None,
            None,
            None,
            None,
        )
        assert os.path.exists(output.receipt_path)
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["unrepaired"] == {"written": False, "status": "not selected"}
        assert payload["outputs"]["repaired"] == {"written": False, "status": "not selected"}
        assert payload["outputs"]["positive"] == {"written": False, "status": "not selected"}

    # -- Tier 2 without a registered engine ----------------------------------

    def test_repaired_without_an_engine_degrades_but_unrepaired_still_writes(self, fake_coolscanpy, no_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True)

        assert output.rgb_path is not None and os.path.exists(output.rgb_path)
        assert output.repaired_rgb_path is None
        assert output.repaired_ir_path is None
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["unrepaired"]["written"] is True
        assert payload["outputs"]["repaired"] == {"written": False, "status": "unavailable: no dust-repair engine registered"}

    def test_tier1_retains_replayable_dice_acquisition_when_repair_is_unavailable(
        self,
        fake_coolscanpy,
        no_repair_engine,
        tmp_path,
    ) -> None:
        frame = _tier_frame(fake_coolscanpy, slot=12)
        original = frame.prepare_digital_ice()
        archive = tmp_path / "archive"

        output = RollScanningService().write_frame(
            frame,
            str(archive),
            'nested/{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
        )

        receipt = self._receipt(output.receipt_path)
        evidence = receipt["outputs"]["repair_acquisition_evidence"]
        assert evidence["retained"] is True
        assert receipt["outputs"]["repaired"]["written"] is False
        replay = roll_service.load_repair_acquisition_evidence(evidence["binding"]["path"])
        assert replay.acquisition_id == original.acquisition_id
        assert replay.capture_attempt_id == original.capture_attempt_id
        np.testing.assert_array_equal(replay.main_rgbi, original.main_rgbi)
        np.testing.assert_array_equal(replay.prepass_rgbi, original.prepass_rgbi)
        np.testing.assert_array_equal(replay.ir_validity, original.ir_validity)

        # Artifact references are relative to the binding as well as recorded
        # absolutely in the receipt, so moving the complete archive preserves
        # its ability to replay after the scanner media is gone.
        relative_binding = Path(evidence["binding"]["path"]).relative_to(archive)
        moved_archive = tmp_path / "moved-archive"
        archive.rename(moved_archive)
        moved_replay = roll_service.load_repair_acquisition_evidence(moved_archive / relative_binding)
        np.testing.assert_array_equal(moved_replay.main_rgbi, original.main_rgbi)

    def test_tier1_survives_disclosed_dice_evidence_retention_failure(
        self,
        fake_coolscanpy,
        tmp_path,
        monkeypatch,
    ) -> None:
        real_atomic_write_bytes = roll_service._atomic_write_bytes

        def fail_prepass(path, payload):
            if path.endswith("prepass.rgbi16.npy"):
                raise OSError("synthetic evidence volume failure")
            return real_atomic_write_bytes(path, payload)

        monkeypatch.setattr(roll_service, "_atomic_write_bytes", fail_prepass)
        output = RollScanningService().write_frame(
            _tier_frame(fake_coolscanpy, slot=13),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=False,
        )

        assert output.rgb_path is not None and Path(output.rgb_path).is_file()
        assert output.ir_path is not None and Path(output.ir_path).is_file()
        evidence = self._receipt(output.receipt_path)["outputs"]["repair_acquisition_evidence"]
        assert evidence["retained"] is False
        assert "synthetic evidence volume failure" in evidence["status"]
        assert not (tmp_path / ".negpy-dice-acquisition").exists()

    @pytest.mark.parametrize(
        ("tamper", "match"),
        [
            ("producer-hash", "producer evidence SHA-256 changed"),
            ("missing-prepass", "No such file"),
            ("symlink-prepass", "regular non-symlink"),
            ("storage-rgb", "storage RGB SHA-256 changed"),
        ],
    )
    def test_dice_acquisition_replay_rejects_tampered_or_missing_archive_parts(
        self,
        fake_coolscanpy,
        tmp_path,
        tamper,
        match,
    ) -> None:
        output = RollScanningService().write_frame(
            _tier_frame(fake_coolscanpy, slot=14),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=False,
        )
        evidence = self._receipt(output.receipt_path)["outputs"]["repair_acquisition_evidence"]
        binding_path = Path(evidence["binding"]["path"])
        document = json.loads(binding_path.read_bytes())
        assert str(tmp_path) not in binding_path.read_text(encoding="utf-8")

        if tamper == "producer-hash":
            document["acquisition"]["evidence_sha256"] = "0" * 64
            binding_path.write_bytes(
                json.dumps(
                    document,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )
        elif tamper in {"missing-prepass", "symlink-prepass"}:
            relative = document["artifacts"]["prepass_rgbi"]["relative_path"]
            prepass_path = (binding_path.parent / relative).resolve()
            if tamper == "missing-prepass":
                prepass_path.unlink()
            else:
                real_prepass = prepass_path.with_suffix(".real")
                prepass_path.rename(real_prepass)
                prepass_path.symlink_to(real_prepass.name)
        else:
            assert output.rgb_path is not None
            tifffile.imwrite(
                output.rgb_path,
                np.zeros_like(tifffile.imread(output.rgb_path)),
                photometric="rgb",
            )

        with pytest.raises((OSError, ValueError), match=match):
            roll_service.load_repair_acquisition_evidence(binding_path)

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO test")
    def test_dice_replay_tiff_regular_to_fifo_swap_never_blocks(
        self,
        fake_coolscanpy,
        tmp_path,
        monkeypatch,
    ) -> None:
        output = RollScanningService().write_frame(
            _tier_frame(fake_coolscanpy, slot=14),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
        )
        evidence = self._receipt(output.receipt_path)["outputs"]["repair_acquisition_evidence"]
        binding_path = Path(evidence["binding"]["path"])
        rgb_path = Path(output.rgb_path)
        real_open = roll_service.os.open
        swapped = False

        def swap_before_open(path, flags, *args):
            nonlocal swapped
            if not swapped and Path(path) == rgb_path:
                swapped = True
                rgb_path.unlink()
                os.mkfifo(rgb_path)
            return real_open(path, flags, *args)

        monkeypatch.setattr(roll_service.os, "open", swap_before_open)
        started = time.monotonic()
        with pytest.raises(ValueError, match="regular non-symlink|changed"):
            roll_service.load_repair_acquisition_evidence(binding_path)
        assert time.monotonic() - started < 1.0

    def test_dice_replay_rejects_npy_header_before_array_allocation(
        self,
        fake_coolscanpy,
        tmp_path,
        monkeypatch,
    ) -> None:
        output = RollScanningService().write_frame(
            _tier_frame(fake_coolscanpy, slot=14),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
        )
        evidence = self._receipt(output.receipt_path)["outputs"]["repair_acquisition_evidence"]
        binding_path = Path(evidence["binding"]["path"])
        document = json.loads(binding_path.read_bytes())
        prepass_row = document["artifacts"]["prepass_rgbi"]
        prepass_path = binding_path.parent / prepass_row["relative_path"]
        malicious = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            malicious,
            {
                "descr": "<u2",
                "fortran_order": False,
                "shape": (1_000_000, 1_000_000, 4),
            },
        )
        prepass_path.write_bytes(malicious.getvalue())
        prepass_row["file_sha256"] = sha256(malicious.getvalue()).hexdigest()
        binding_path.write_bytes(
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        monkeypatch.setattr(
            roll_service.np,
            "load",
            lambda *_args, **_kwargs: pytest.fail("np.load must not run before NPY header validation"),
        )

        with pytest.raises(ValueError, match="NPY header geometry"):
            roll_service.load_repair_acquisition_evidence(binding_path)

    def test_rescan_removes_stale_dice_acquisition_evidence(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        service = RollScanningService()
        first = service.write_frame(
            _tier_frame(fake_coolscanpy, slot=15, attempt=1),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
        )
        first_evidence = self._receipt(first.receipt_path)["outputs"]["repair_acquisition_evidence"]
        old_directory = Path(first_evidence["binding"]["path"]).parent
        assert old_directory.is_dir()

        second = service.write_frame(
            _tier_frame(fake_coolscanpy, slot=15, seed=2, attempt=2),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
        )
        second_evidence = self._receipt(second.receipt_path)["outputs"]["repair_acquisition_evidence"]

        assert Path(second_evidence["binding"]["path"]).parent != old_directory
        assert not old_directory.exists()
        assert Path(second_evidence["binding"]["path"]).is_file()

    def test_rescan_preserves_dice_evidence_owned_by_a_sibling_receipt(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        service = RollScanningService()
        first = service.write_frame(
            _tier_frame(fake_coolscanpy, slot=16, attempt=1),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
        )
        first_evidence = self._receipt(first.receipt_path)["outputs"]["repair_acquisition_evidence"]
        old_binding = Path(first_evidence["binding"]["path"])
        sibling_receipt = tmp_path / "shared_receipt.json"
        sibling_receipt.write_text(
            json.dumps({"outputs": {"shared_dice": first_evidence}}),
            encoding="utf-8",
        )

        with pytest.raises(OSError, match="owned by another receipt"):
            service.write_frame(
                _tier_frame(fake_coolscanpy, slot=16, seed=3, attempt=2),
                str(tmp_path),
                '{{ "%03d" % seq }}',
                write_unrepaired=True,
            )

        assert old_binding.is_file()
        replay = roll_service.load_repair_acquisition_evidence(old_binding)
        assert replay.capture_attempt_id.endswith("001")

    def test_positive_without_an_engine_degrades_too_since_it_needs_tier_2(self, fake_coolscanpy, no_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_positive=True)

        assert output.rgb_path is not None and os.path.exists(output.rgb_path)
        assert output.positive_path is None
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["unrepaired"]["written"] is True
        assert payload["outputs"]["positive"]["written"] is False
        assert "Tier 2" in payload["outputs"]["positive"]["status"]
        assert "no dust-repair engine registered" in payload["outputs"]["positive"]["status"]

    # -- Tier 2 with a registered engine -------------------------------------

    def test_repair_runs_scanner_native_then_rotates_once_to_storage(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        native_main = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
        storage_rgbi = np.ascontiguousarray(np.rot90(native_main, k=1, axes=(0, 1)))
        meter = np.arange(2 * 2 * 4, dtype=np.uint16).reshape(2, 2, 4)
        validity = np.ones(native_main.shape[:2], dtype=np.bool_)
        validity[-1, 0] = False
        acquisition_id, evidence_sha256 = roll_service._derive_digital_ice_producer_binding(
            slot=11,
            reservation_id="reservation-011",
            capture_attempt_id="fine-slot-11-attempt-001",
            main_rgbi=native_main,
            prepass_rgbi=meter,
            ir_validity=validity,
        )
        acquisition = roll_repair.RepairAcquisition.from_arrays(
            acquisition_id=acquisition_id,
            slot=11,
            reservation_id="reservation-011",
            capture_attempt_id="fine-slot-11-attempt-001",
            storage_transform=roll_repair.DIGITAL_ICE_STORAGE_TRANSFORM,
            evidence_sha256=evidence_sha256,
            main_rgbi=native_main,
            prepass_rgbi=meter,
            ir_validity=validity,
        )
        receipt = fake_coolscanpy.Receipt(
            version=1,
            slot=11,
            dpi=4000,
            depth=16,
            device_id="usb:1:2",
            transport_smear_verdict="clean",
        )

        class _Frame:
            slot = 11
            rgb = storage_rgbi[..., :3]
            ir = storage_rgbi[..., 3]
            ir_validity = np.rot90(validity, k=1, axes=(0, 1))
            meter_rgbi = meter

            def __init__(self) -> None:
                self.receipt = receipt

            def prepare_digital_ice(self):
                return acquisition

        frame = _Frame()
        fake_repair_engine.transform = lambda rgb: np.ascontiguousarray(rgb + 17)

        output = RollScanningService().write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_repaired=True,
        )

        assert output.repaired_rgb_path is not None
        seen = fake_repair_engine.calls[0][0]
        np.testing.assert_array_equal(seen.main_rgbi, native_main)
        np.testing.assert_array_equal(seen.prepass_rgbi, meter)
        np.testing.assert_array_equal(seen.ir_validity, validity)
        expected = np.ascontiguousarray(np.rot90(native_main[..., :3] + 17, k=1, axes=(0, 1)))
        np.testing.assert_array_equal(
            tifffile.imread(output.repaired_rgb_path),
            expected,
        )

    def test_hybrid_persists_output_aligned_mask_and_native_verification_evidence(
        self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch
    ) -> None:
        frame = _tier_frame(fake_coolscanpy)
        acquisition = frame.prepare_digital_ice()
        native_rgb = np.ascontiguousarray(acquisition.main_rgbi[..., :3] + 3)
        native_mask = np.zeros(acquisition.main_rgbi.shape[:2], dtype=np.bool_)
        native_mask[2, 4] = True
        native_stream = io.BytesIO()
        Image.fromarray(native_mask.astype(np.uint8) * 255, mode="L").save(
            native_stream,
            format="PNG",
        )
        native_mask_png = native_stream.getvalue()
        storage_mask = acquisition.storage_mask(native_mask)
        storage_stream = io.BytesIO()
        Image.fromarray(storage_mask.astype(np.uint8) * 255, mode="L").save(
            storage_stream,
            format="PNG",
        )
        storage_mask_png = storage_stream.getvalue()
        hybrid_output_hash = sha256(native_rgb.astype("<u2").tobytes()).hexdigest()
        hybrid_counts = {
            "at_floor_pixels": 1,
            "final_regions": 1,
            "frame_pixels": native_mask.size,
            "synthesis_pixels": 1,
        }
        hybrid_receipt = json.dumps(
            {
                "artifacts": [
                    {
                        "raw_sha256": hybrid_output_hash,
                        "role": "hybrid_output_rgb16",
                    },
                    {
                        "dtype": "|u1",
                        "file_sha256": sha256(native_mask_png).hexdigest(),
                        "raw_sha256": sha256((native_mask.astype(np.uint8) * 255).tobytes()).hexdigest(),
                        "role": "synthesis_mask_png",
                        "shape": list(native_mask.shape),
                    },
                ],
                "composite": {
                    "hybrid_rgb16_raw_sha256": hybrid_output_hash,
                },
                "routing": {"counts": hybrid_counts},
                "schema": "fauxce-hybrid-receipt-v2",
                "synthesis": {
                    "fraction": 1 / native_mask.size,
                    "frame_pixel_count": native_mask.size,
                    "pixel_count": 1,
                    "within_budget": True,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        hybrid_receipt += b"\n"

        def hybrid_repair(
            acquisition,
            mode,
            *,
            hybrid_runtime=None,
            progress=None,
            cancel=None,
        ):
            return roll_repair.RepairResult(
                rgb=acquisition.storage_rgb(native_rgb),
                engine="digital-fauxice",
                engine_version="0.3.0",
                mode_requested=roll_repair.RepairMode.HYBRID,
                mode_resolved=roll_repair.RepairMode.HYBRID,
                reason="verified hybrid applied",
                acquisition_id=acquisition.acquisition_id,
                slot=acquisition.slot,
                reservation_id=acquisition.reservation_id,
                evidence_sha256=acquisition.evidence_sha256,
                backend_requested="auto",
                backend_used="cpu-fast",
                backend_selection_reason="parity self-test passed",
                native_output_rgb_sha256=sha256(native_rgb.astype("<u2").tobytes()).hexdigest(),
                storage_output_rgb_sha256=sha256(acquisition.storage_rgb(native_rgb).astype("<u2").tobytes()).hexdigest(),
                native_synthesis_mask_png=native_mask_png,
                native_synthesis_mask_sha256=sha256(native_mask_png).hexdigest(),
                native_synthesis_mask_shape=native_mask.shape,
                routed_native_synthesis_mask_png=native_mask_png,
                routed_native_synthesis_mask_sha256=sha256(native_mask_png).hexdigest(),
                routed_native_synthesis_mask_shape=native_mask.shape,
                storage_synthesis_mask_png=storage_mask_png,
                storage_synthesis_mask_sha256=sha256(storage_mask_png).hexdigest(),
                storage_synthesis_mask_shape=storage_mask.shape,
                synthesis_mask_transform=acquisition.storage_transform,
                synthesis_fraction=1 / native_mask.size,
                routing_counts=hybrid_counts,
                hybrid_receipt=hybrid_receipt,
                hybrid_receipt_sha256=sha256(hybrid_receipt).hexdigest(),
                hybrid_provenance_class="caller_asserted_bare_npy",
                hybrid_receipt_output_rgb_sha256=hybrid_output_hash,
            )

        fake_repair_engine.repair = hybrid_repair
        output = RollScanningService().write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_repaired=True,
            repair_mode="hybrid",
        )

        assert output.synthesis_mask_path is not None
        with Image.open(output.synthesis_mask_path) as image:
            np.testing.assert_array_equal(
                np.asarray(image.convert("L")) != 0,
                storage_mask,
            )
        assert Path(output.native_synthesis_mask_path).read_bytes() == native_mask_png
        assert Path(output.hybrid_receipt_path).read_bytes() == hybrid_receipt
        payload = self._receipt(output.receipt_path)["outputs"]["repaired"]
        assert payload["mode_requested"] == "hybrid"
        assert payload["mode_resolved"] == "hybrid"
        assert payload["degraded"] is False
        assert payload["acquisition"]["main_rgbi_sha256"] == acquisition.main_rgbi_sha256
        applied = payload["disclosure_mask"]["applied_final"]
        assert applied["storage"]["path"] == output.synthesis_mask_path
        assert applied["transform"] == acquisition.storage_transform
        routed = payload["disclosure_mask"]["routed_raw"]
        assert routed["routing_counts"] == hybrid_counts
        assert Path(routed["native"]["path"]).read_bytes() == native_mask_png
        assert payload["hybrid_receipt"]["path"] == output.hybrid_receipt_path
        assert payload["hybrid_receipt"]["provenance_class"] == "caller_asserted_bare_npy"

        real_atomic_write_bytes = roll_service._atomic_write_bytes

        def fail_after_partial_hybrid_evidence(path, contents):
            if path.endswith("synth-mask-routed-scanner-native.png"):
                raise OSError("synthetic partial hybrid evidence failure")
            return real_atomic_write_bytes(path, contents)

        monkeypatch.setattr(
            roll_service,
            "_atomic_write_bytes",
            fail_after_partial_hybrid_evidence,
        )
        degraded = RollScanningService().write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            repair_mode="hybrid",
        )
        assert degraded.rgb_path is not None and Path(degraded.rgb_path).is_file()
        assert degraded.repaired_rgb_path is None
        assert degraded.synthesis_mask_path is None
        degraded_repair = self._receipt(degraded.receipt_path)["outputs"]["repaired"]
        assert degraded_repair["written"] is False
        assert "partial hybrid evidence failure" in degraded_repair["status"]
        assert not (tmp_path / ".negpy-dice-hybrid").exists()
        assert not any(path.name.startswith(".negpy-frame-stage-") for path in tmp_path.iterdir())
        monkeypatch.setattr(
            roll_service,
            "_atomic_write_bytes",
            real_atomic_write_bytes,
        )

        stale_paths = (
            output.repaired_rgb_path,
            output.repaired_ir_path,
            output.synthesis_mask_path,
            output.native_synthesis_mask_path,
            output.hybrid_receipt_path,
        )
        replacement = RollScanningService().write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_repaired=False,
            write_positive=False,
        )

        assert all(path is not None and not os.path.exists(path) for path in stale_paths)
        replaced_payload = self._receipt(replacement.receipt_path)["outputs"]
        assert replaced_payload["repaired"]["status"] == "not selected"

    def test_hybrid_result_binding_accepts_synthesis_context_outside_at_floor_evidence(
        self,
        fake_coolscanpy,
    ) -> None:
        acquisition = _tier_frame(fake_coolscanpy).prepare_digital_ice()
        result = _valid_hybrid_result(
            acquisition,
            routed_pixel_count=2,
            at_floor_pixel_count=1,
        )

        roll_service._validate_repair_result_binding(
            acquisition,
            result,
            requested_mode=roll_repair.RepairMode.HYBRID,
        )

    @pytest.mark.parametrize(
        ("tamper", "match"),
        [
            ("mode", "resolved mode is invalid"),
            ("native-output", "scanner-native RGB SHA-256 changed"),
            ("receipt-output", "receipt output binding changed"),
            ("receipt-composite", "receipt output binding changed"),
            ("routing-regions", "routing counts disagree"),
            ("noncanonical-receipt", "not canonical"),
            ("permuted-mask", "routed mask binding changed"),
        ],
    )
    def test_hybrid_result_binding_rejects_tampered_result_surface(
        self,
        fake_coolscanpy,
        tamper,
        match,
    ) -> None:
        acquisition = _tier_frame(fake_coolscanpy).prepare_digital_ice()
        result = _valid_hybrid_result(acquisition)

        if tamper == "mode":
            result = dataclasses.replace(result, mode_resolved=None)
        elif tamper == "native-output":
            result = dataclasses.replace(
                result,
                native_output_rgb_sha256="0" * 64,
            )
        elif tamper in {
            "receipt-output",
            "receipt-composite",
            "routing-regions",
        }:
            document = json.loads(result.hybrid_receipt)
            routing_counts = result.routing_counts
            if tamper == "receipt-output":
                document["artifacts"][0]["raw_sha256"] = "0" * 64
            elif tamper == "receipt-composite":
                document["composite"]["hybrid_rgb16_raw_sha256"] = "0" * 64
            else:
                document["routing"]["counts"]["final_regions"] = 2
                routing_counts = {**routing_counts, "final_regions": 2}
            receipt = (
                json.dumps(
                    document,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )
            result = dataclasses.replace(
                result,
                hybrid_receipt=receipt,
                hybrid_receipt_sha256=sha256(receipt).hexdigest(),
                routing_counts=routing_counts,
            )
        elif tamper == "noncanonical-receipt":
            receipt = json.dumps(
                json.loads(result.hybrid_receipt),
                indent=2,
            ).encode()
            result = dataclasses.replace(
                result,
                hybrid_receipt=receipt,
                hybrid_receipt_sha256=sha256(receipt).hexdigest(),
            )
        else:
            routed = np.zeros(acquisition.main_rgbi.shape[:2], dtype=np.bool_)
            routed.reshape(-1)[1] = True
            applied = np.ascontiguousarray(routed & acquisition.ir_validity)

            def png(mask):
                stream = io.BytesIO()
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                    stream,
                    format="PNG",
                )
                return stream.getvalue()

            routed_png = png(routed)
            applied_png = png(applied)
            storage_png = png(acquisition.storage_mask(applied))
            result = dataclasses.replace(
                result,
                routed_native_synthesis_mask_png=routed_png,
                routed_native_synthesis_mask_sha256=sha256(routed_png).hexdigest(),
                native_synthesis_mask_png=applied_png,
                native_synthesis_mask_sha256=sha256(applied_png).hexdigest(),
                storage_synthesis_mask_png=storage_png,
                storage_synthesis_mask_sha256=sha256(storage_png).hexdigest(),
            )

        with pytest.raises(ValueError, match=match):
            roll_service._validate_repair_result_binding(
                acquisition,
                result,
                requested_mode=roll_repair.RepairMode.HYBRID,
            )

    def test_service_snapshots_mutable_hybrid_result_before_validation_and_write(
        self,
        fake_coolscanpy,
        fake_repair_engine,
        tmp_path,
        monkeypatch,
    ) -> None:
        frame = _tier_frame(fake_coolscanpy, slot=18)
        acquisition = frame.prepare_digital_ice()
        valid = _valid_hybrid_result(acquisition)
        mutable_rgb = np.array(valid.rgb, copy=True)
        mutable_counts = dict(valid.routing_counts)
        returned = dataclasses.replace(
            valid,
            rgb=mutable_rgb,
            routing_counts=mutable_counts,
        )
        expected = mutable_rgb.copy()
        fake_repair_engine.repair = lambda *_args, **_kwargs: returned
        real_validate = roll_service._validate_repair_result_binding

        def mutate_producer_after_snapshot(acquisition, result, *, requested_mode):
            mutable_rgb.fill(0)
            mutable_counts["synthesis_pixels"] = 999
            return real_validate(
                acquisition,
                result,
                requested_mode=requested_mode,
            )

        monkeypatch.setattr(
            roll_service,
            "_validate_repair_result_binding",
            mutate_producer_after_snapshot,
        )
        output = RollScanningService().write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_repaired=True,
            repair_mode="hybrid",
        )

        assert output.repaired_rgb_path is not None
        np.testing.assert_array_equal(
            tifffile.imread(output.repaired_rgb_path),
            expected,
        )

    def test_malformed_repair_result_degrades_without_losing_tier1(
        self,
        fake_coolscanpy,
        fake_repair_engine,
        tmp_path,
    ) -> None:
        frame = _tier_frame(fake_coolscanpy, slot=18)
        malformed = dataclasses.replace(
            _valid_hybrid_result(frame.prepare_digital_ice()),
            rgb=object(),
        )
        fake_repair_engine.repair = lambda *_args, **_kwargs: malformed

        output = RollScanningService().write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            repair_mode="hybrid",
        )

        assert output.rgb_path is not None and Path(output.rgb_path).is_file()
        assert output.repaired_rgb_path is None
        repaired = self._receipt(output.receipt_path)["outputs"]["repaired"]
        assert repaired["written"] is False
        assert "repair evidence failed" in repaired["status"]

    def test_tier2_only_rejects_noncanonical_producer_evidence_before_engine(
        self,
        fake_coolscanpy,
        fake_repair_engine,
        tmp_path,
    ) -> None:
        frame = _tier_frame(fake_coolscanpy, slot=18)
        forged = dataclasses.replace(
            frame.prepare_digital_ice(),
            evidence_sha256="0" * 64,
        )
        frame = dataclasses.replace(frame, digital_ice_acquisition=forged)

        output = RollScanningService().write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_repaired=True,
        )

        assert output.repaired_rgb_path is None
        assert fake_repair_engine.calls == []
        repaired = self._receipt(output.receipt_path)["outputs"]["repaired"]
        assert repaired["written"] is False
        assert "producer evidence SHA-256 changed" in repaired["status"]

    def test_repaired_writes_rgb_and_retains_original_ir(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        fake_repair_engine.transform = lambda rgb: np.clip(rgb.astype(np.int32) + 1000, 0, 65535).astype(np.uint16)
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True)

        assert output.repaired_rgb_path is not None
        assert output.repaired_rgb_path.endswith("_repaired.tif")
        readback = tifffile.imread(output.repaired_rgb_path)
        np.testing.assert_array_equal(readback, fake_repair_engine.transform(frame.rgb))
        assert not np.array_equal(readback, frame.rgb)  # actually repaired, not a copy

        assert output.repaired_ir_path is not None
        assert output.repaired_ir_path.endswith("_repaired_IR.tif")
        ir_readback = tifffile.imread(output.repaired_ir_path)
        np.testing.assert_array_equal(ir_readback, frame.ir)  # Tier 1's own IR, unchanged

    def test_transaction_failure_keeps_prior_artifacts_and_receipt(
        self,
        fake_coolscanpy,
        fake_repair_engine,
        tmp_path,
        monkeypatch,
    ) -> None:
        first = _tier_frame(fake_coolscanpy, slot=13, seed=1)
        service = RollScanningService()
        published = service.write_frame(
            first,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
        )
        prior_receipt = Path(published.receipt_path).read_bytes()
        prior_repaired = Path(published.repaired_rgb_path).read_bytes()
        second = _tier_frame(fake_coolscanpy, slot=13, seed=2)

        def fail_receipt(_path, _payload):
            raise OSError("synthetic receipt staging failure")

        monkeypatch.setattr(roll_service, "_atomic_write_json", fail_receipt)
        with pytest.raises(OSError, match="receipt staging failure"):
            service.write_frame(
                second,
                str(tmp_path),
                '{{ "%03d" % seq }}',
                write_unrepaired=True,
                write_repaired=True,
            )

        assert Path(published.receipt_path).read_bytes() == prior_receipt
        assert Path(published.repaired_rgb_path).read_bytes() == prior_repaired
        assert not any(path.name.startswith(".negpy-frame-stage-") for path in tmp_path.iterdir())

    def test_frame_lock_contention_fails_closed_without_blocking(
        self,
        tmp_path,
    ) -> None:
        receipt_path = str(tmp_path / "013_receipt.json")
        first = roll_service._OutputTransaction(str(tmp_path))
        second = roll_service._OutputTransaction(str(tmp_path))
        first._acquire_frame_lock(receipt_path)
        started = time.monotonic()
        try:
            with pytest.raises(OSError, match="busy in another NegPy process"):
                second._acquire_frame_lock(receipt_path)
            assert time.monotonic() - started < 0.5
        finally:
            second.abort()
            first.abort()

        # The failed contender closed its descriptor and the owner released
        # normally, so a later transaction can acquire the same frame lock.
        retry = roll_service._OutputTransaction(str(tmp_path))
        try:
            retry._acquire_frame_lock(receipt_path)
        finally:
            retry.abort()
        assert not any(path.name.startswith(".negpy-frame-stage-") for path in tmp_path.iterdir())

    def test_frame_publication_fails_closed_when_platform_has_no_fcntl(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        transaction = roll_service._OutputTransaction(str(tmp_path))
        monkeypatch.setattr(roll_service, "fcntl", None)
        try:
            with pytest.raises(OSError, match="locking is unavailable"):
                transaction._acquire_frame_lock(str(tmp_path / "013_receipt.json"))
        finally:
            transaction.abort()
        assert not any(path.name.startswith(".negpy-frame-stage-") for path in tmp_path.iterdir())

    def test_receipt_commit_failure_rolls_back_new_derived_artifacts(
        self,
        fake_coolscanpy,
        fake_repair_engine,
        tmp_path,
        monkeypatch,
    ) -> None:
        service = RollScanningService()
        first = service.write_frame(
            _tier_frame(fake_coolscanpy, slot=14, seed=1),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
        )
        prior_receipt = Path(first.receipt_path).read_bytes()
        prior_repaired = Path(first.repaired_rgb_path).read_bytes()
        real_replace = roll_service.os.replace
        failed = False

        def replace_with_failed_receipt_commit(source, destination):
            nonlocal failed
            if (
                not failed
                and os.path.abspath(destination) == os.path.abspath(first.receipt_path)
                and ".negpy-frame-stage-" in os.path.abspath(source)
                and "backups" not in os.path.abspath(source)
            ):
                failed = True
                raise OSError("synthetic receipt commit failure")
            return real_replace(source, destination)

        monkeypatch.setattr(roll_service.os, "replace", replace_with_failed_receipt_commit)
        with pytest.raises(OSError, match="receipt commit failure"):
            service.write_frame(
                _tier_frame(fake_coolscanpy, slot=14, seed=2),
                str(tmp_path),
                '{{ "%03d" % seq }}',
                write_unrepaired=True,
                write_repaired=True,
            )

        assert failed is True
        assert Path(first.receipt_path).read_bytes() == prior_receipt
        assert Path(first.repaired_rgb_path).read_bytes() == prior_repaired

    def test_incomplete_rollback_retains_prior_file_in_recovery_directory(
        self,
        fake_coolscanpy,
        fake_repair_engine,
        tmp_path,
        monkeypatch,
    ) -> None:
        service = RollScanningService()
        first = service.write_frame(
            _tier_frame(fake_coolscanpy, slot=15, seed=1),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
        )
        prior_repaired = Path(first.repaired_rgb_path).read_bytes()
        real_replace = roll_service.os.replace
        receipt_commit_failed = False
        repair_restore_failed = False

        def fail_commit_and_one_restore(source, destination):
            nonlocal receipt_commit_failed, repair_restore_failed
            source_path = os.path.abspath(source)
            destination_path = os.path.abspath(destination)
            if (
                not receipt_commit_failed
                and destination_path == os.path.abspath(first.receipt_path)
                and ".negpy-frame-stage-" in source_path
                and "backups" not in source_path
            ):
                receipt_commit_failed = True
                raise OSError("synthetic receipt commit failure")
            if (
                receipt_commit_failed
                and not repair_restore_failed
                and destination_path == os.path.abspath(first.repaired_rgb_path)
                and "backups" in source_path
            ):
                repair_restore_failed = True
                raise OSError("synthetic repaired restore failure")
            return real_replace(source, destination)

        monkeypatch.setattr(
            roll_service.os,
            "replace",
            fail_commit_and_one_restore,
        )
        with pytest.raises(roll_service.OutputRollbackError) as raised:
            service.write_frame(
                _tier_frame(fake_coolscanpy, slot=15, seed=2),
                str(tmp_path),
                '{{ "%03d" % seq }}',
                write_unrepaired=True,
                write_repaired=True,
            )

        assert receipt_commit_failed is True
        assert repair_restore_failed is True
        recovery = Path(raised.value.recovery_path)
        assert recovery.is_dir()
        assert recovery.name.startswith(".negpy-recovery-")
        manifest = json.loads((recovery / "RECOVERY.json").read_text())
        backup_relative = manifest["unrestored_backups"][os.path.abspath(first.repaired_rgb_path)]
        assert (recovery / backup_relative).read_bytes() == prior_repaired

    def test_output_parent_symlink_cannot_escape_selected_folder(
        self,
        fake_coolscanpy,
        fake_repair_engine,
        tmp_path,
    ) -> None:
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir()
        (tmp_path / "sub").symlink_to(outside, target_is_directory=True)

        with pytest.raises(OSError, match="symbolic link"):
            RollScanningService().write_frame(
                _tier_frame(fake_coolscanpy, slot=16),
                str(tmp_path),
                'sub/{{ "%03d" % seq }}',
                write_unrepaired=True,
            )

        assert list(outside.iterdir()) == []
        assert not any(path.name.startswith(".negpy-frame-stage-") for path in tmp_path.iterdir())

    def test_unreferenced_staged_output_aborts_publication(
        self,
        fake_coolscanpy,
        tmp_path,
        monkeypatch,
    ) -> None:
        real_atomic_write_json = roll_service._atomic_write_json

        def stage_orphan_before_receipt(path, payload):
            roll_service._atomic_write_bytes(
                str(tmp_path / "unreferenced.bin"),
                b"orphan",
            )
            return real_atomic_write_json(path, payload)

        monkeypatch.setattr(
            roll_service,
            "_atomic_write_json",
            stage_orphan_before_receipt,
        )
        with pytest.raises(OSError, match="absent from its receipt"):
            RollScanningService().write_frame(
                _tier_frame(fake_coolscanpy, slot=17),
                str(tmp_path),
                '{{ "%03d" % seq }}',
                write_unrepaired=True,
            )

        assert not (tmp_path / "unreferenced.bin").exists()
        assert not any(path.name.endswith("_receipt.json") for path in tmp_path.iterdir())
        assert not any(path.name.startswith(".negpy-frame-stage-") for path in tmp_path.iterdir())

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO test")
    def test_existing_fifo_receipt_fails_without_blocking_or_publishing(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        receipt_path = tmp_path / "019_receipt.json"
        os.mkfifo(receipt_path)
        started = time.monotonic()

        with pytest.raises(OSError, match="regular non-symlink"):
            RollScanningService().write_frame(
                _tier_frame(fake_coolscanpy, slot=19),
                str(tmp_path),
                '{{ "%03d" % seq }}',
                write_unrepaired=True,
            )

        assert time.monotonic() - started < 1.0
        assert not (tmp_path / "019.tif").exists()
        assert not (tmp_path / "019_IR.tif").exists()

    @pytest.mark.parametrize(
        "hostile_receipt",
        [
            b'{"outputs":{"first":{"path":"owned"}},"outputs":{}}',
            b'{"outputs":{"value":NaN}}',
        ],
    )
    def test_unsafe_sibling_receipt_conservatively_blocks_overwrite(
        self,
        fake_coolscanpy,
        tmp_path,
        hostile_receipt,
    ) -> None:
        (tmp_path / "hostile_receipt.json").write_bytes(hostile_receipt)

        with pytest.raises(OSError, match="owned by another receipt"):
            RollScanningService().write_frame(
                _tier_frame(fake_coolscanpy, slot=20),
                str(tmp_path),
                '{{ "%03d" % seq }}',
                write_unrepaired=True,
            )

        assert not (tmp_path / "020.tif").exists()
        assert not any(path.name.startswith(".negpy-frame-stage-") for path in tmp_path.iterdir())

    def test_receipt_rejects_non_json_values_instead_of_stringifying_them(
        self,
        fake_coolscanpy,
        tmp_path,
    ) -> None:
        @dataclasses.dataclass(frozen=True)
        class InvalidReceipt:
            version: int
            slot: int
            opaque: object

        frame = dataclasses.replace(
            _tier_frame(fake_coolscanpy, slot=18),
            receipt=InvalidReceipt(version=1, slot=18, opaque=object()),
        )

        with pytest.raises(TypeError, match="not JSON serializable"):
            RollScanningService().write_frame(
                frame,
                str(tmp_path),
                '{{ "%03d" % seq }}',
                write_unrepaired=True,
            )

        assert not any(path.name.endswith("_receipt.json") for path in tmp_path.iterdir())
        assert not any(path.name.endswith(".tif") for path in tmp_path.iterdir())

    def test_repaired_forwards_the_coolscanpy_meter_prepass_unchanged(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        prepass = np.zeros((4, 5, 4), dtype=np.uint16)
        frame = _tier_frame(
            fake_coolscanpy,
            ir=True,
            meter_rgbi=prepass,
        )

        RollScanningService().write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=False, write_repaired=True)

        assert len(fake_repair_engine.prepasses) == 1
        np.testing.assert_array_equal(fake_repair_engine.prepasses[0], prepass)

    def test_exact_nikon_positive_uses_injected_evaluator_and_binds_both_receipts(
        self, fake_coolscanpy, fake_repair_engine, tmp_path
    ) -> None:
        fake_repair_engine.transform = lambda rgb: np.clip(rgb.astype(np.int32) + 731, 0, 65535).astype(np.uint16)
        builder = _ExactStage1Builder()
        evaluator = _ExactColorEvaluator()
        builder_receipt = _builder_receipt()
        frame = _tier_frame(fake_coolscanpy)

        output = RollScanningService(
            exact_color_builder=builder,
            exact_color_evaluator=evaluator,
        ).write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=builder_receipt,
        )

        assert output.positive_path is not None
        repaired = fake_repair_engine.transform(frame.rgb)
        readback = tifffile.imread(output.positive_path)
        assert readback.dtype == np.uint16
        np.testing.assert_array_equal(readback, evaluator.results[0].rgb)
        with tifffile.TiffFile(output.positive_path) as exact_tiff:
            embedded_profile = exact_tiff.pages[0].tags[34675].value
        assert len(embedded_profile) == 492
        assert sha256(embedded_profile).hexdigest() == "a8d0d753bd6129357cc2647435ce675e8637a679eb526fa180fba460874ce1d3"
        positive_receipt = self._receipt(output.receipt_path)["outputs"]["positive"]
        tiff_artifact = positive_receipt["tiff_artifact"]
        assert tiff_artifact["file_sha256"] == sha256(Path(output.positive_path).read_bytes()).hexdigest()
        assert tiff_artifact["pixel_sha256"] == exact_color.rgb16_content_sha256(readback)
        assert tiff_artifact["icc_sha256"] == sha256(embedded_profile).hexdigest()
        assert tiff_artifact["page_count"] == 1
        assert tiff_artifact["bits_per_sample"] == [16, 16, 16]
        assert tiff_artifact["orientation"] == "top-left"
        np.testing.assert_array_equal(builder.calls[0][0], repaired)
        np.testing.assert_array_equal(evaluator.calls[0][0], repaired)
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert entry["color_mode"] == "nikon-exact"
        assert entry["input_rgb_sha256"] == exact_color.rgb16_content_sha256(repaired)
        assert entry["output_rgb_sha256"] == exact_color.rgb16_content_sha256(readback)
        assert entry["builder_receipt_sha256"] == builder_receipt.sha256
        assert entry["builder_receipt"]["schema"] == exact_color.BUILDER_RECEIPT_SCHEMA
        assert entry["builder_application_receipt"]["stage1_input_rgb_sha256"] == entry["input_rgb_sha256"]
        assert entry["cms_receipt"]["algorithm"] == exact_color.CMS_ALGORITHM_ID
        assert entry["cms_receipt"]["validation"]["mismatched_u16"] == 0
        assert entry["cms_receipt"]["input_rgb_sha256"] == entry["input_rgb_sha256"]
        assert entry["icc_profile"] == {
            "name": "Nikon Adobe RGB 4.0.0.3000",
            "bytes": 492,
            "sha256": "a8d0d753bd6129357cc2647435ce675e8637a679eb526fa180fba460874ce1d3",
        }
        evidence = entry["retained_builder_evidence"]
        assert evidence["scope"] == exact_color.STAGE3_REPLAY_SCOPE
        assert evidence["native_per_acquisition_builder"] is False
        rows = [evidence["stage3_report"], *evidence["pre_f_luts"]]
        for row in rows:
            payload = Path(row["path"]).read_bytes()
            assert len(payload) == row["bytes"]
            assert sha256(payload).hexdigest() == row["sha256"]
        reproduced = exact_color.load_stage3_replay_builder_receipt(evidence["stage3_report"]["path"])
        assert reproduced.sha256 == builder_receipt.sha256
        assert reproduced.pre_f_lut_sha256 == builder_receipt.pre_f_lut_sha256

    @pytest.mark.parametrize("corruption", ["pixels", "icc", "orientation"])
    def test_exact_positive_rejects_tiff_write_corruption_before_publication(
        self,
        corruption,
        fake_coolscanpy,
        fake_repair_engine,
        tmp_path,
        monkeypatch,
    ) -> None:
        builder = _ExactStage1Builder()
        evaluator = _ExactColorEvaluator()
        real_imwrite = roll_service.tifffile.imwrite

        def corrupting_imwrite(path, array, **kwargs):
            if kwargs.get("iccprofile") is not None:
                if corruption == "pixels":
                    array = np.array(array, copy=True)
                    array[0, 0, 0] ^= np.uint16(1)
                elif corruption == "icc":
                    kwargs.pop("iccprofile")
                else:
                    kwargs["extratags"] = ((274, "H", 1, 6, False),)
            return real_imwrite(path, array, **kwargs)

        monkeypatch.setattr(roll_service.tifffile, "imwrite", corrupting_imwrite)
        output = RollScanningService(
            exact_color_builder=builder,
            exact_color_evaluator=evaluator,
        ).write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=_builder_receipt(),
        )

        assert output.rgb_path is not None and Path(output.rgb_path).is_file()
        assert output.repaired_rgb_path is not None and Path(output.repaired_rgb_path).is_file()
        assert output.positive_path is None
        receipt = self._receipt(output.receipt_path)
        assert receipt["outputs"]["unrepaired"]["written"] is True
        assert receipt["outputs"]["repaired"]["written"] is True
        positive = receipt["outputs"]["positive"]
        assert positive["written"] is False
        assert "unavailable: exact Nikon color" in positive["status"]
        assert "exact_nikon_color" not in positive
        assert "retained_builder_evidence" not in positive
        assert not any(path.name.endswith("_positive.tif") for path in tmp_path.iterdir())
        assert not (tmp_path / ".negpy-stage3-replay").exists()
        assert not (tmp_path / ".negpy-native-builder").exists()
        assert not any(path.name.startswith(".negpy-frame-stage-") for path in tmp_path.iterdir())

    def test_exact_positive_detects_path_swap_after_stable_read(
        self,
        fake_coolscanpy,
        fake_repair_engine,
        tmp_path,
        monkeypatch,
    ) -> None:
        real_lstat = roll_service.os.lstat
        positive_lstats = 0

        def swap_before_final_identity_check(path, *args, **kwargs):
            nonlocal positive_lstats
            path_string = os.fspath(path)
            if path_string.endswith("_positive.tif"):
                positive_lstats += 1
                if positive_lstats == 2:
                    replacement = path_string + ".swapped"
                    Path(replacement).write_bytes(b"not the verified TIFF")
                    os.replace(replacement, path_string)
            return real_lstat(path, *args, **kwargs)

        monkeypatch.setattr(
            roll_service.os,
            "lstat",
            swap_before_final_identity_check,
        )
        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=_ExactColorEvaluator(),
        ).write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=_builder_receipt(),
        )

        assert positive_lstats >= 2
        assert output.rgb_path is not None and Path(output.rgb_path).is_file()
        assert output.repaired_rgb_path is not None
        assert output.positive_path is None
        positive = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "changed while it was verified" in positive["status"]
        assert not any(path.name.endswith("_positive.tif") for path in tmp_path.iterdir())

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO test")
    def test_exact_positive_regular_to_fifo_swap_never_blocks(
        self,
        fake_coolscanpy,
        fake_repair_engine,
        tmp_path,
        monkeypatch,
    ) -> None:
        real_open = roll_service.os.open
        builder_receipt = _builder_receipt()
        swapped = False

        def swap_before_open(path, flags, *args, **kwargs):
            nonlocal swapped
            path_string = os.fspath(path)
            if not swapped and path_string.endswith("_positive.tif"):
                swapped = True
                os.unlink(path_string)
                os.mkfifo(path_string)
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(roll_service.os, "open", swap_before_open)
        started = time.monotonic()
        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=_ExactColorEvaluator(),
        ).write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=builder_receipt,
        )

        assert swapped is True
        assert time.monotonic() - started < 1.0
        assert output.positive_path is None
        assert not any(path.name.endswith("_positive.tif") for path in tmp_path.iterdir())

    def test_omitted_c41_positive_mode_defaults_to_fail_closed_nikon_exact(
        self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            roll_service.roll_positive,
            "render_positive",
            lambda *_args, **_kwargs: pytest.fail("approximate color must require explicit selection"),
        )

        output = RollScanningService().write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
        )

        assert output.positive_path is None
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert entry["written"] is False
        assert entry["color_mode"] == "nikon-exact"
        assert "frame has no native builder evidence" in entry["status"]

    def test_exact_nikon_positive_fails_closed_when_embedded_icc_identity_is_invalid(
        self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            roll_service.roll_nikon_icc,
            "nikon_adobe_rgb_profile",
            MagicMock(side_effect=roll_service.roll_nikon_icc.NikonICCProfileError("profile hash mismatch")),
        )

        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=_ExactColorEvaluator(),
        ).write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            builder_receipt=_builder_receipt(),
        )

        assert output.positive_path is None
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert entry["written"] is False
        assert entry["color_mode"] == "nikon-exact"
        assert "profile hash mismatch" in entry["status"]

    def test_exact_nikon_positive_is_unavailable_without_portable_evaluator(
        self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch
    ) -> None:
        builder_receipt = _builder_receipt()
        monkeypatch.setattr(
            roll_service.roll_positive,
            "render_positive",
            lambda *_args, **_kwargs: pytest.fail("approximate renderer must not satisfy exact mode"),
        )

        output = RollScanningService(exact_color_builder=_ExactStage1Builder()).write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=builder_receipt,
        )

        assert output.positive_path is None
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert entry["written"] is False
        assert entry["color_mode"] == "nikon-exact"
        assert "verified portable CMS evaluator is not supplied" in entry["status"]

    def test_exact_nikon_positive_fails_closed_when_replay_evidence_cannot_be_retained(
        self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            roll_service,
            "_atomic_write_bytes",
            MagicMock(side_effect=OSError("read-only evidence volume")),
        )

        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=_ExactColorEvaluator(),
        ).write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=_builder_receipt(),
        )

        assert output.positive_path is None
        assert not (tmp_path / "011_positive.tif").exists()
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "cannot retain Stage-3 replay evidence" in entry["status"]

    def test_exact_nikon_positive_is_unavailable_without_stage1_builder(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        evaluator = _ExactColorEvaluator()

        output = RollScanningService(exact_color_evaluator=evaluator).write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=_builder_receipt(),
        )

        assert output.positive_path is None
        assert evaluator.calls == []
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "verified Stage-1 builder applicator is not supplied" in entry["status"]

    def test_exact_nikon_positive_is_unavailable_without_validated_builder_receipt(
        self, fake_coolscanpy, fake_repair_engine, tmp_path
    ) -> None:
        evaluator = _ExactColorEvaluator()

        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=evaluator,
        ).write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
        )

        assert output.positive_path is None
        assert evaluator.calls == []
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "validated Stage-3 builder receipt is not supplied" in entry["status"]

    def test_exact_nikon_positive_rejects_tampered_builder_receipt(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        evaluator = _ExactColorEvaluator()
        receipt = _builder_receipt()
        tampered = dataclasses.replace(receipt, payload=receipt.payload + b" ")

        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=evaluator,
        ).write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=tampered,
        )

        assert output.positive_path is None
        assert evaluator.calls == []
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "builder receipt payload does not match its SHA-256" in entry["status"]

    def test_exact_nikon_positive_rejects_malformed_cms_receipt(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        builder_receipt = _builder_receipt()
        frame = _tier_frame(fake_coolscanpy)
        valid_result = _ExactColorEvaluator().evaluate(frame.rgb, builder_receipt=builder_receipt)
        malformed_payload = b"this is not a JSON receipt"
        evaluator = MagicMock()
        evaluator.evaluate.return_value = dataclasses.replace(
            valid_result,
            cms_receipt=_replace_cms_payload(valid_result.cms_receipt, malformed_payload),
        )

        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=evaluator,
        ).write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=builder_receipt,
        )

        assert output.positive_path is None
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "CMS receipt is not valid JSON" in entry["status"]

    def test_exact_nikon_positive_rejects_cms_receipt_input_mismatch(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        builder_receipt = _builder_receipt()
        frame = _tier_frame(fake_coolscanpy)
        valid_result = _ExactColorEvaluator().evaluate(frame.rgb, builder_receipt=builder_receipt)
        cms = exact_color.receipt_payload(valid_result.cms_receipt)
        cms["input_rgb_sha256"] = "0" * 64
        evaluator = MagicMock()
        evaluator.evaluate.return_value = dataclasses.replace(
            valid_result,
            cms_receipt=_replace_cms_payload(valid_result.cms_receipt, cms),
        )

        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=evaluator,
        ).write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=builder_receipt,
        )

        assert output.positive_path is None
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "CMS receipt does not bind its builder, input, and output" in entry["status"]

    def test_exact_nikon_positive_rejects_self_attested_xor_evaluator_even_with_full_cms_contract(
        self, fake_coolscanpy, fake_repair_engine, tmp_path
    ) -> None:
        builder_receipt = _builder_receipt()
        frame = _tier_frame(fake_coolscanpy)
        valid_result = _ExactColorEvaluator().evaluate(frame.rgb, builder_receipt=builder_receipt)
        xor_output = np.bitwise_xor(frame.rgb, np.uint16(0xFFFF))
        xor_output_hash = exact_color.rgb16_content_sha256(xor_output)
        forged_payload = production_cms_payload(
            builder_receipt_sha256=builder_receipt.sha256,
            input_rgb_sha256=valid_result.input_rgb_sha256,
            output_rgb_sha256=xor_output_hash,
        )
        evaluator = MagicMock()
        evaluator.evaluate.return_value = dataclasses.replace(
            valid_result,
            rgb=xor_output,
            output_rgb_sha256=xor_output_hash,
            cms_receipt=_self_attested_cms_receipt(forged_payload),
        )

        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=evaluator,
        ).write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=builder_receipt,
        )

        assert output.positive_path is None
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "trusted portable CMS adapter" in entry["status"]

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda cms: cms.update(kind="wrong"), id="kind"),
            pytest.param(lambda cms: cms.update(version=True), id="version-type"),
            pytest.param(lambda cms: cms.update(algorithm="unverified"), id="algorithm"),
            pytest.param(lambda cms: cms["assets"].pop(next(iter(cms["assets"])), None), id="nine-assets"),
            pytest.param(lambda cms: cms["oracle_source"].update(sha256="0" * 64), id="oracle-source"),
            pytest.param(lambda cms: cms["validation"].update(mismatched_u16=1), id="validation"),
            pytest.param(lambda cms: cms.update(scope="expanded"), id="scope"),
            pytest.param(lambda cms: cms.update(stage_order=["stage2", "stage1"]), id="stage-order"),
            pytest.param(lambda cms: cms.update(dll_free=False), id="dll-free"),
            pytest.param(lambda cms: cms.update(upstream_builder_included=True), id="builder-scope"),
            pytest.param(lambda cms: cms.update(chunk_pixels=0), id="chunk-size"),
        ],
    )
    def test_exact_nikon_positive_requires_every_cms_contract_field(self, fake_coolscanpy, fake_repair_engine, tmp_path, mutate) -> None:
        builder_receipt = _builder_receipt()
        frame = _tier_frame(fake_coolscanpy)
        valid_result = _ExactColorEvaluator().evaluate(frame.rgb, builder_receipt=builder_receipt)
        cms = exact_color.receipt_payload(valid_result.cms_receipt)
        mutate(cms)
        evaluator = MagicMock()
        evaluator.evaluate.return_value = dataclasses.replace(
            valid_result,
            cms_receipt=_replace_cms_payload(valid_result.cms_receipt, cms),
        )

        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=evaluator,
        ).write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=builder_receipt,
        )

        assert output.positive_path is None
        assert self._receipt(output.receipt_path)["outputs"]["positive"]["written"] is False

    def test_exact_nikon_positive_rejects_input_identity_mismatch(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        builder_receipt = _builder_receipt()
        frame = _tier_frame(fake_coolscanpy)
        valid_result = _ExactColorEvaluator().evaluate(frame.rgb, builder_receipt=builder_receipt)
        evaluator = MagicMock()
        evaluator.evaluate.return_value = dataclasses.replace(
            valid_result,
            input_rgb_sha256="0" * 64,
        )

        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=evaluator,
        ).write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=builder_receipt,
        )

        assert output.positive_path is None
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "input hash does not match the Stage-1 input content" in entry["status"]

    def test_exact_nikon_positive_rejects_unbound_builder_output(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        builder_receipt = _builder_receipt()
        honest_builder = _ExactStage1Builder()
        builder = MagicMock()

        def unbound(rgb, *, builder_receipt):
            valid = honest_builder.apply(rgb, builder_receipt=builder_receipt)
            return dataclasses.replace(valid, stage1_input_rgb_sha256="0" * 64)

        builder.apply.side_effect = unbound
        evaluator = _ExactColorEvaluator()

        output = RollScanningService(
            exact_color_builder=builder,
            exact_color_evaluator=evaluator,
        ).write_frame(
            _tier_frame(fake_coolscanpy),
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=builder_receipt,
        )

        assert output.positive_path is None
        assert evaluator.calls == []
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "builder output hash does not match the Stage-1 input content" in entry["status"]

    def test_exact_nikon_positive_rejects_unbound_output_hash(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        builder_receipt = _builder_receipt()
        frame = _tier_frame(fake_coolscanpy)
        valid_result = _ExactColorEvaluator().evaluate(frame.rgb, builder_receipt=builder_receipt)
        evaluator = MagicMock()
        evaluator.evaluate.return_value = dataclasses.replace(
            valid_result,
            output_rgb_sha256="f" * 64,
        )

        output = RollScanningService(
            exact_color_builder=_ExactStage1Builder(),
            exact_color_evaluator=evaluator,
        ).write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="nikon-exact",
            builder_receipt=builder_receipt,
        )

        assert output.positive_path is None
        assert not (tmp_path / "011_positive.tif").exists()
        entry = self._receipt(output.receipt_path)["outputs"]["positive"]
        assert "output hash does not match the returned RGB content" in entry["status"]

    def test_repaired_naming_does_not_collide_with_unrepaired(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True)

        paths = {output.rgb_path, output.ir_path, output.repaired_rgb_path, output.repaired_ir_path}
        assert len(paths) == 4  # four distinct files
        assert all(os.path.exists(p) for p in paths)

    def test_repaired_receipt_records_engine_provenance(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=False, write_repaired=True, repair_mode="hybrid"
        )

        payload = self._receipt(output.receipt_path)
        entry = payload["outputs"]["repaired"]
        assert entry["written"] is True
        assert entry["engine"] == "test-repair-engine"
        assert entry["engine_version"] == "0.0.1-test"
        assert entry["mode_requested"] == "hybrid"
        assert entry["mode_resolved"] == "exact"
        assert entry["degraded"] is True
        assert "no hybrid runtime" in entry["reason"]
        assert entry["rgb_path"] == output.repaired_rgb_path

    def test_repair_engine_failure_degrades_without_losing_unrepaired(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        fake_repair_engine.raise_error = RuntimeError("inpainting model unavailable")
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True, write_positive=True
        )

        assert output.rgb_path is not None and os.path.exists(output.rgb_path)
        assert output.repaired_rgb_path is None
        assert output.positive_path is None
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["unrepaired"]["written"] is True
        assert "repair failed" in payload["outputs"]["repaired"]["status"]
        assert "inpainting model unavailable" in payload["outputs"]["repaired"]["status"]
        assert "Tier 2" in payload["outputs"]["positive"]["status"]

    def test_frame_without_infrared_degrades_repaired_and_positive(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy, ir=False)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True, write_positive=True
        )

        assert fake_repair_engine.calls == []  # never even attempted
        assert output.repaired_rgb_path is None
        assert output.positive_path is None
        payload = self._receipt(output.receipt_path)
        assert "no infrared plane" in payload["outputs"]["repaired"]["status"]

    def test_invalid_repair_mode_fails_before_writing_or_repair(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        with pytest.raises(ValueError, match="unknown repair mode"):
            service.write_frame(
                frame,
                str(tmp_path),
                '{{ "%03d" % seq }}',
                write_repaired=True,
                repair_mode="bogus-mode",
            )

        assert fake_repair_engine.calls == []
        assert list(tmp_path.iterdir()) == []

    def test_positive_requested_without_write_repaired_still_repairs_in_memory(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="negpy-approximate",
        )

        assert len(fake_repair_engine.calls) == 1  # repair ran to feed Tier 3...
        assert output.repaired_rgb_path is None  # ...but Tier 2 itself was never written
        assert output.positive_path is not None
        payload = self._receipt(output.receipt_path)
        entry = payload["outputs"]["repaired"]
        assert entry["written"] is False
        assert entry["status"] == "not selected (computed in memory for the positive)"
        assert entry["engine"] == "test-repair-engine"  # still recorded even though unwritten

    # -- Tier 3 -----------------------------------------------------------------

    def test_positive_is_written_and_derived_from_repaired_not_raw_rgb(
        self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch
    ) -> None:
        fake_repair_engine.transform = lambda rgb: np.clip(rgb.astype(np.int32) + 5000, 0, 65535).astype(np.uint16)
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        from negpy.services.roll import positive as roll_positive_module

        captured = {}
        real_render = roll_positive_module.render_positive

        def _spy(rgb_u16, *, processor):
            captured["rgb_u16"] = rgb_u16
            return real_render(rgb_u16, processor=processor)

        monkeypatch.setattr(roll_positive_module, "render_positive", _spy)

        output = service.write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="negpy-approximate",
        )

        np.testing.assert_array_equal(captured["rgb_u16"], fake_repair_engine.transform(frame.rgb))
        assert not np.array_equal(captured["rgb_u16"], frame.rgb)
        assert output.positive_path is not None
        readback = tifffile.imread(output.positive_path)
        assert readback.shape == frame.rgb.shape
        assert readback.dtype == np.uint16

    def test_positive_receipt_records_inversion_and_repair_provenance(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="negpy-approximate",
        )

        payload = self._receipt(output.receipt_path)
        entry = payload["outputs"]["positive"]
        assert entry["written"] is True
        assert entry["rgb_path"] == output.positive_path
        assert entry["color_mode"] == "negpy-approximate"
        assert entry["exact_nikon_color"] is False
        assert entry["inversion_path"] == "negpy.services.rendering.image_processor.ImageProcessor.run_pipeline"
        assert entry["render_intent"] == "print"
        assert entry["process_mode"] == "C41"
        assert entry["auto_exposure"] is True
        assert entry["negpy_version"]
        assert entry["repair_engine"] == "test-repair-engine"
        assert entry["repair_engine_version"] == "0.0.1-test"
        assert entry["repair_mode"] == "exact"
        assert "icc_profile" not in entry
        with tifffile.TiffFile(output.positive_path) as approximate_tiff:
            assert 34675 not in approximate_tiff.pages[0].tags

    def test_positive_filename_has_no_infrared_companion(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=False,
            write_positive=True,
            positive_mode="negpy-approximate",
        )

        assert output.positive_path is not None
        assert output.positive_path.endswith("_positive.tif")
        assert os.path.exists(output.positive_path)
        assert not any(name.endswith("_positive_IR.tif") for name in os.listdir(str(tmp_path)))

    def test_inversion_unavailable_degrades_positive_but_keeps_unrepaired_and_repaired(
        self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch
    ) -> None:
        from negpy.services.roll import positive as roll_positive_module

        monkeypatch.setattr(roll_positive_module, "available", lambda: False)
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            write_positive=True,
            positive_mode="negpy-approximate",
        )

        assert output.rgb_path is not None and os.path.exists(output.rgb_path)
        assert output.repaired_rgb_path is not None and os.path.exists(output.repaired_rgb_path)
        assert output.positive_path is None
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["unrepaired"]["written"] is True
        assert payload["outputs"]["repaired"]["written"] is True
        assert payload["outputs"]["positive"] == {
            "written": False,
            "status": "unavailable: inversion path not available",
            "color_mode": "negpy-approximate",
        }

    def test_inversion_failure_degrades_positive_only(self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch) -> None:
        from negpy.services.roll import positive as roll_positive_module

        def _boom(rgb_u16, *, processor):
            raise RuntimeError("no CPU render backend available")

        monkeypatch.setattr(roll_positive_module, "render_positive", _boom)
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            write_positive=True,
            positive_mode="negpy-approximate",
        )

        assert output.rgb_path is not None and os.path.exists(output.rgb_path)
        assert output.repaired_rgb_path is not None and os.path.exists(output.repaired_rgb_path)
        assert output.positive_path is None
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["repaired"]["written"] is True
        assert "inversion failed" in payload["outputs"]["positive"]["status"]
        assert "no CPU render backend available" in payload["outputs"]["positive"]["status"]

    # -- all three together, and receipt backward-compatibility -----------------

    def test_all_three_tiers_together(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame,
            str(tmp_path),
            '{{ "%03d" % seq }}',
            write_unrepaired=True,
            write_repaired=True,
            write_positive=True,
            positive_mode="negpy-approximate",
        )

        for path in (output.rgb_path, output.ir_path, output.repaired_rgb_path, output.repaired_ir_path, output.positive_path):
            assert path is not None and os.path.exists(path)
        assert len({output.rgb_path, output.ir_path, output.repaired_rgb_path, output.repaired_ir_path, output.positive_path}) == 5
        for untagged_path in (output.rgb_path, output.repaired_rgb_path, output.positive_path):
            with tifffile.TiffFile(untagged_path) as untagged_tiff:
                assert 34675 not in untagged_tiff.pages[0].tags
        payload = self._receipt(output.receipt_path)
        assert all(payload["outputs"][tier]["written"] for tier in ("unrepaired", "repaired", "positive"))

    def test_receipt_keeps_the_original_scan_receipt_fields_alongside_outputs(self, fake_coolscanpy, tmp_path) -> None:
        """`outputs` must be additive -- write_frame's pre-tiering callers (and
        TestWriteFrame's tests above) read coolscanpy's own receipt fields
        (slot, dpi, transport_smear_verdict, ...) at the JSON root."""
        frame = _tier_frame(fake_coolscanpy, slot=42)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}')

        payload = self._receipt(output.receipt_path)
        assert payload["slot"] == 42
        assert payload["dpi"] == 4000
        assert payload["transport_smear_verdict"] == "clean"
        assert "outputs" in payload
