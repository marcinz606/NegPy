"""Wire scanner-native portable Digital ICE into the roll repair seam."""

from __future__ import annotations

import hashlib
import io
import threading
from typing import Callable

import numpy as np
from PIL import Image

from negpy.infrastructure.roll import repair as roll_repair
from negpy.infrastructure.roll.repair import (
    RepairAcquisition,
    RepairMode,
    RepairResult,
)

from negpy.services.repair.fauxice_ir_repair import (
    FauxiceRepairCancelled,
    FauxiceRepairConfig,
    engine_available,
    repair_ir_dust,
)


def _raw_rgb_sha256(rgb: np.ndarray) -> str:
    canonical = np.array(rgb, dtype="<u2", order="C", copy=True)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def _encode_mask_png(mask: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        stream,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return stream.getvalue()


class _FauxiceEngine:
    def repair(
        self,
        acquisition: RepairAcquisition,
        mode: RepairMode,
        *,
        hybrid_runtime=None,
        progress: Callable[[float], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> RepairResult:
        from negpy.services.repair.fauxice_ir_repair import (
            RepairMode as FauxiceMode,
            RepairStatus,
            _engine_version,
        )

        fauxice_mode = FauxiceMode.HYBRID if mode is RepairMode.HYBRID else FauxiceMode.EXACT
        config = FauxiceRepairConfig(enabled=True, mode=fauxice_mode)
        main = acquisition.main_rgbi
        try:
            result = repair_ir_dust(
                main[..., :3],
                main[..., 3],
                same_frame_id=acquisition.acquisition_id,
                config=config,
                prepass_rgbi=acquisition.prepass_rgbi,
                validity_mask=acquisition.ir_validity,
                hybrid_runtime=hybrid_runtime,
                progress=progress,
                cancel=cancel,
            )
        except FauxiceRepairCancelled as error:
            raise roll_repair.RepairCancelled(str(error)) from error

        if result.status is not RepairStatus.APPLIED or result.repaired_rgb16 is None:
            raise RuntimeError(f"fauxice repair did not produce output: {result.reason}")
        if result.mode_resolved is None:
            raise RuntimeError("fauxice repair omitted its resolved mode")
        resolved_mode = RepairMode(result.mode_resolved.value)
        native_repaired = np.asarray(result.repaired_rgb16)
        if native_repaired.dtype != np.uint16 or native_repaired.shape != (*main.shape[:2], 3) or not native_repaired.flags.c_contiguous:
            raise RuntimeError("fauxice repair returned invalid scanner-native RGB geometry")
        native_output_sha256 = _raw_rgb_sha256(native_repaired)
        if result.native_output_rgb_sha256 is not None and result.native_output_rgb_sha256 != native_output_sha256:
            raise RuntimeError("fauxice native output SHA-256 changed")
        storage_rgb = acquisition.storage_rgb(native_repaired)

        mask_fields: dict[str, object] = {}
        if resolved_mode is RepairMode.HYBRID:
            routed_native_mask = result.hybrid_mask
            routed_native_mask_png = result.hybrid_mask_png
            if (
                routed_native_mask is None
                or routed_native_mask_png is None
                or result.hybrid_mask_sha256 is None
                or result.hybrid_receipt is None
                or result.hybrid_receipt_sha256 is None
            ):
                raise RuntimeError("hybrid repair omitted its verified disclosure evidence")
            routed_native_mask = np.asarray(routed_native_mask)
            if (
                routed_native_mask.dtype != np.bool_
                or routed_native_mask.shape != main.shape[:2]
                or not routed_native_mask.flags.c_contiguous
            ):
                raise RuntimeError("hybrid disclosure mask geometry is invalid")
            if (
                hashlib.sha256(routed_native_mask_png).hexdigest()
                != result.hybrid_mask_sha256
            ):
                raise RuntimeError("hybrid native mask SHA-256 changed")
            if hashlib.sha256(result.hybrid_receipt).hexdigest() != result.hybrid_receipt_sha256:
                raise RuntimeError("hybrid receipt SHA-256 changed")
            native_mask = np.ascontiguousarray(
                routed_native_mask & acquisition.ir_validity
            )
            native_mask_png = _encode_mask_png(native_mask)
            storage_mask = acquisition.storage_mask(native_mask)
            storage_mask_png = _encode_mask_png(storage_mask)
            mask_fields = {
                "native_synthesis_mask_png": native_mask_png,
                "native_synthesis_mask_sha256": hashlib.sha256(
                    native_mask_png
                ).hexdigest(),
                "native_synthesis_mask_shape": tuple(native_mask.shape),
                "routed_native_synthesis_mask_png": routed_native_mask_png,
                "routed_native_synthesis_mask_sha256": result.hybrid_mask_sha256,
                "routed_native_synthesis_mask_shape": tuple(
                    routed_native_mask.shape
                ),
                "storage_synthesis_mask_png": storage_mask_png,
                "storage_synthesis_mask_sha256": hashlib.sha256(storage_mask_png).hexdigest(),
                "storage_synthesis_mask_shape": tuple(storage_mask.shape),
                "synthesis_mask_transform": acquisition.storage_transform,
                "synthesis_fraction": float(np.count_nonzero(native_mask))
                / float(native_mask.size),
                "routing_counts": result.hybrid_routing_counts,
                "hybrid_receipt": result.hybrid_receipt,
                "hybrid_receipt_sha256": result.hybrid_receipt_sha256,
                "hybrid_provenance_class": result.hybrid_provenance_class,
                "hybrid_receipt_output_rgb_sha256": (result.hybrid_receipt_output_rgb_sha256),
            }

        return RepairResult(
            rgb=storage_rgb,
            engine="digital-fauxice",
            engine_version=result.engine_version or _engine_version() or "unknown",
            mode_requested=mode,
            mode_resolved=resolved_mode,
            reason=result.reason,
            acquisition_id=acquisition.acquisition_id,
            slot=acquisition.slot,
            reservation_id=acquisition.reservation_id,
            evidence_sha256=acquisition.evidence_sha256,
            backend_requested=result.backend_requested,
            backend_used=result.backend_used,
            backend_selection_reason=result.backend_selection_reason,
            native_output_rgb_sha256=native_output_sha256,
            storage_output_rgb_sha256=_raw_rgb_sha256(storage_rgb),
            **mask_fields,
        )


if engine_available():
    roll_repair.register_engine(_FauxiceEngine())


__all__ = ["_FauxiceEngine"]
