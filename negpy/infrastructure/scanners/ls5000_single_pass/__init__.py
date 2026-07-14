"""Proven scanner-free core for Nikon LS-5000 single-pass RGB4x + IR."""

from .bundle import CAPTURE_BUNDLE_SHA256, CAPTURE_WORKER_SHA256
from .capture_process import (
    CaptureAttemptResult,
    CaptureMode,
    CaptureOutcome,
    CaptureProcessAdapter,
    CaptureRequest,
)
from .meter import (
    DEFAULT_EXPOSURES,
    DecodedMeterPass,
    MeterObservation,
    MeterProposal,
    MeterResult,
    SafetyRefusal,
    decode_meter_pass,
    evaluate_meter_sequence,
    observe_meter_pass,
    propose_next_exposures,
    verify_final_convergence,
)
from .packed import decode_full_records, decode_records, infer_record_geometry
from .plan import (
    CANONICAL_PLAN_SHA256,
    CanonicalPlanError,
    canonical_plan_bytes,
    load_canonical_plan,
    verify_canonical_plan,
)
from .roll_index import (
    IndexDecodeError,
    IndexGeometry,
    RollDetection,
    TransportMapping,
    decode_full_index_bytes,
    derive_transport_mapping,
    detect_roll_frames,
    parse_live_transport_records_bytes,
)

__all__ = [
    "CANONICAL_PLAN_SHA256",
    "CAPTURE_BUNDLE_SHA256",
    "CAPTURE_WORKER_SHA256",
    "DEFAULT_EXPOSURES",
    "CanonicalPlanError",
    "CaptureAttemptResult",
    "CaptureMode",
    "CaptureOutcome",
    "CaptureProcessAdapter",
    "CaptureRequest",
    "DecodedMeterPass",
    "IndexDecodeError",
    "IndexGeometry",
    "MeterObservation",
    "MeterProposal",
    "MeterResult",
    "RollDetection",
    "SafetyRefusal",
    "TransportMapping",
    "canonical_plan_bytes",
    "decode_full_index_bytes",
    "decode_full_records",
    "decode_meter_pass",
    "decode_records",
    "derive_transport_mapping",
    "detect_roll_frames",
    "evaluate_meter_sequence",
    "infer_record_geometry",
    "load_canonical_plan",
    "observe_meter_pass",
    "parse_live_transport_records_bytes",
    "propose_next_exposures",
    "verify_canonical_plan",
    "verify_final_convergence",
]
