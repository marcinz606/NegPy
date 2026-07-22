"""Roll-scanning orchestration -- drives a coolscanpy Device + Roll extension
through preview, spacing correction, approval and batch fine-scanning.

No Qt dependencies (mirrors `negpy.services.capture` and
`negpy.services.scanning`).
"""

from negpy.services.roll.exact_color import (
    BuilderReceipt,
    ExactColorIntegrityError,
    ExactColorResult,
    ExactColorUnavailable,
    FIXED_COMPOSITION_SHA256,
    NATIVE_BUILDER_SCOPE,
    NativeValidatedBuilderReceipt,
    PositiveColorMode,
    STAGE3_REPLAY_SCOPE,
    Stage1BuilderResult,
    ValidatedBuilderReceipt,
    VerifiedBuilderApplicationReceipt,
    VerifiedCMSReceipt,
    VerifiedPortableCMSEvaluator,
    VerifiedStage1Builder,
    load_stage3_replay_builder_receipt,
)
from negpy.services.roll.native_builder import NativeBuilderEvidence, build_native_builder_receipt
from negpy.services.roll.portable_builder import PortableStage1Builder
from negpy.services.roll.portable_cms import PortableCMSOnEvaluator
from negpy.services.roll.service import RollFrameOutput, RollScanningError, RollScanningService, available

__all__ = [
    "ExactColorIntegrityError",
    "ExactColorResult",
    "ExactColorUnavailable",
    "FIXED_COMPOSITION_SHA256",
    "BuilderReceipt",
    "NATIVE_BUILDER_SCOPE",
    "NativeBuilderEvidence",
    "NativeValidatedBuilderReceipt",
    "PositiveColorMode",
    "PortableCMSOnEvaluator",
    "PortableStage1Builder",
    "RollFrameOutput",
    "RollScanningError",
    "RollScanningService",
    "STAGE3_REPLAY_SCOPE",
    "Stage1BuilderResult",
    "ValidatedBuilderReceipt",
    "VerifiedBuilderApplicationReceipt",
    "VerifiedCMSReceipt",
    "VerifiedPortableCMSEvaluator",
    "VerifiedStage1Builder",
    "available",
    "build_native_builder_receipt",
    "load_stage3_replay_builder_receipt",
]
