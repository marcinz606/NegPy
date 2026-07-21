"""Optional post-import repair steps. No Qt dependencies."""

from negpy.services.repair.fauxice_hybrid_runner import HybridRuntimeConfig
from negpy.services.repair.fauxice_ir_repair import (
    FauxiceRepairConfig,
    FauxiceRepairResult,
    RepairMode,
    RepairStatus,
    engine_available,
    hybrid_available,
    repair_frame_files,
    repair_ir_dust,
)

__all__ = [
    "FauxiceRepairConfig",
    "FauxiceRepairResult",
    "HybridRuntimeConfig",
    "RepairMode",
    "RepairStatus",
    "engine_available",
    "hybrid_available",
    "repair_frame_files",
    "repair_ir_dust",
]
