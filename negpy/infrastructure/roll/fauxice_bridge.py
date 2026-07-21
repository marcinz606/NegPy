"""Wire digital-fauxice into the roll repair seam.

Import this module and ``roll_repair.available()`` flips True when
``portable-digital-ice`` is installed -- same guarded-import trick
``coolscanpy_roll`` uses. Without the engine, import does nothing.

The 285 dpi RGBI meter pass (``Frame.meter_rgbi``, 425x281x4 uint16)
rides along as ``prepass_rgbi``; fauxice needs both captures of the
same physical frame and will not run without the prepass.
"""

from __future__ import annotations

from negpy.infrastructure.roll import repair as roll_repair
from negpy.infrastructure.roll.repair import RepairMode, RepairResult

try:
    from negpy.services.repair.fauxice_ir_repair import (
        FauxiceRepairConfig,
        engine_available,
        repair_ir_dust,
    )
except ImportError:
    pass
else:
    if engine_available():

        class _FauxiceEngine:
            def repair(self, rgb, ir, mode: RepairMode, *, prepass_rgbi=None) -> RepairResult:
                from negpy.services.repair.fauxice_ir_repair import RepairMode as FauxiceMode, RepairStatus

                fauxice_mode = FauxiceMode.HYBRID if mode is RepairMode.HYBRID else FauxiceMode.EXACT
                config = FauxiceRepairConfig(enabled=True, mode=fauxice_mode)
                result = repair_ir_dust(
                    rgb, ir, same_frame_id="roll-frame", config=config, prepass_rgbi=prepass_rgbi
                )

                if result.status is RepairStatus.APPLIED and result.repaired_rgb16 is not None:
                    from negpy.services.repair.fauxice_ir_repair import _engine_version

                    return RepairResult(
                        rgb=result.repaired_rgb16,
                        engine="digital-fauxice",
                        engine_version=_engine_version() or "unknown",
                        mode=mode,
                    )
                raise RuntimeError(f"fauxice repair did not produce output: {result.reason}")

        roll_repair.register_engine(_FauxiceEngine())
