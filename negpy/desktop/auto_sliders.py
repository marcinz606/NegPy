"""The Auto Density / Auto Grade slider conversion for the open frame: its render path's
inverse, fed by the meters of the frame's last plain render.

The meters are kept per file because last_metrics outlives a frame switch and a
navigate-back memo paint shows a frame before its own render lands.
"""

from typing import Any, Dict, Mapping

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.auto_sliders import print_shown_values, print_stored_value
from negpy.features.process.path import RenderPath, render_path
from negpy.features.transparency.logic import transfer_shown_values, transfer_stored_value

METER_KEYS = (
    "metered_anchor",
    "textural_range",
    "norm_density_range",
    "shadow_point",
    "highlight_point",
    "final_bounds",
    "log_bounds",
    "shadow_log_refs",
    "neutral_axis_refs",
)


def record_meters(meters_by_file: Dict[str, Dict[str, Any]], current_hash: str, metrics: Mapping[str, Any]) -> None:
    """Keep the meters of a plain render of the open frame. Only those carry a memo key:
    proxy, override, splash and crop-tool frames meter something else."""
    if metrics.get("memo_key") and not metrics.get("diptych") and metrics.get("source_hash") == current_hash:
        meters_by_file[current_hash] = {k: metrics.get(k) for k in METER_KEYS}


def shown_values(config: WorkspaceConfig, meters: Mapping[str, Any]) -> Dict[str, float]:
    if not meters:
        return {}
    if render_path(config.process) is RenderPath.PRINT:
        return print_shown_values(config.exposure, config.process.process_mode, meters)
    return transfer_shown_values(config.exposure, meters)


def stored_value(config: WorkspaceConfig, meters: Mapping[str, Any], field: str, shown: float) -> float:
    if not meters:
        return shown
    if render_path(config.process) is RenderPath.PRINT:
        return print_stored_value(config.exposure, config.process.process_mode, meters, field, shown)
    return transfer_stored_value(config.exposure, meters, field, shown)
