"""Auto Density, Auto Grade and Auto Masking slider conversion for the open frame: its render path's inverse,
fed by the meters of the frame's last plain render. Meters are kept per file: last_metrics
outlives a frame switch, and a memo paint shows a frame before its own render lands.
"""

from typing import Any, Dict, Mapping, Optional

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.auto_sliders import print_shown_values, print_stored_value
from negpy.features.lab.logic import NEUTRAL_MASKING, effective_masking
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
    "neutral_axis_refs",
    "auto_masking",
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


def auto_masking_share(config: WorkspaceConfig, meters: Mapping[str, Any]) -> Optional[float]:
    """The masking Auto Masking measured on the last render, or None when it is off or unmetered."""
    return meters.get("auto_masking") if config.lab.sharpen_auto_masking else None


def shown_masking(config: WorkspaceConfig, meters: Mapping[str, Any]) -> float:
    return effective_masking(config.lab, auto_masking_share(config, meters))


def stored_masking(config: WorkspaceConfig, meters: Mapping[str, Any], shown: float) -> float:
    auto = auto_masking_share(config, meters)
    return shown if auto is None else shown - auto + NEUTRAL_MASKING


def stored_value(config: WorkspaceConfig, meters: Mapping[str, Any], field: str, shown: float) -> float:
    if not meters:
        return shown
    if render_path(config.process) is RenderPath.PRINT:
        return print_stored_value(config.exposure, config.process.process_mode, meters, field, shown)
    return transfer_stored_value(config.exposure, meters, field, shown)
