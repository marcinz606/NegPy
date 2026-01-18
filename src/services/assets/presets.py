import json
import os
from typing import List, Dict, Any, Optional
from src.kernel.system.config import APP_CONFIG
from src.domain.models import WorkspaceConfig


class Presets:
    """
    JSON I/O for user presets.
    """

    @staticmethod
    def save_preset(name: str, settings: WorkspaceConfig) -> None:
        """
        Saves partial WorkspaceConfig to JSON.
        """
        os.makedirs(APP_CONFIG.presets_dir, exist_ok=True)

        # Exclude keys that are not relevant for presets
        exclude_keys = {
            "rotation",
            "fine_rotation",
            "autocrop",
            "autocrop_offset",
            "manual_dust_spots",
            "local_adjustments",
            "active_adjustment_idx",
            "crosstalk_matrix",
            "export_path",
            "apply_iccicc_profile_path",
            "autocrop_assist_point",
            "autocrop_assist_luma",
            "manual_dust_size",
        }

        settings_dict = settings.to_dict()
        default_dict = WorkspaceConfig().to_dict()

        filtered = {
            k: v
            for k, v in settings_dict.items()
            if k in default_dict and k not in exclude_keys
        }

        filepath = os.path.join(APP_CONFIG.presets_dir, f"{name}.json")
        with open(filepath, "w") as f_out:
            json.dump(filtered, f_out, indent=4)

    @staticmethod
    def load_preset(name: str) -> Optional[Dict[str, Any]]:
        filepath = os.path.join(APP_CONFIG.presets_dir, f"{name}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r") as f_in:
            res = json.load(f_in)
            if isinstance(res, dict):
                return res
            return None

    @staticmethod
    def list_presets() -> List[str]:
        if not os.path.exists(APP_CONFIG.presets_dir):
            return []
        return [
            f[:-5] for f in os.listdir(APP_CONFIG.presets_dir) if f.endswith(".json")
        ]
