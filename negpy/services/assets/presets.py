import json
import os
from typing import List, Dict, Any, Optional
from negpy.kernel.system.config import APP_CONFIG


class Presets:
    """
    JSON I/O for user presets.
    """

    @staticmethod
    def save_preset(name: str, settings: Dict[str, Any]) -> None:
        os.makedirs(APP_CONFIG.presets_dir, exist_ok=True)
        filepath = os.path.join(APP_CONFIG.presets_dir, f"{name}.json")
        with open(filepath, "w") as f_out:
            json.dump(settings, f_out, indent=4)

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
        return [f[:-5] for f in os.listdir(APP_CONFIG.presets_dir) if f.endswith(".json")]

    @staticmethod
    def delete_preset(name: str) -> bool:
        filepath = os.path.join(APP_CONFIG.presets_dir, f"{name}.json")
        if not os.path.exists(filepath):
            return False
        os.remove(filepath)
        return True
