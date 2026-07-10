"""Film-stock capture presets — named R/G/B level + per-channel shutter recipes.

Persisted via the session repo (no Qt), so each film stock can be metered once
and recalled. Mirrors how the scanner settings are stored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScanlightPreset:
    """One film stock's capture recipe."""

    r_level: int = 255
    g_level: int = 255
    b_level: int = 255
    shutter_r: str = ""
    shutter_g: str = ""
    shutter_b: str = ""


class PresetStore:
    """Named `ScanlightPreset`s persisted under one repo global-setting key."""

    KEY = "scanlight_presets"

    def __init__(self, repo) -> None:
        self._repo = repo

    def _all(self) -> dict:
        data = self._repo.get_global_setting(self.KEY, default={})
        return dict(data) if isinstance(data, dict) else {}

    def names(self) -> list[str]:
        return sorted(self._all().keys())

    def get(self, name: str) -> ScanlightPreset | None:
        raw = self._all().get(name)
        if not isinstance(raw, dict):
            return None
        fields = ScanlightPreset.__dataclass_fields__
        try:
            return ScanlightPreset(**{k: v for k, v in raw.items() if k in fields})
        except Exception:
            return None

    def save(self, name: str, preset: ScanlightPreset) -> None:
        data = self._all()
        data[name] = asdict(preset)
        self._repo.save_global_setting(self.KEY, data)

    def delete(self, name: str) -> None:
        data = self._all()
        if name in data:
            del data[name]
            self._repo.save_global_setting(self.KEY, data)
