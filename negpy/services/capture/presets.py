"""Film-stock capture presets — named R/G/B level + per-channel shutter recipes.

Persisted via the session repo (no Qt), so each film stock can be metered once
and recalled. Mirrors how the scanner settings are stored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


#: Framing runs all three channels at once against a single-channel scan exposure, so the
#: live view blows out at full levels (issue #573). 3 stops ≈ the 3× light plus headroom.
FRAMING_DIM_STOPS = 3


def framing_levels(r: int, g: int, b: int, stops: int = FRAMING_DIM_STOPS) -> tuple[int, int, int]:
    """The framing-light mix for a scan-level RGB recipe: the same colour, `stops` dimmer.

    Dimming the light rather than speeding the shutter keeps the scan start free of the
    1-2 s a verified shutter write costs, and works on bodies whose shutter NegPy cannot
    drive. A lit channel never dims to 0, so the mix keeps its hue at single-digit levels.
    """

    def dim(level: int) -> int:
        return max(1, level >> stops) if level > 0 else 0

    return dim(r), dim(g), dim(b)


@dataclass(frozen=True)
class ScanlightPreset:
    """One film stock's capture recipe."""

    r_level: int = 255
    g_level: int = 255
    b_level: int = 255
    w_level: int = 0  # RGB presets don't use white; a white-light preset stores 255
    shutter_r: str = ""
    shutter_g: str = ""
    shutter_b: str = ""
    iso: str = ""  # camera ISO label baked at calibration (e.g. "100"); "" = not captured
    aperture: str = ""  # aperture label (e.g. "f/8"); "" for a manual lens (no electronic aperture)


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
