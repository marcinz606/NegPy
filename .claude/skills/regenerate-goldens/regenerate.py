"""Recompute NegPy's rendered characterization goldens after an INTENTIONAL look change.

Prints ready-to-paste values for the two snapshot tests that pin full-engine /
print-curve output. It imports the tests' own image/points/curve helpers, so those
stay in sync — only the per-config list below mirrors the test and must be kept aligned
if configs are added/removed there.

Run from the repo root:  uv run python .claude/skills/regenerate-goldens/regenerate.py
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for `tests` + `negpy`

from negpy.features.exposure.models import ExposureConfig
from negpy.services.rendering.engine import DarkroomEngine
from tests.test_characteristic_curve import _curve
from tests.test_scene_linear_relocation import _base_settings, _synthetic_image, _POINTS


def relocation_goldens() -> None:
    """test_scene_linear_relocation.py -> _GOLDEN (full engine, default + exposure variants)."""
    configs = {
        "default": _base_settings(),
        "expo_dark": replace(_base_settings(), exposure=ExposureConfig(density=-1.0, grade=2.0)),
        "expo_cmy": replace(_base_settings(), exposure=ExposureConfig(wb_cyan=0.3, wb_magenta=-0.2, wb_yellow=0.5)),
    }
    img = _synthetic_image()
    eng = DarkroomEngine()
    print("# --- test_scene_linear_relocation.py : paste over _GOLDEN ---")
    print("_GOLDEN = {")
    for name, cfg in configs.items():
        out = eng.process(img, cfg, f"regen_{name}")
        print(f'    "{name}": [')
        for y, x in _POINTS:
            p = out[y, x]
            print(f"        ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}),")
        print("    ],")
    print("}")


def characteristic_curve_golden() -> None:
    """test_characteristic_curve.py::test_default_curve_shape -> golden list."""
    _, out = _curve()
    idx = [0, 64, 128, 192, 256]
    vals = [round(float(out[i]), 3) for i in idx]
    print("\n# --- test_characteristic_curve.py::test_default_curve_shape : paste over `golden` ---")
    print(f"        golden = {vals}")


if __name__ == "__main__":
    relocation_goldens()
    characteristic_curve_golden()
