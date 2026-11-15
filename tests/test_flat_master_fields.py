"""reaches_flat_master names exactly the config the Flat master render reads."""

from dataclasses import fields, replace
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
import pytest

import fade_crossover_scene as scene
from negpy.desktop.view.sidebar import controls_panel
from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import _FLAT_FIELDS, _FLAT_SECTIONS, WorkspaceConfig, flat_master_config, reaches_flat_master
from negpy.features.exposure.models import BwFilter
from negpy.features.process.models import ProcessMode
from negpy.services.rendering.engine import DarkroomEngine


def _negative() -> np.ndarray:
    """Gradient plus color noise, so every channel has a range to stretch."""
    rng = np.random.default_rng(7)
    ramp = np.linspace(0.08, 0.85, 128, dtype=np.float32)[None, :, None]
    img = ramp * np.array([1.0, 0.8, 0.6], np.float32) + rng.uniform(-0.05, 0.05, (96, 128, 3)).astype(np.float32)
    return np.clip(img, 0.01, 1.0).astype(np.float32)


@lru_cache(maxsize=None)
def _image(mode: ProcessMode) -> np.ndarray:
    if mode == ProcessMode.E6:
        return scene.scan_image(scene.linear_fade(scene.CROSSOVER, (0.8, 0.9, 0.6)), scene.build_scene(192, 144))
    return _negative()


# A field reaches the flat master if it moves it in at least one of these.
_MODES = (
    {"process_mode": ProcessMode.C41},
    {"process_mode": ProcessMode.BW, "source_color": True},
    {"process_mode": ProcessMode.E6},
)

# Values that move a field off its default where the generic rule (_moved) does not.
_VALUES: Dict[str, Any] = {
    "paper_profile": "frontier",
    "bw_filter": BwFilter.RED_25,
    "grade": 140.0,
    "crosstalk_strength": 1.0,
}

# Fields this render cannot test, each with the reason.
_UNTESTED: Dict[str, str] = {
    "render_intent": "flat_master_config sets it",
    "process_mode": "the axis of the test",
    "source_color": "the axis of the test",
    "linear_raw": "decode",
    "source_encoding": "decode",
    "positive_source": "routes the base as process_mode does",
    "demosaic_preview": "decode",
    "demosaic_export": "decode",
    "highlight_reconstruction": "decode",
    "sensor_matrix": "source bake in ImageProcessor",
    "sensor_profile": "source bake in ImageProcessor",
    "sensor_unmix": "source bake in ImageProcessor",
    "analysis_rect": "drawn on the canvas",
    "manual_dmin": "picked on the canvas",
    "manual_dmax": "picked on the canvas",
    "manual_dmin_at": "picked on the canvas",
    "manual_dmax_at": "picked on the canvas",
    "crosstalk_matrix": "written by the matrix editor",
    "crosstalk_profile": "a label: the render reads crosstalk_matrix",
    "lock_bounds": "keeps the stored bounds from a new measurement; moves no pixel by itself",
    "fade_auto_red": "a Fade Estimate setting; moves no pixel in either view",
    "fade_red_trim": "a Fade Estimate setting; moves no pixel in either view",
    "fade_neutrals_mix": "a Fade Estimate setting; moves no pixel in either view",
    "fade_crossover_table": "measured by the render",
    "fade_white_space": "a label of the measured white",
    "fade_crossover_space": "a label of the measured table",
    "locked_neutral_axis": "measured by Roll Analysis; only Cast Removal reads it",
}

# _process_flat collapses to gray only on a B&W negative, so a B&W slide's flat master stays color.
_KNOWN_GAPS = {("process", "slide_bw")}


@lru_cache(maxsize=None)
def _locked_bounds() -> tuple[tuple, tuple]:
    """A roll baseline off this frame's own bounds, by a different amount per channel."""
    cfg = WorkspaceConfig()
    img = _negative()
    ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=cfg.process.process_mode)
    DarkroomEngine().process(img, flat_master_config(cfg), "bounds", ctx)
    bounds = ctx.metrics["log_bounds"]
    shift = (0.15, 0.0, -0.15)
    return tuple(f + s for f, s in zip(bounds.floors, shift)), tuple(c - s for c, s in zip(bounds.ceils, shift))


def _parents(name: str) -> Dict[str, Any]:
    """The fields that must be on before *name* can act."""
    if name in ("use_luma_average", "use_color_average"):
        floors, ceils = _locked_bounds()
        return {"locked_floors": floors, "locked_ceils": ceils}
    if name == "crosstalk_process":
        return {"crosstalk_strength": 1.0}
    if name == "fade_crossover":
        return {"fade_ratio_r": 0.8}
    if name in ("fade_white_r", "fade_white_g", "fade_white_b"):
        return {"fade_white_anchor": True}
    if name == "fade_white_anchor":
        return {"fade_white_r": 0.3, "fade_white_g": 0.2, "fade_white_b": 0.1}
    return {}


def _card_process_fields() -> tuple[str, ...]:
    owned = (
        controls_panel._FILM_FIELDS
        + controls_panel._METERING_FIELDS
        + controls_panel._BASELINE_FIELDS
        + controls_panel._SENSOR_FIELDS
        + controls_panel._FADE_FIELDS
        + controls_panel._DEMOSAIC_FIELDS
    )
    return tuple(dict.fromkeys(owned))


def _moved(name: str, default: Any) -> Optional[Any]:
    if name in _VALUES:
        return _VALUES[name]
    if isinstance(default, bool):
        return not default
    if isinstance(default, Enum):
        return next(m for m in type(default) if m != default)
    if isinstance(default, (int, float)):
        return default + 0.3 if default == 0 else default * 1.3
    return None


def _with(cfg: WorkspaceConfig, section: str, **values: Any) -> WorkspaceConfig:
    return replace(cfg, **{section: replace(getattr(cfg, section), **values)})


def _flat(cfg: WorkspaceConfig) -> np.ndarray:
    flat = flat_master_config(cfg)
    img = _image(flat.process.process_mode)
    ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=flat.process.process_mode)
    return np.asarray(DarkroomEngine().process(img, flat, "fields", ctx), dtype=np.float32)


def _moves_the_flat_master(section: str, name: str) -> bool:
    for mode in _MODES:
        base = _with(_with(WorkspaceConfig(), "process", **mode), section, **_parents(name))
        value = _moved(name, getattr(getattr(base, section), name))
        assert value is not None, f"{section}.{name}: add a value to _VALUES or a reason to _UNTESTED"
        if not np.allclose(_flat(base), _flat(_with(base, section, **{name: value})), atol=1e-5):
            return True
    return False


_CASES = [
    pytest.param(section, name, id=f"{section}.{name}", marks=pytest.mark.xfail(strict=True) if (section, name) in _KNOWN_GAPS else ())
    for section, name in [("exposure", f.name) for f in fields(WorkspaceConfig().exposure)]
    + [("process", n) for n in _card_process_fields()]
    if name not in _UNTESTED
]


@pytest.mark.parametrize(("section", "name"), _CASES)
def test_reaches_flat_master_matches_the_render(section, name):
    assert _moves_the_flat_master(section, name) == reaches_flat_master(section, name)


def test_every_untested_field_still_exists():
    names = {f.name for f in fields(WorkspaceConfig().exposure)} | {f.name for f in fields(WorkspaceConfig().process)}
    assert set(_UNTESTED) <= names


def test_every_config_section_is_classified():
    """A new section must be named here or in models.py, so no card is gated by accident."""
    print_only = {"lab", "local", "retouch", "altproc", "toning", "finish"}
    not_pixels = {"metadata", "export"}
    sections = {f.name for f in fields(WorkspaceConfig())}
    assert sections == _FLAT_SECTIONS | set(_FLAT_FIELDS) | print_only | not_pixels


def test_a_section_with_no_fields_named_reaches_when_any_part_does():
    assert reaches_flat_master("geometry")
    assert reaches_flat_master("exposure")
    assert not reaches_flat_master("toning")


def test_flat_master_config_is_unchanged():
    """This change adds a lookup and moves no pixel: the flat config of an edited frame is as before."""
    edited = _with(
        _with(WorkspaceConfig(), "exposure", density=1.4, grade=90.0, toe=0.3, wb_cyan=0.1, preflash=0.2, paper_profile="frontier"),
        "process",
        hue_trim=5.0,
    )
    flat = flat_master_config(edited).exposure
    assert (flat.density, flat.grade, flat.toe, flat.wb_cyan, flat.preflash, flat.paper_profile) == (1.4, 90.0, 0.0, 0.1, 0.0, "frontier")
    assert flat.render_intent.value == "flat"
    assert not (flat.auto_exposure or flat.auto_normalize_contrast or flat.paper_dmin)
    assert flat.paper_black and flat.cast_removal_strength == 0.0
