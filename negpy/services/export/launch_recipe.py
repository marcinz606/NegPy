"""The export recipe a command-line launch sets for the files it opens.

A recipe lives for one session and applies at export time only: the frame's own export
settings stay as saved, and every value the recipe leaves out comes from them."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Optional

from negpy.domain.models import (
    AspectRatio,
    ColorSpace,
    ExportConfig,
    ExportFormat,
    ExportPreset,
    ExportPresetOutputMode,
    ExportResolutionMode,
    RenderIntent,
)

INTENTS = ("print", "flat", "linear")

# The Export tab groups a recipe can set. A set group hides as a whole.
GROUP_INTENT = "intent"
GROUP_FORMAT = "format"
GROUP_SIZE = "size"
GROUP_COLOR = "color"
GROUP_DESTINATION = "destination"


def _slug(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


_FORMATS = {_slug(f.value): f.value for f in ExportFormat} | {"jpg": ExportFormat.JPEG.value, "tif": ExportFormat.TIFF.value}
_COLOR_SPACES = {_slug(c.value): c.value for c in ColorSpace} | {"grayscale": ColorSpace.GREYSCALE.value}
_RATIOS = {r.value.lower(): r.value for r in AspectRatio if r not in (AspectRatio.FREE,)}


def parse_format(value: str) -> str:
    try:
        return _FORMATS[_slug(value)]
    except KeyError:
        raise ValueError(f"unknown format {value!r}; use jpeg, tiff, png, jxl or webp") from None


def parse_color_space(value: str) -> str:
    try:
        return _COLOR_SPACES[_slug(value)]
    except KeyError:
        raise ValueError(f"unknown color space {value!r}; use one of {', '.join(sorted(_COLOR_SPACES))}") from None


def parse_ratio(value: str) -> str:
    try:
        return _RATIOS[value.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown paper ratio {value!r}; use original or a ratio such as 3:2") from None


@dataclass(frozen=True)
class LaunchRecipe:
    preset: Optional[str] = None
    intent: Optional[str] = None
    export_fmt: Optional[str] = None
    export_bit_depth: Optional[int] = None
    quality: Optional[int] = None
    color_space: Optional[str] = None
    profile: Optional[str] = None
    original_size: bool = False
    long_edge: Optional[int] = None
    print_size: Optional[float] = None
    dpi: Optional[int] = None
    paper_aspect_ratio: Optional[str] = None
    filename_pattern: Optional[str] = None
    output_dir: Optional[str] = None
    overwrite: bool = True
    quit_after_export: bool = False
    auto_crop: bool = False  # --render only: arm Auto Crop on a frame with no crop of its own
    # Set by the window when this recipe came with the launch that started NegPy.
    started_app: bool = False


def recipe_groups(recipe: LaunchRecipe, preset: Optional[ExportPreset]) -> frozenset[str]:
    """The Export tab groups the recipe sets, so the tab hides them."""
    groups = set()
    if preset is not None:
        groups |= {GROUP_INTENT, GROUP_FORMAT, GROUP_SIZE, GROUP_COLOR, GROUP_DESTINATION}
    if recipe.intent:
        groups.add(GROUP_INTENT)
    if recipe.export_fmt or recipe.export_bit_depth or recipe.quality:
        groups.add(GROUP_FORMAT)
    if recipe.original_size or recipe.long_edge or recipe.print_size or recipe.dpi or recipe.paper_aspect_ratio:
        groups.add(GROUP_SIZE)
    if recipe.color_space or recipe.profile:
        groups.add(GROUP_COLOR)
    if recipe.output_dir or recipe.filename_pattern:
        groups.add(GROUP_DESTINATION)
    return frozenset(groups)


def recipe_intent(recipe: LaunchRecipe, preset: Optional[ExportPreset]) -> Optional[str]:
    """``print``, ``flat`` or ``linear`` when the recipe or its preset sets the intent, else None."""
    if recipe.intent:
        return recipe.intent
    if preset is not None:
        return "flat" if preset.render_intent == RenderIntent.FLAT else "print"
    return None


_PRESET_SKIP = {"id", "name", "enabled", "render_intent", "output_path", "icc_input_path"}


def apply_recipe(conf: ExportConfig, recipe: LaunchRecipe, preset: Optional[ExportPreset]) -> ExportConfig:
    """``conf`` with the recipe's values on top: the preset first, then each flag."""
    changes: dict = {}
    if preset is not None:
        names = {f.name for f in fields(ExportConfig)}
        changes.update({f.name: getattr(preset, f.name) for f in fields(ExportPreset) if f.name in names and f.name not in _PRESET_SKIP})
        changes["export_path"] = preset.output_path
    if recipe.export_fmt:
        changes["export_fmt"] = ExportFormat(recipe.export_fmt)
    if recipe.export_bit_depth:
        changes["export_bit_depth"] = recipe.export_bit_depth
    if recipe.quality:
        changes["jpeg_quality"] = recipe.quality
        changes["webp_quality"] = recipe.quality
    if recipe.original_size:
        changes["export_resolution_mode"] = ExportResolutionMode.ORIGINAL.value
    if recipe.long_edge:
        changes["export_resolution_mode"] = ExportResolutionMode.TARGET_PX.value
        changes["export_target_long_edge_px"] = recipe.long_edge
    if recipe.print_size or recipe.dpi:
        changes["export_resolution_mode"] = ExportResolutionMode.PRINT.value
        if recipe.print_size:
            changes["export_print_size"] = recipe.print_size
        if recipe.dpi:
            changes["export_dpi"] = recipe.dpi
    if recipe.paper_aspect_ratio:
        changes["paper_aspect_ratio"] = recipe.paper_aspect_ratio
    if recipe.color_space:
        changes["export_color_space"] = recipe.color_space
        changes["icc_output_path"] = None
    if recipe.profile:
        changes["icc_output_path"] = recipe.profile
    if recipe.output_dir:
        changes["output_mode"] = ExportPresetOutputMode.ABSOLUTE
        changes["export_path"] = recipe.output_dir
    if recipe.filename_pattern:
        changes["filename_pattern"] = recipe.filename_pattern
    changes["overwrite"] = recipe.overwrite
    return replace(conf, **changes)


def recipe_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))
