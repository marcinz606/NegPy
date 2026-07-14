import os
import re
import tomllib
from dataclasses import dataclass
from typing import Dict, Optional

from negpy.kernel.system.config import APP_CONFIG
from negpy.services.export.contact_sheet import CELL_PX, GAP, MARGIN, MAX_TILES_PER_SHEET

DEFAULT_NAME = "Default"

_CELL_PX_RANGE = (100, 4000)
_GAP_RANGE = (0, 200)
_MARGIN_RANGE = (0, 500)
_MAX_TILES_RANGE = (1, 200)


@dataclass(frozen=True)
class ContactSheetLayout:
    cell_px: int = CELL_PX
    gap: int = GAP
    margin: int = MARGIN
    max_tiles: int = MAX_TILES_PER_SHEET


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    return slug or "template"


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _clamp_int(value: object, lo: int, hi: int, default: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return max(lo, min(hi, value))


class ContactSheetTemplates:
    """
    TOML I/O for user contact sheet layout presets.

    Files live in APP_CONFIG.contact_sheet_templates_dir. The built-in layout
    is exposed as "Default" (no file). Disk I/O happens on folder scan and save.
    """

    DEFAULT_NAME = DEFAULT_NAME

    @staticmethod
    def _templates_dir() -> str:
        return APP_CONFIG.contact_sheet_templates_dir

    @staticmethod
    def _scan() -> Dict[str, ContactSheetLayout]:
        """Maps display name -> layout for valid custom .toml files."""
        result: Dict[str, ContactSheetLayout] = {}
        templates_dir = ContactSheetTemplates._templates_dir()
        if not os.path.isdir(templates_dir):
            return result
        for fname in os.listdir(templates_dir):
            if not fname.endswith(".toml"):
                continue
            path = os.path.join(templates_dir, fname)
            parsed = ContactSheetTemplates._parse_file(path)
            if parsed is None:
                continue
            name, layout = parsed
            if name != DEFAULT_NAME:
                result[name] = layout
        return result

    @staticmethod
    def _parse_file(path: str) -> Optional[tuple[str, ContactSheetLayout]]:
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            layout_data = data.get("layout")
            if not isinstance(layout_data, dict):
                return None
            layout = ContactSheetLayout(
                cell_px=_clamp_int(layout_data.get("cell_px"), *_CELL_PX_RANGE, CELL_PX),
                gap=_clamp_int(layout_data.get("gap"), *_GAP_RANGE, GAP),
                margin=_clamp_int(layout_data.get("margin"), *_MARGIN_RANGE, MARGIN),
                max_tiles=_clamp_int(layout_data.get("max_tiles"), *_MAX_TILES_RANGE, MAX_TILES_PER_SHEET),
            )
            raw_name = data.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                name = raw_name.strip()
            else:
                name = os.path.splitext(os.path.basename(path))[0]
            return name, layout
        except Exception:
            return None

    @staticmethod
    def list_templates() -> list[str]:
        """["Default", *sorted custom display names]."""
        return [DEFAULT_NAME, *sorted(ContactSheetTemplates._scan().keys())]

    @staticmethod
    def default_layout() -> ContactSheetLayout:
        """Built-in NegPy contact sheet layout (factory defaults)."""
        return ContactSheetLayout()

    @staticmethod
    def default_layout_from_export(export) -> ContactSheetLayout:
        """User's Default template snapshot, with legacy fallback from active layout fields."""
        factory = ContactSheetTemplates.default_layout()
        stored = ContactSheetLayout(
            cell_px=export.contact_sheet_default_cell_px,
            gap=export.contact_sheet_default_gap,
            margin=export.contact_sheet_default_margin,
            max_tiles=export.contact_sheet_default_max_tiles,
        )
        if stored != factory:
            return stored
        if not export.contact_sheet_template.strip():
            active = ContactSheetLayout(
                cell_px=export.contact_sheet_cell_px,
                gap=export.contact_sheet_gap,
                margin=export.contact_sheet_margin,
                max_tiles=export.contact_sheet_max_tiles,
            )
            if active != factory:
                return active
        return factory

    @staticmethod
    def default_layout_field_updates(layout: ContactSheetLayout) -> dict[str, int]:
        return {
            "contact_sheet_default_cell_px": layout.cell_px,
            "contact_sheet_default_gap": layout.gap,
            "contact_sheet_default_margin": layout.margin,
            "contact_sheet_default_max_tiles": layout.max_tiles,
        }

    @staticmethod
    def active_layout_field_updates(layout: ContactSheetLayout) -> dict[str, int]:
        return {
            "contact_sheet_cell_px": layout.cell_px,
            "contact_sheet_gap": layout.gap,
            "contact_sheet_margin": layout.margin,
            "contact_sheet_max_tiles": layout.max_tiles,
        }

    @staticmethod
    def get_layout(name: str) -> Optional[ContactSheetLayout]:
        """Layout for a named template, or None for Default / missing / invalid."""
        if not name or name == DEFAULT_NAME:
            return None
        return ContactSheetTemplates._scan().get(name)

    @staticmethod
    def path_for_name(name: str) -> str:
        """Filesystem path a template with this display name would use."""
        return os.path.join(ContactSheetTemplates._templates_dir(), f"{_slugify(name)}.toml")

    @staticmethod
    def template_exists(name: str) -> bool:
        if not name or name == DEFAULT_NAME:
            return False
        return name in ContactSheetTemplates._scan()

    @staticmethod
    def delete(name: str) -> bool:
        """Delete a custom template by display name; Default is not deletable.
        The display name comes from the TOML `name` field, not the filename, so
        the file is found by parsing rather than via path_for_name."""
        if not name or name == DEFAULT_NAME:
            return False
        templates_dir = ContactSheetTemplates._templates_dir()
        if not os.path.isdir(templates_dir):
            return False
        for fname in os.listdir(templates_dir):
            if not fname.endswith(".toml"):
                continue
            path = os.path.join(templates_dir, fname)
            parsed = ContactSheetTemplates._parse_file(path)
            if parsed is not None and parsed[0] == name:
                os.remove(path)
                return True
        return False

    @staticmethod
    def save(name: str, layout: ContactSheetLayout) -> str:
        """Write a template TOML and return its path."""
        templates_dir = ContactSheetTemplates._templates_dir()
        os.makedirs(templates_dir, exist_ok=True)
        path = ContactSheetTemplates.path_for_name(name)
        content = (
            f'name = "{_escape_toml_string(name)}"\n\n'
            "[layout]\n"
            f"cell_px = {layout.cell_px}\n"
            f"gap = {layout.gap}\n"
            f"margin = {layout.margin}\n"
            f"max_tiles = {layout.max_tiles}\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
