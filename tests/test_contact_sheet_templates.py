import os

from negpy.domain.models import ExportConfig
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.export.contact_sheet_templates import (
    ContactSheetLayout,
    ContactSheetTemplates,
)


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_list_and_get_custom(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "contact_sheet_templates_dir", str(tmp_path))

    _write(
        os.path.join(tmp_path, "tight.toml"),
        'name = "Tight 35mm"\n\n[layout]\ncell_px = 400\ngap = 8\nmargin = 16\nmax_tiles = 48\n',
    )

    assert ContactSheetTemplates.list_templates() == ["Default", "Tight 35mm"]
    layout = ContactSheetTemplates.get_layout("Tight 35mm")
    assert layout == ContactSheetLayout(cell_px=400, gap=8, margin=16, max_tiles=48)


def test_name_falls_back_to_stem(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "contact_sheet_templates_dir", str(tmp_path))
    _write(
        os.path.join(tmp_path, "my_layout.toml"),
        "[layout]\ncell_px = 500\ngap = 10\nmargin = 20\nmax_tiles = 30\n",
    )
    assert "my_layout" in ContactSheetTemplates.list_templates()


def test_default_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "contact_sheet_templates_dir", str(tmp_path))
    assert ContactSheetTemplates.get_layout("Default") is None
    assert ContactSheetTemplates.get_layout("") is None
    assert ContactSheetTemplates.get_layout("nonexistent") is None


def test_malformed_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "contact_sheet_templates_dir", str(tmp_path))
    _write(os.path.join(tmp_path, "bad_toml.toml"), "[layout]\ncell_px = not_int\n")
    _write(os.path.join(tmp_path, "no_layout.toml"), 'name = "x"\n')
    assert ContactSheetTemplates.list_templates() == ["Default"]


def test_partial_layout_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "contact_sheet_templates_dir", str(tmp_path))
    _write(
        os.path.join(tmp_path, "partial.toml"),
        'name = "Partial"\n\n[layout]\ncell_px = 700\n',
    )
    layout = ContactSheetTemplates.get_layout("Partial")
    assert layout.cell_px == 700
    assert layout.gap == 16
    assert layout.margin == 32
    assert layout.max_tiles == 38


def test_save_writes_readable_template(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "contact_sheet_templates_dir", str(tmp_path))
    layout = ContactSheetLayout(cell_px=450, gap=12, margin=24, max_tiles=40)
    path = ContactSheetTemplates.save("My Roll", layout)
    assert path.endswith("my_roll.toml")
    assert os.path.isfile(path)
    assert ContactSheetTemplates.list_templates() == ["Default", "My Roll"]
    assert ContactSheetTemplates.get_layout("My Roll") == layout


def test_default_layout_matches_service_defaults():
    layout = ContactSheetTemplates.default_layout()
    assert layout == ContactSheetLayout()


def test_default_layout_from_export_uses_stored_default():
    export = ExportConfig(
        contact_sheet_default_cell_px=450,
        contact_sheet_default_gap=10,
        contact_sheet_default_margin=20,
        contact_sheet_default_max_tiles=24,
    )
    assert ContactSheetTemplates.default_layout_from_export(export) == ContactSheetLayout(cell_px=450, gap=10, margin=20, max_tiles=24)


def test_default_layout_from_export_legacy_active_fields():
    export = ExportConfig(
        contact_sheet_cell_px=650,
        contact_sheet_gap=20,
        contact_sheet_margin=40,
        contact_sheet_max_tiles=20,
        contact_sheet_template="",
    )
    assert ContactSheetTemplates.default_layout_from_export(export) == ContactSheetLayout(cell_px=650, gap=20, margin=40, max_tiles=20)


def test_default_layout_from_export_named_template_ignores_active_fields():
    export = ExportConfig(
        contact_sheet_cell_px=650,
        contact_sheet_default_cell_px=500,
        contact_sheet_template="Other",
    )
    assert ContactSheetTemplates.default_layout_from_export(export).cell_px == 500


def test_template_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "contact_sheet_templates_dir", str(tmp_path))
    ContactSheetTemplates.save("Existing", ContactSheetLayout())
    assert ContactSheetTemplates.template_exists("Existing")
    assert not ContactSheetTemplates.template_exists("Default")
    assert not ContactSheetTemplates.template_exists("Missing")


def test_delete_by_display_name_not_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "contact_sheet_templates_dir", str(tmp_path))
    # Display name differs from the file stem — delete must resolve by parsing.
    _write(
        os.path.join(tmp_path, "odd_stem.toml"),
        'name = "Pretty Name"\n\n[layout]\ncell_px = 400\ngap = 8\nmargin = 16\nmax_tiles = 48\n',
    )
    assert ContactSheetTemplates.delete("Pretty Name") is True
    assert not os.path.exists(os.path.join(tmp_path, "odd_stem.toml"))
    assert ContactSheetTemplates.list_templates() == ["Default"]


def test_delete_guards_default_and_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "contact_sheet_templates_dir", str(tmp_path))
    ContactSheetTemplates.save("Keep", ContactSheetLayout())
    assert ContactSheetTemplates.delete("Default") is False
    assert ContactSheetTemplates.delete("Missing") is False
    assert ContactSheetTemplates.template_exists("Keep")
