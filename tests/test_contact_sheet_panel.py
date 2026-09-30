"""The Export panel's Contact Sheet section, its shortcut, and the retired grid settings."""

from conftest import FakeController, FakeRepo

import negpy.desktop.view.shortcut_registry as registry
from negpy.desktop.view.sidebar.export import ExportSidebar
from negpy.domain.models import WorkspaceConfig


def test_contact_sheet_button_opens_the_dialog_through_the_controller():
    controller = FakeController(repo=FakeRepo())
    sidebar = ExportSidebar(controller)
    assert sidebar.contact_sheet_btn.text().strip() == "Contact Sheet…"
    sidebar.contact_sheet_btn.click()
    controller.request_contact_sheet.assert_called_once()


def test_contact_sheet_tooltip_carries_its_bound_key(monkeypatch):
    sidebar = ExportSidebar(FakeController(repo=FakeRepo()))
    rebound = registry.display_key("Ctrl+Shift+K")
    assert rebound not in sidebar.contact_sheet_btn.toolTip()
    monkeypatch.setattr(registry, "key_for", lambda action_id, bindings=None: "Ctrl+Shift+K" if action_id == "contact_sheet" else "")
    sidebar.apply_shortcut_tooltips()
    assert rebound in sidebar.contact_sheet_btn.toolTip()


def test_the_action_is_registered_without_a_default_key():
    entry = registry.REGISTRY["contact_sheet"]
    assert entry.default_key == ""
    assert entry.description == "Contact Sheet…"


def test_retired_grid_settings_drop_silently(caplog):
    old = WorkspaceConfig().to_dict()
    old.update(
        {
            "contact_sheet_cell_px": 800,
            "contact_sheet_template": "Tight 35mm",
            "contact_sheet_default_label_color": "#ffffff",
            "contact_sheet_output_path": "/kept",
        }
    )
    with caplog.at_level("WARNING"):
        config = WorkspaceConfig.from_flat_dict(old)
    assert config.export.contact_sheet_output_path == "/kept"
    assert "Dropping unknown config keys" not in caplog.text
