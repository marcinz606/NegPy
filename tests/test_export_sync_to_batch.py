"""Export tab's Protect Original Metadata and Sync To Batch checkboxes: export-time
behaviors over the Metadata tab's per-frame fields, not metadata content themselves,
so they live beside the Export button rather than on that tab."""

from __future__ import annotations

from dataclasses import replace

from conftest import FakeController
from negpy.desktop.view.sidebar.export import ExportSidebar


def _sidebar() -> ExportSidebar:
    controller = FakeController()
    controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
    return ExportSidebar(controller)


def test_off_by_default() -> None:
    sidebar = _sidebar()
    assert sidebar.sync_check.isChecked() is False
    assert sidebar.protect_check.isChecked() is False


def test_toggle_persists_to_the_metadata_config() -> None:
    sidebar = _sidebar()
    sidebar.sync_check.setChecked(True)
    assert sidebar.state.config.metadata.sync_to_batch is True


def test_protect_toggle_persists_and_disables_sync_to_batch() -> None:
    sidebar = _sidebar()

    sidebar.protect_check.setChecked(True)
    assert sidebar.state.config.metadata.protect_original_metadata is True
    assert sidebar.sync_check.isEnabled() is False

    sidebar.protect_check.setChecked(False)
    assert sidebar.state.config.metadata.protect_original_metadata is False
    assert sidebar.sync_check.isEnabled() is True


def test_sync_ui_reflects_protect_state_from_an_unrelated_config_change() -> None:
    sidebar = _sidebar()
    sidebar.state.config = replace(sidebar.state.config, metadata=replace(sidebar.state.config.metadata, protect_original_metadata=True))

    sidebar.sync_ui()

    assert sidebar.protect_check.isChecked() is True
    assert sidebar.sync_check.isEnabled() is False
