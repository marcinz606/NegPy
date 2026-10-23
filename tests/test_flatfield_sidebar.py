from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from negpy.desktop.view.sidebar.flatfield import FlatFieldSidebar
from negpy.domain.models import WorkspaceConfig


def _sidebar(profile_id: str, apply: bool) -> FlatFieldSidebar:
    base = WorkspaceConfig()
    config = replace(base, flatfield=replace(base.flatfield, profile_id=profile_id, apply=apply))
    controller = MagicMock()
    controller.state = SimpleNamespace(config=config)
    with patch("negpy.services.assets.flatfield.FlatFieldProfiles.list_profiles", return_value=[("ff-35", "35mm"), ("ff-120", "120")]):
        sb = FlatFieldSidebar(controller)
        sb.sync_ui()
    return sb


def test_the_picker_shows_the_frames_own_profile(qapp):
    sb = _sidebar("ff-120", True)
    assert sb.profile_combo.currentData() == "ff-120"
    assert sb.enable_btn.isEnabled() and sb.enable_btn.isChecked()


def test_a_deleted_profile_reads_as_none(qapp):
    sb = _sidebar("gone", True)
    assert sb.profile_combo.currentData() == ""
    assert not sb.enable_btn.isEnabled()
    assert not sb.enable_btn.isChecked()
