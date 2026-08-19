"""Sync-to-batch hands the active frame's whole MetadataConfig to every file.

Whole object, not a field list: a metadata field added to the panel is synced without
touching this path. The test pins that, so a future narrowing shows up here.
"""

from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.models import MetadataConfig

_ACTIVE = MetadataConfig(
    sync_to_batch=True,
    capture_date="1998-07-14 16:30",
    gps_latitude=35.6762,
    gps_longitude=139.6503,
    location_city="Tokyo",
    location_state="Tokyo",
    location_country="Japan",
    film="Portra 400",
)
_PER_FILE = MetadataConfig(film="Velvia 50")


def _controller(active: MetadataConfig) -> MagicMock:
    controller = MagicMock()
    controller.state.config = replace(WorkspaceConfig(), metadata=active)
    controller.state.current_file_hash = "not-this-one"
    controller._batch_params_for.return_value = replace(WorkspaceConfig(), metadata=_PER_FILE)
    controller._tasks_for_file.return_value = []
    return controller


def _synced_metadata(active: MetadataConfig) -> MetadataConfig:
    controller = _controller(active)
    AppController._build_preset_export_tasks(controller, [{"hash": "h1", "path": "/a/1.nef"}], [SimpleNamespace()])
    return controller._tasks_for_file.call_args.kwargs["metadata_config"]


def test_sync_on_sends_the_active_frames_config_to_every_file() -> None:
    assert _synced_metadata(_ACTIVE) is _ACTIVE


def test_sync_off_sends_each_files_own_config() -> None:
    assert _synced_metadata(replace(_ACTIVE, sync_to_batch=False)) is _PER_FILE


def test_every_metadata_field_is_carried() -> None:
    """The active config is passed whole, so no field can be left behind."""
    synced = _synced_metadata(_ACTIVE)
    for field in fields(MetadataConfig):
        assert getattr(synced, field.name) == getattr(_ACTIVE, field.name)


def test_capture_date_and_place_reach_the_export() -> None:
    synced = _synced_metadata(_ACTIVE)
    assert synced.capture_date == "1998-07-14 16:30"
    assert (synced.gps_latitude, synced.gps_longitude) == (35.6762, 139.6503)
    assert (synced.location_city, synced.location_country) == ("Tokyo", "Japan")
