"""A batch or preset export writes each file's own metadata. Metadata cards are roll
defaults, so a roll's frames already agree unless one was deliberately set apart."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.models import MetadataConfig


def test_preset_export_sends_each_files_own_metadata() -> None:
    per_file = MetadataConfig(film="Velvia 50")
    controller = MagicMock()
    controller.state.config = replace(WorkspaceConfig(), metadata=MetadataConfig(film="Portra 400"))
    controller.state.current_file_hash = "not-this-one"
    controller._batch_params_for.return_value = replace(WorkspaceConfig(), metadata=per_file)
    controller._tasks_for_file.return_value = []

    AppController._build_preset_export_tasks(controller, [{"hash": "h1", "path": "/a/1.nef"}], [SimpleNamespace()])

    assert controller._tasks_for_file.call_args.kwargs["metadata_config"] is per_file
