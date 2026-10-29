from dataclasses import replace

import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import BwFilter
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.migrations.source_color import migrate_bw_source_color


@pytest.fixture
def repo(tmp_path):
    r = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    r.initialize()
    return r


def _config(mode=ProcessMode.BW, source_color=True, bw_filter=BwFilter.NONE):
    cfg = WorkspaceConfig()
    return replace(
        cfg,
        process=replace(cfg.process, process_mode=mode, source_color=source_color),
        exposure=replace(cfg.exposure, bw_filter=bw_filter),
    )


def test_clears_source_color_on_a_bw_frame_without_a_filter(repo):
    repo.save_file_settings("h1", _config(), file_path="/a.nef")

    migrate_bw_source_color(repo)

    assert repo.load_file_settings("h1").process.source_color is None


def test_keeps_source_color_where_it_reaches_the_print(repo):
    repo.save_file_settings("h1", _config(bw_filter=BwFilter.RED_25), file_path="/a.nef")
    repo.save_file_settings("h2", _config(mode=ProcessMode.C41), file_path="/b.nef")

    migrate_bw_source_color(repo)

    assert repo.load_file_settings("h1").process.source_color is True
    assert repo.load_file_settings("h2").process.source_color is True


def test_runs_once(repo):
    migrate_bw_source_color(repo)
    repo.save_file_settings("h1", _config(), file_path="/a.nef")

    migrate_bw_source_color(repo)

    assert repo.load_file_settings("h1").process.source_color is True
