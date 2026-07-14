import gc
from dataclasses import replace
from unittest.mock import MagicMock, patch

from negpy.desktop.controller import AppController, _autocrop_fingerprint
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.desktop.workers.render import BatchAutoCropResult
from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.models import AutocropMode
from negpy.services.rendering.preview_manager import PreviewManager


class TestBatchAutoCropController:
    def setup_method(self) -> None:
        self.session = MagicMock(spec=DesktopSessionManager)
        self.session.state = AppState()
        self.session.repo = MagicMock()
        self.session.asset_model = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as render_worker,
            patch("negpy.desktop.controller.PreviewManager") as preview_manager,
        ):
            render_worker.return_value = MagicMock()
            preview_manager.side_effect = lambda: MagicMock(spec=PreviewManager)
            self.controller = AppController(self.session)

        # Keep dispatch synchronous and observable; worker behavior has its own tests.
        self.controller.batch_autocrop_requested.disconnect(self.controller.batch_autocrop_worker.process)
        self.tasks = []
        self.controller.batch_autocrop_requested.connect(self.tasks.append)

    def teardown_method(self) -> None:
        self.controller.batch_autocrop_worker.cancel()
        for thread in (
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_request_dispatches_visible_uncropped_frames_and_preserves_manual(self) -> None:
        files = [
            {"name": "active.dng", "path": "/roll/active.dng", "hash": "active"},
            {"name": "manual.dng", "path": "/roll/manual.dng", "hash": "manual"},
            {"name": "fresh.dng", "path": "/roll/fresh.dng", "hash": "fresh"},
            {"name": "hidden.dng", "path": "/roll/hidden.dng", "hash": "hidden"},
        ]
        self.controller.state.uploaded_files = files
        self.controller.state.current_file_hash = "active"
        self.controller.state.current_file_path = files[0]["path"]
        active = replace(WorkspaceConfig(), geometry=replace(WorkspaceConfig().geometry, autocrop_ratio="4:3"))
        manual = replace(WorkspaceConfig(), geometry=replace(WorkspaceConfig().geometry, manual_crop_rect=(0.1, 0.1, 0.9, 0.9)))
        fresh = WorkspaceConfig()
        self.controller.state.config = active
        self.session.asset_model.visible_actual_indices_ordered.return_value = [0, 1, 2]
        self.session.config_for_asset.side_effect = lambda asset: {"manual": manual, "fresh": fresh}[asset["hash"]]

        self.controller.request_batch_auto_crop()

        assert len(self.tasks) == 1
        assert [frame.file_info["hash"] for frame in self.tasks[0].frames] == ["active", "fresh"]
        assert self.tasks[0].frames[0].config.geometry.autocrop_ratio == "4:3"
        assert self.controller._autocrop_preflight_skipped == 1
        assert self.controller._active_batch == "autocrop"
        assert self.tasks[0].generation == self.controller._autocrop_batch_token
        self.controller._on_batch_autocrop_cancelled()

    def test_batch_autocrop_uses_a_private_preview_cache(self) -> None:
        assert self.controller.batch_autocrop_preview_service is not self.controller.preview_service

    def test_request_skips_nonactive_film_mode_assets(self) -> None:
        active_asset = {"name": "active.dng", "path": "/roll/active.dng", "hash": "active"}
        film_asset = {"name": "film.dng", "path": "/roll/film.dng", "hash": "film"}
        self.controller.state.uploaded_files = [active_asset, film_asset]
        self.controller.state.current_file_hash = "active"
        self.controller.state.current_file_path = active_asset["path"]
        self.controller.state.config = WorkspaceConfig()
        film_geometry = replace(WorkspaceConfig().geometry, autocrop_mode=AutocropMode.FILM)
        self.session.config_for_asset.return_value = replace(WorkspaceConfig(), geometry=film_geometry)
        self.session.asset_model.visible_actual_indices_ordered.return_value = [0, 1]

        self.controller.request_batch_auto_crop()

        assert [frame.file_info["hash"] for frame in self.tasks[0].frames] == ["active"]
        assert self.controller._autocrop_preflight_skipped == 1
        self.controller._on_batch_autocrop_cancelled()

    def test_autocrop_fingerprint_changes_with_mode(self) -> None:
        image = WorkspaceConfig()
        film = replace(image, geometry=replace(image.geometry, autocrop_mode=AutocropMode.FILM))

        assert _autocrop_fingerprint(image, "Display P3") != _autocrop_fingerprint(film, "Display P3")

    def test_request_rejects_film_mode(self) -> None:
        geometry = replace(WorkspaceConfig().geometry, autocrop_mode=AutocropMode.FILM)
        self.controller.state.config = replace(WorkspaceConfig(), geometry=geometry)

        self.controller.request_batch_auto_crop()

        assert self.tasks == []
        assert self.controller._active_batch is None

    def test_finish_merges_only_crop_and_rotation_then_invalidates_bounds(self) -> None:
        active_asset = {"name": "a.dng", "path": "/roll/a.dng", "hash": "a"}
        other_asset = {"name": "b.dng", "path": "/roll/b.dng", "hash": "b"}
        process = replace(
            WorkspaceConfig().process,
            local_floors=(0.1, 0.2, 0.3),
            local_ceils=(0.7, 0.8, 0.9),
        )
        active = replace(WorkspaceConfig(), process=process, geometry=replace(WorkspaceConfig().geometry, fine_rotation=1.0))
        other = replace(WorkspaceConfig(), process=process, geometry=replace(WorkspaceConfig().geometry, fine_rotation=-0.5))
        self.controller.state.current_file_hash = "a"
        self.controller.state.current_file_path = active_asset["path"]
        self.controller.state.config = active
        self.session.config_for_asset.return_value = other
        self.controller.request_render = MagicMock()
        token = self.controller._begin_batch("autocrop", "Auto cropping roll", True)
        self.controller._autocrop_batch_token = token
        self.controller._autocrop_dispatched = 2
        self.controller._autocrop_preflight_skipped = 0
        rect_a = (0.1, 0.2, 0.9, 0.8)
        rect_b = (0.12, 0.22, 0.88, 0.78)
        results = [
            BatchAutoCropResult(
                active_asset,
                _autocrop_fingerprint(active, self.controller.state.workspace_color_space),
                rect_a,
                0.75,
                0.9,
                False,
            ),
            BatchAutoCropResult(
                other_asset,
                _autocrop_fingerprint(other, self.controller.state.workspace_color_space),
                rect_b,
                0.25,
                0.8,
                True,
            ),
        ]

        self.controller._on_batch_autocrop_finished(results)

        active_saved = self.session.persist_active_batch_config.call_args.args[0]
        assert active_saved.geometry.manual_crop_rect == rect_a
        assert active_saved.geometry.fine_rotation == 1.75
        assert not active_saved.geometry.auto_crop_enabled
        assert active_saved.process.local_floors == (0.0, 0.0, 0.0)
        assert active_saved.process.local_ceils == (0.0, 0.0, 0.0)
        self.session.persist_active_batch_config.assert_called_once_with(active_saved)
        self.session.update_config.assert_not_called()

        _, other_saved = self.session.repo.save_file_settings.call_args.args[:2]
        assert other_saved.geometry.manual_crop_rect == rect_b
        assert other_saved.geometry.fine_rotation == -0.25
        assert other_saved.process.local_floors == (0.0, 0.0, 0.0)
        self.controller.request_render.assert_called_once_with()
        assert self.controller._active_batch is None

    def test_persistence_failure_releases_batch_lane(self) -> None:
        asset = {"name": "bad.dng", "path": "/roll/bad.dng", "hash": "bad"}
        config = WorkspaceConfig()
        self.session.config_for_asset.return_value = config
        self.session.repo.save_file_settings.side_effect = RuntimeError("database unavailable")
        token = self.controller._begin_batch("autocrop", "Auto cropping roll", True)
        self.controller._autocrop_batch_token = token
        self.controller._autocrop_dispatched = 1
        result = BatchAutoCropResult(
            asset,
            _autocrop_fingerprint(config, self.controller.state.workspace_color_space),
            (0.1, 0.1, 0.9, 0.9),
            0.0,
            0.8,
            False,
        )

        self.controller._on_batch_autocrop_finished([result])

        assert self.controller._active_batch is None
        assert self.controller._autocrop_batch_token is None

    def test_active_persistence_failure_does_not_expose_crop_in_memory(self) -> None:
        asset = {"name": "active.dng", "path": "/roll/active.dng", "hash": "active"}
        config = WorkspaceConfig()
        self.controller.state.current_file_hash = "active"
        self.controller.state.current_file_path = asset["path"]
        self.controller.state.config = config
        self.controller.request_render = MagicMock()
        self.session.persist_active_batch_config.side_effect = RuntimeError("database unavailable")
        token = self.controller._begin_batch("autocrop", "Auto cropping roll", True)
        self.controller._autocrop_batch_token = token
        self.controller._autocrop_dispatched = 1
        result = BatchAutoCropResult(
            asset,
            _autocrop_fingerprint(config, self.controller.state.workspace_color_space),
            (0.1, 0.1, 0.9, 0.9),
            0.5,
            0.8,
            False,
        )

        self.controller._on_batch_autocrop_finished([result])

        assert self.controller.state.config is config
        self.controller.request_render.assert_not_called()
        assert self.controller._active_batch is None

    def test_cancel_requested_discards_a_queued_finished_result(self) -> None:
        asset = {"name": "late.dng", "path": "/roll/late.dng", "hash": "late"}
        config = WorkspaceConfig()
        token = self.controller._begin_batch("autocrop", "Auto cropping roll", True)
        self.controller._autocrop_batch_token = token
        self.controller._autocrop_cancel_requested = True
        result = BatchAutoCropResult(
            asset,
            _autocrop_fingerprint(config, self.controller.state.workspace_color_space),
            (0.1, 0.1, 0.9, 0.9),
            0.0,
            0.8,
            False,
        )

        self.controller._on_batch_autocrop_finished([result])

        self.session.repo.save_file_settings.assert_not_called()
        assert self.controller._active_batch is None

    def test_finish_preserves_new_manual_crop_and_changed_geometry(self) -> None:
        manual_asset = {"name": "manual.dng", "path": "/roll/manual.dng", "hash": "manual"}
        changed_asset = {"name": "changed.dng", "path": "/roll/changed.dng", "hash": "changed"}
        original = WorkspaceConfig()
        manual = replace(original, geometry=replace(original.geometry, manual_crop_rect=(0.1, 0.1, 0.8, 0.8)))
        changed = replace(original, geometry=replace(original.geometry, fine_rotation=2.0))
        self.session.config_for_asset.side_effect = [manual, changed]
        token = self.controller._begin_batch("autocrop", "Auto cropping roll", True)
        self.controller._autocrop_batch_token = token
        self.controller._autocrop_dispatched = 2
        results = [
            BatchAutoCropResult(
                manual_asset,
                _autocrop_fingerprint(original, self.controller.state.workspace_color_space),
                (0.1, 0.1, 0.9, 0.9),
                0.0,
                0.8,
                False,
            ),
            BatchAutoCropResult(
                changed_asset,
                _autocrop_fingerprint(original, self.controller.state.workspace_color_space),
                (0.1, 0.1, 0.9, 0.9),
                0.0,
                0.8,
                False,
            ),
        ]

        self.controller._on_batch_autocrop_finished(results)

        self.session.repo.save_file_settings.assert_not_called()
        self.session.update_config.assert_not_called()
        assert self.controller._active_batch is None

    def test_active_batch_blocks_other_batch_entry_points_and_queues_discovery(self) -> None:
        token = self.controller._begin_batch("autocrop", "Auto cropping roll", True)
        self.controller._autocrop_batch_token = token
        self.controller.state.uploaded_files = [{"name": "a", "path": "/a", "hash": "a"}]
        self.controller.state.thumbnails = {}
        self.session.asset_model.visible_actual_indices_ordered.return_value = [0]
        normalization = []
        thumbnails = []
        discovery = []
        self.controller.normalization_requested.connect(normalization.append)
        self.controller.thumbnail_requested.connect(thumbnails.append)
        self.controller.asset_discovery_requested.connect(discovery.append)

        self.controller.request_batch_normalization()
        self.controller.request_batch_export()
        self.controller.request_contact_sheet()
        self.controller.generate_missing_thumbnails()
        self.controller.request_asset_discovery(["/roll"])

        assert normalization == []
        assert thumbnails == []
        assert discovery == []
        assert len(self.controller._pending_asset_discoveries) == 1
        self.controller._pending_asset_discoveries.clear()
        self.controller._on_batch_autocrop_cancelled()

    def test_stale_completion_cannot_overwrite_a_new_batch(self) -> None:
        self.controller._autocrop_batch_token = 1
        self.controller._active_batch = "normalization"
        self.controller._active_batch_token = 2

        self.controller._on_batch_autocrop_finished([])

        self.session.repo.save_file_settings.assert_not_called()
        assert self.controller._active_batch == "normalization"
