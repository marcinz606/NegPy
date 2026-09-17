import gc
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.desktop.workers.render import ThumbnailUpdateTask
from negpy.domain.models import WorkspaceConfig
from negpy.services.rendering.preview_manager import PreviewManager


class TestThumbnailRefreshController:
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
        self.controller.thumbnail_render_requested.disconnect(self.controller.thumbnail_render_worker.process)
        self.tasks = []
        self.controller.thumbnail_render_requested.connect(self.tasks.append)
        self.thumbnail_updates = []
        self.controller.thumbnail_update_requested.connect(self.thumbnail_updates.append)

        self.files = [
            {"name": "active.dng", "path": "/roll/active.dng", "hash": "active", "diptych": False},
            {"name": "other.dng", "path": "/roll/other.dng", "hash": "other", "diptych": False},
            {"name": "third.dng", "path": "/roll/third.dng", "hash": "third", "diptych": False},
        ]
        self.controller.state.uploaded_files = self.files
        self.controller.state.current_file_hash = "active"
        self.controller.state.current_file_path = self.files[0]["path"]
        self.controller.state.config = WorkspaceConfig()
        self.session.config_for_asset.return_value = WorkspaceConfig()

    def teardown_method(self) -> None:
        self.controller.thumbnail_render_worker.cancel()
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

    def test_uses_a_private_preview_cache(self) -> None:
        assert self.controller.thumbnail_render_preview_service is not self.controller.preview_service
        assert self.controller.thumbnail_render_preview_service is not self.controller.batch_autocrop_preview_service

    def test_dispatch_excludes_active_frame(self) -> None:
        self.controller.refresh_thumbnails_for(["active", "other"])

        assert len(self.tasks) == 1
        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["other"]
        assert self.controller._thumbnail_render_running is True
        assert self.tasks[0].generation == self.controller._thumbnail_render_generation
        self.controller._on_thumbnail_render_cancelled()

    def test_dispatch_excludes_diptych_rows(self) -> None:
        self.files[1]["diptych"] = True
        pair = (WorkspaceConfig(), WorkspaceConfig())

        def _diptych_pair(asset):
            return pair if asset["hash"] == "other" else None

        with patch.object(self.controller, "diptych_pair", side_effect=_diptych_pair):
            self.controller.refresh_thumbnails_for(["other", "third"])
        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["third"]
        self.controller._on_thumbnail_render_cancelled()

    def test_dispatch_dedupes_by_thumbnail_key(self) -> None:
        composite = [
            {"name": "a.dng", "path": "/roll/a.dng", "hash": "dup", "diptych": False},
            {"name": "b.dng", "path": "/roll/b.dng", "hash": "dup", "diptych": False},
        ]
        self.controller.state.uploaded_files = composite
        self.controller.refresh_thumbnails_for(["dup"])
        assert len(self.tasks[0].frames) == 1
        self.controller._on_thumbnail_render_cancelled()

    def test_dispatch_with_no_matching_frames_starts_nothing(self) -> None:
        self.controller.refresh_thumbnails_for(["not-loaded"])
        assert self.tasks == []
        assert self.controller._thumbnail_render_running is False

    def test_dispatch_is_not_blocked_by_another_batch_owning_the_lane(self) -> None:
        """The whole point of running off the shared lane: Apply Settings must not be
        refused just because Export, Auto Crop All or anything else is in progress."""
        token = self.controller._begin_batch("autocrop", "Auto cropping roll", True)
        assert token is not None

        self.controller.refresh_thumbnails_for(["other"])

        assert len(self.tasks) == 1
        assert self.controller._thumbnail_render_running is True
        self.controller._end_batch("autocrop", token)
        self.controller._on_thumbnail_render_cancelled()

    def test_dispatch_while_already_running_is_folded_into_resume_not_dropped(self) -> None:
        """A bulk write landing while a generation is already using norm_thread (e.g.
        Batch Analysis's own completion write, arriving during its own pre-emption
        window) must not be lost outright — it gets picked up the moment the current
        generation ends, same as a pre-emption's own leftover frames."""
        self.controller.refresh_thumbnails_for(["other"])
        assert len(self.tasks) == 1

        self.controller.refresh_thumbnails_for(["third"])

        assert len(self.tasks) == 1  # not dispatched yet, but not dropped either
        assert self.controller._thumbnail_render_resume == {"third"}

        self.controller._on_thumbnail_render_finished(1)

        assert len(self.tasks) == 2
        assert [f.file_info["hash"] for f in self.tasks[1].frames] == ["third"]
        self.controller._on_thumbnail_render_cancelled()

    def test_begin_batch_preempts_a_running_refresh(self) -> None:
        """A real batch claiming the lane must give the refresh's own cancel machinery
        a nudge, so it yields norm_thread within one frame instead of finishing the roll."""
        self.controller.refresh_thumbnails_for(["other"])
        generation = self.controller._thumbnail_render_generation
        self.controller.thumbnail_render_worker.cancel = MagicMock()

        token = self.controller._begin_batch("autocrop", "Auto cropping roll", True)

        self.controller.thumbnail_render_worker.cancel.assert_called_once_with(generation)
        self.controller._end_batch("autocrop", token)
        self.controller._on_thumbnail_render_cancelled()

    def test_begin_batch_does_not_preempt_when_no_refresh_is_running(self) -> None:
        self.controller.thumbnail_render_worker.cancel = MagicMock()

        token = self.controller._begin_batch("autocrop", "Auto cropping roll", True)

        self.controller.thumbnail_render_worker.cancel.assert_not_called()
        self.controller._end_batch("autocrop", token)

    def test_begin_batch_refused_while_another_batch_runs_does_not_preempt(self) -> None:
        self.controller.refresh_thumbnails_for(["other"])
        first = self.controller._begin_batch("autocrop", "Auto cropping roll", True)
        self.controller.thumbnail_render_worker.cancel = MagicMock()

        second = self.controller._begin_batch("normalization", "Analyzing roll", True)

        assert second is None
        self.controller.thumbnail_render_worker.cancel.assert_not_called()
        self.controller._end_batch("autocrop", first)
        self.controller._on_thumbnail_render_cancelled()

    def test_begin_batch_does_not_preempt_for_an_owner_off_norm_thread(self) -> None:
        """Export, Stitch, HDR, discovery and thumbnail generation share no thread or
        CPU with the refresh, so a routine single-frame export must not strand the
        rest of a roll-wide refresh mid-flight — only autocrop/normalization contend."""
        self.controller.refresh_thumbnails_for(["other"])
        self.controller.thumbnail_render_worker.cancel = MagicMock()

        token = self.controller._begin_batch("export", "Exporting", True)

        self.controller.thumbnail_render_worker.cancel.assert_not_called()
        assert self.controller._thumbnail_render_running is True
        self.controller._end_batch("export", token)
        self.controller._on_thumbnail_render_cancelled()

    def test_on_rendered_emits_thumbnail_update_task_with_persist_true(self) -> None:
        self.controller.refresh_thumbnails_for(["other"])
        buf = np.full((4, 6, 3), 0.5, dtype=np.float32)

        self.controller._on_thumbnail_rendered(self.tasks[0].frames[0], buf)

        assert len(self.thumbnail_updates) == 1
        task: ThumbnailUpdateTask = self.thumbnail_updates[0]
        assert task.file_hash == self.tasks[0].frames[0].thumbnail_key
        assert task.persist is True
        assert task.buffer is buf
        self.controller._on_thumbnail_render_cancelled()

    def test_on_rendered_passes_the_frames_own_process_to_the_display_transform(self) -> None:
        seen: list = []
        original = self.controller.display_transform_params

        def _spy(*args, **kwargs):
            seen.append(kwargs.get("process"))
            return original(*args, **kwargs)

        self.controller.display_transform_params = _spy
        self.controller.refresh_thumbnails_for(["other"])
        frame = self.tasks[0].frames[0]

        self.controller._on_thumbnail_rendered(frame, np.zeros((2, 2, 3), dtype=np.float32))

        assert seen == [frame.config.process]
        self.controller._on_thumbnail_render_cancelled()

    def test_on_rendered_after_abort_for_a_frame_no_longer_eligible_emits_nothing(self) -> None:
        """A stale signal from a cut-short generation is dropped when the frame is no
        longer eligible for the resume that cancellation immediately triggers (it
        became the active frame in the meantime)."""
        self.controller.refresh_thumbnails_for(["other"])
        frame = self.tasks[0].frames[0]
        self.controller.state.current_file_hash = "other"
        self.controller._on_thumbnail_render_cancelled()
        assert self.controller._thumbnail_render_running is False  # resume found nothing eligible

        self.controller._on_thumbnail_rendered(frame, np.zeros((2, 2, 3), dtype=np.float32))

        assert self.thumbnail_updates == []

    def test_cancellation_immediately_resumes_stranded_frames(self) -> None:
        """Pre-emption isn't the end of the story: whatever didn't get its turn is
        requested again as soon as the interrupting batch is done with norm_thread —
        Qt's own queueing keeps this from ever racing that real batch work."""
        self.controller.refresh_thumbnails_for(["other", "third"])
        assert len(self.tasks) == 1
        first_frames = {f.thumbnail_key for f in self.tasks[0].frames}

        self.controller._on_thumbnail_render_cancelled()

        assert len(self.tasks) == 2
        resumed_frames = {f.thumbnail_key for f in self.tasks[1].frames}
        assert resumed_frames == first_frames
        assert self.tasks[1].generation != self.tasks[0].generation
        assert self.controller._thumbnail_render_running is True
        self.controller._on_thumbnail_render_cancelled()

    def test_cancellation_with_nothing_pending_does_not_resume(self) -> None:
        self.controller.refresh_thumbnails_for(["other"])
        frame = self.tasks[0].frames[0]
        self.controller._on_thumbnail_rendered(frame, np.zeros((2, 2, 3), dtype=np.float32))  # drains pending

        self.controller._on_thumbnail_render_cancelled()

        assert len(self.tasks) == 1  # no resume dispatch
        assert self.controller._thumbnail_render_running is False

    def test_error_does_not_resume_the_stranded_frames(self) -> None:
        """A hard worker failure isn't retried automatically — an error that recurs
        deterministically must not become an infinite resume loop."""
        self.controller.refresh_thumbnails_for(["other"])

        self.controller._on_thumbnail_render_error("boom")

        assert len(self.tasks) == 1  # no resume dispatch
        assert self.controller._thumbnail_render_running is False

    def test_on_rendered_for_the_now_active_frame_emits_nothing(self) -> None:
        """Opened since dispatch: the live render already owns this frame's thumbnail,
        and persisting a background render over it would fight the canvas."""
        self.controller.refresh_thumbnails_for(["other"])
        frame = self.tasks[0].frames[0]
        self.controller.state.current_file_hash = "other"

        self.controller._on_thumbnail_rendered(frame, np.zeros((2, 2, 3), dtype=np.float32))

        assert self.thumbnail_updates == []
        self.controller._on_thumbnail_render_cancelled()

    def test_on_rendered_for_a_config_edited_since_dispatch_emits_nothing(self) -> None:
        """The frame's own settings changed again while the background render was in
        flight — the dispatched config is stale, so the result must not be persisted."""
        self.controller.refresh_thumbnails_for(["other"])
        frame = self.tasks[0].frames[0]
        self.session.config_for_asset.return_value = replace(
            WorkspaceConfig(), process=replace(WorkspaceConfig().process, analysis_buffer=0.9)
        )

        self.controller._on_thumbnail_rendered(frame, np.zeros((2, 2, 3), dtype=np.float32))

        assert self.thumbnail_updates == []
        self.controller._on_thumbnail_render_cancelled()

    def test_on_rendered_for_a_removed_frame_emits_nothing(self) -> None:
        self.controller.refresh_thumbnails_for(["other"])
        frame = self.tasks[0].frames[0]
        self.controller.state.uploaded_files = [f for f in self.files if f["hash"] != "other"]

        self.controller._on_thumbnail_rendered(frame, np.zeros((2, 2, 3), dtype=np.float32))

        assert self.thumbnail_updates == []
        self.controller._on_thumbnail_render_cancelled()

    def test_on_rendered_does_not_touch_live_render_state(self) -> None:
        """The background path shares no signal, state slot or worker with the live
        render — this is the guard against the crosstalk the design avoids."""
        self.controller.refresh_thumbnails_for(["other"])
        self.controller.state.last_metrics["source_hash"] = "sentinel-active-frame-hash"
        sentinel_metrics = dict(self.controller.state.last_metrics)
        self.controller._is_rendering = False
        image_updated = MagicMock()
        render_requested = MagicMock()
        self.controller.image_updated.connect(image_updated)
        self.controller.render_requested.connect(render_requested)

        self.controller._on_thumbnail_rendered(self.tasks[0].frames[0], np.full((4, 6, 3), 0.5, dtype=np.float32))

        assert self.controller.state.last_metrics == sentinel_metrics
        assert self.controller._is_rendering is False
        image_updated.assert_not_called()
        render_requested.assert_not_called()
        self.controller._on_thumbnail_render_cancelled()

    def test_finished_clears_the_running_flag_and_reports_status(self) -> None:
        self.controller.refresh_thumbnails_for(["other"])
        assert self.controller._thumbnail_render_running is True

        self.controller._on_thumbnail_render_finished(1)

        assert self.controller._thumbnail_render_running is False

    def test_error_clears_the_running_flag(self) -> None:
        self.controller.refresh_thumbnails_for(["other"])

        self.controller._on_thumbnail_render_error("boom")

        assert self.controller._thumbnail_render_running is False

    def test_abort_active_batch_never_touches_the_refresh_worker(self) -> None:
        """The refresh holds no batch lane, so the shared Abort action has nothing to
        do with it — only `_begin_batch` (via `_preempt_background_thumbnail_refresh`)
        pre-empts a running refresh."""
        self.controller.refresh_thumbnails_for(["other"])
        self.controller.thumbnail_render_worker.cancel = MagicMock()

        self.controller.abort_active_batch()

        self.controller.thumbnail_render_worker.cancel.assert_not_called()
        self.controller._on_thumbnail_render_cancelled()

    def test_dispatch_marks_icc_input_active_for_a_narrowband_frame(self) -> None:
        narrowband = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, narrowband_scan=True))
        self.session.config_for_asset.return_value = narrowband

        self.controller.refresh_thumbnails_for(["other"])

        assert self.tasks[0].frames[0].icc_input_active is True
        self.controller._on_thumbnail_render_cancelled()

    def test_dispatch_leaves_icc_input_inactive_for_an_ordinary_frame(self) -> None:
        self.controller.refresh_thumbnails_for(["other"])

        assert self.tasks[0].frames[0].icc_input_active is False
        self.controller._on_thumbnail_render_cancelled()

    def test_uses_config_for_asset_resolved_per_frame(self) -> None:
        custom = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, analysis_buffer=0.3))
        self.session.config_for_asset.return_value = custom

        self.controller.refresh_thumbnails_for(["other"])

        assert self.tasks[0].frames[0].config == custom
        self.controller._on_thumbnail_render_cancelled()
