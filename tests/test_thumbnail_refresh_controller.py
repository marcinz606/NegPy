import gc
import shutil
import tempfile
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from negpy.desktop.controller import _THUMBNAIL_REFRESH_MEMORY_RETRY_MS, AppController, thumbnail_refresh_progress_text
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.desktop.workers.render import ThumbnailRenderWorker, ThumbnailUpdateTask
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.local_asset_store import LocalAssetStore
from negpy.services.rendering.preview_manager import PreviewManager, _linear_preview_key


class TestThumbnailRefreshController:
    def setup_method(self) -> None:
        self.session = MagicMock(spec=DesktopSessionManager)
        self.session.state = AppState()
        self.session.repo = MagicMock()
        self.session.asset_model = MagicMock()
        # A private thumbnail cache: the refresh writes real JPEGs (and their fingerprints)
        # through the thumbnail worker, which must not land in the user's own cache or
        # leak from one test into the next.
        self._thumb_cache = tempfile.mkdtemp(prefix="negpy-thumb-test-")
        with (
            patch("negpy.desktop.controller.RenderWorker") as render_worker,
            patch("negpy.desktop.controller.PreviewManager") as preview_manager,
            patch(
                "negpy.desktop.controller.LocalAssetStore",
                side_effect=lambda _cache, icc: LocalAssetStore(self._thumb_cache, icc),
            ),
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
        shutil.rmtree(self._thumb_cache, ignore_errors=True)

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

    def test_dispatch_under_memory_pressure_is_deferred_not_dropped(self) -> None:
        """A refresh must not grow its own preview cache on top of an already-tight
        system, but the request itself is retried later rather than lost outright."""
        with (
            patch("negpy.desktop.controller.available_system_memory_bytes", return_value=1),
            patch("negpy.desktop.controller.QTimer.singleShot") as singleshot,
        ):
            self.controller.refresh_thumbnails_for(["other"])

        assert self.tasks == []
        assert self.controller._thumbnail_render_running is False
        delay, retry = singleshot.call_args[0]
        assert delay == _THUMBNAIL_REFRESH_MEMORY_RETRY_MS

        retry()  # memory pressure has passed by the time the timer fires

        assert len(self.tasks) == 1
        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["other"]
        self.controller._on_thumbnail_render_cancelled()

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
        Roll Analysis's own completion write, arriving during its own pre-emption
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

    def test_request_thumbnail_refresh_selection_scope_uses_the_multi_selection(self) -> None:
        self.controller.state.selected_indices = [1, 2]  # other, third

        self.controller.request_thumbnail_refresh("selection")

        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["other", "third"]
        self.controller._on_thumbnail_render_cancelled()

    def test_request_thumbnail_refresh_selection_scope_falls_back_to_the_active_frame(self) -> None:
        """No multi-selection: the currently active/single-selected frame is the target
        (matching Keep/Reject's own targets fallback in the context menu)."""
        self.controller.state.selected_indices = []
        self.controller.state.selected_file_idx = 1  # other

        self.controller.request_thumbnail_refresh("selection")

        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["other"]
        self.controller._on_thumbnail_render_cancelled()

    def test_request_thumbnail_refresh_roll_scope_uses_every_visible_frame(self) -> None:
        self.session.asset_model.visible_actual_indices_ordered.return_value = [0, 1, 2]

        self.controller.request_thumbnail_refresh("roll")

        # The active frame is excluded by refresh_thumbnails_for itself.
        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["other", "third"]
        self.controller._on_thumbnail_render_cancelled()

    def test_request_thumbnail_refresh_with_nothing_selected_reports_status_not_a_dispatch(self) -> None:
        self.controller.state.selected_indices = []
        self.controller.state.selected_file_idx = -1
        statuses = []
        self.controller.status_message_requested.connect(lambda msg, *_a, **_k: statuses.append(msg))

        self.controller.request_thumbnail_refresh("selection")

        assert self.tasks == []
        assert statuses == ["Nothing to update"]

    def test_thumbnail_refresh_state_changed_fires_on_dispatch_and_on_finish(self) -> None:
        states = []
        self.controller.thumbnail_refresh_state_changed.connect(states.append)

        self.controller.refresh_thumbnails_for(["other"])
        assert states == [True]

        self.controller._on_thumbnail_render_finished(1)
        assert states == [True, False]

    def test_cancel_thumbnail_refresh_stops_a_running_generation_without_resuming(self) -> None:
        """The user's own escape hatch: unlike a real batch's pre-emption, nothing
        gets requested again after this — a very large accidental folder must
        actually be stoppable, not just paused."""
        self.controller.refresh_thumbnails_for(["other", "third"])
        assert self.controller.thumbnail_refresh_running is True

        self.controller.cancel_thumbnail_refresh()
        self.controller._on_thumbnail_render_cancelled()  # simulates the worker's async signal

        assert self.controller.thumbnail_refresh_running is False
        assert self.controller._thumbnail_render_resume == set()
        assert len(self.tasks) == 1  # no resume dispatch

    def test_cancel_thumbnail_refresh_sets_a_status_message(self) -> None:
        self.controller.refresh_thumbnails_for(["other"])
        statuses = []
        self.controller.status_message_requested.connect(lambda msg, *_a, **_k: statuses.append(msg))

        self.controller.cancel_thumbnail_refresh()
        self.controller._on_thumbnail_render_cancelled()

        assert statuses == ["Thumbnail update cancelled"]

    def test_cancel_thumbnail_refresh_also_discards_an_already_folded_backlog(self) -> None:
        """A bulk write folded into the resume backlog while a manual refresh was
        running must not survive a Cancel either — the user asked to stop, full stop."""
        self.controller.refresh_thumbnails_for(["other"])
        self.controller.refresh_thumbnails_for(["third"])  # folded, not dispatched
        assert self.controller._thumbnail_render_resume == {"third"}

        self.controller.cancel_thumbnail_refresh()
        self.controller._on_thumbnail_render_cancelled()

        assert self.controller._thumbnail_render_resume == set()
        assert len(self.tasks) == 1

    def test_cancel_thumbnail_refresh_is_a_noop_with_nothing_running(self) -> None:
        self.controller.thumbnail_render_worker.cancel = MagicMock()

        self.controller.cancel_thumbnail_refresh()

        self.controller.thumbnail_render_worker.cancel.assert_not_called()

    def test_progress_running_mean_resets_on_a_new_generation(self) -> None:
        with patch.object(self.controller, "set_status"):
            self.controller.refresh_thumbnails_for(["other", "third"])
            self.controller._on_thumbnail_render_progress(1, 2, "other.dng", 20.0, 1.0)
            self.controller._on_thumbnail_render_progress(2, 2, "third.dng", 20.0, 1.0)
            assert self.controller._thumbnail_render_means() == (20.0, 1.0)
            self.controller._on_thumbnail_render_finished(2)

            self.controller.refresh_thumbnails_for(["other"])
            assert self.controller._thumbnail_render_means() == (0.0, 0.0)
            self.controller._on_thumbnail_render_progress(1, 1, "other.dng", 1.0, 0.5)
            assert self.controller._thumbnail_render_means() == (1.0, 0.5)
        self.controller._on_thumbnail_render_cancelled()

    def test_a_refresh_reports_progress_off_the_toast_and_clears_it_at_the_end(self) -> None:
        bar, lines = [], []
        self.controller.status_progress_requested.connect(lambda cur, tot: bar.append((cur, tot)))
        self.controller.thumbnail_refresh_progress.connect(lines.append)
        with patch.object(self.controller, "set_status") as set_status:
            self.controller.refresh_thumbnails_for(["other", "third"])
            set_status.reset_mock()
            self.controller._on_thumbnail_render_frame_started(1, 2, "other.dng")
            self.controller._on_thumbnail_render_progress(1, 2, "other.dng", 1.0, 0.5)
            self.controller._on_thumbnail_render_frame_started(2, 2, "third.dng")
            self.controller._on_thumbnail_render_progress(2, 2, "third.dng", 1.0, 0.5)
            assert set_status.call_count == 0
            self.controller._on_thumbnail_render_finished(2)
        assert bar == [(0, 2), (1, 2), (2, 2), (0, 0)]
        assert lines == ["Thumbnails 1/2", "Thumbnails 1/2", "Thumbnails 2/2", "Thumbnails 2/2", ""]

    def test_frame_started_counts_the_frame_in_flight_as_left(self) -> None:
        lines = []
        self.controller.thumbnail_refresh_progress.connect(lines.append)
        self.controller.refresh_thumbnails_for(["other", "third"])
        self.controller._thumbnail_render_timing = [4.0, 6.0, 2]
        self.controller._on_thumbnail_render_frame_started(3, 5, "c.dng")
        # Frames 3, 4 and 5 are left at 5 s each.
        assert lines[-1] == "Thumbnails 3/5 · ~15 s left"
        self.controller._on_thumbnail_render_cancelled()

    def test_a_cancelled_refresh_clears_its_progress(self) -> None:
        bar, lines = [], []
        self.controller.status_progress_requested.connect(lambda cur, tot: bar.append((cur, tot)))
        self.controller.thumbnail_refresh_progress.connect(lines.append)
        self.controller.refresh_thumbnails_for(["other"])
        self.controller._on_thumbnail_render_progress(1, 2, "other.dng", 1.0, 0.5)
        self.controller.cancel_thumbnail_refresh()
        self.controller._on_thumbnail_render_cancelled()
        assert bar[-1] == (0, 0)
        assert lines[-1] == ""

    def test_live_worker_reads_the_navigation_preview_service(self) -> None:
        assert self.controller.thumbnail_render_worker._live_preview_service is self.controller.preview_service

    def test_neighbor_prefetch_key_matches_the_thumbnail_refresh_key(self) -> None:
        """A plain whole frame the neighbor prefetch cached is a live-cache hit for the refresh."""
        asset = self.files[1]
        self.session.repo.load_file_settings.return_value = WorkspaceConfig()
        task = self.controller._neighbor_prefetch_task(asset, 0, ())
        assert task is not None
        live = PreviewManager()
        live._cache.put(
            _linear_preview_key(
                task.file_hash,
                color_space=task.workspace_color_space,
                use_camera_wb=task.use_camera_wb,
                full_resolution=task.full_resolution,
                half_slice=task.half_slice,
                demosaic=task.demosaic,
                positive_source=task.positive_source,
                highlight_mode=task.highlight_mode,
                bake_camera_wb=task.bake_camera_wb,
                lens_corrections=task.lens_corrections,
                lens_flatfield=task.lens_flatfield,
            ),
            np.zeros((4, 6, 3), dtype=np.float32),
            (6, 4),
            {},
        )
        self.controller.refresh_thumbnails_for([asset["hash"]])
        frame = self.tasks[0].frames[0]
        worker = ThumbnailRenderWorker(MagicMock(), live)

        assert worker._peek_live_preview(frame, self.tasks[0].workspace_color_space) is not None
        self.controller._on_thumbnail_render_cancelled()

    # Fingerprints: roll-scope Update Thumbnails renders only frames whose stored
    # fingerprint does not match their current settings.

    def _save(self, asset_hash: str, fingerprint) -> None:
        from PIL import Image

        from negpy.kernel.system.config import APP_CONFIG
        from negpy.services.assets.thumbnails import asset_thumbnail_key

        asset = next(f for f in self.files if f["hash"] == asset_hash)
        ts = APP_CONFIG.thumbnail_size
        self.controller.asset_store.save_thumbnail(asset_thumbnail_key(asset), Image.new("RGB", (ts, ts)), fingerprint=fingerprint)

    def _current(self, asset_hash: str) -> str:
        asset = next(f for f in self.files if f["hash"] == asset_hash)
        config = self.controller._config_for_batch_asset(asset)
        return self.controller.thumbnail_fingerprint_for(config)

    def test_roll_scope_skips_a_frame_whose_thumbnail_matches(self) -> None:
        self.session.asset_model.visible_actual_indices_ordered.return_value = [0, 1, 2]
        self._save("other", self._current("other"))
        self._save("third", "some-older-render")

        self.controller.request_thumbnail_refresh("roll")

        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["third"]
        self.controller._on_thumbnail_render_cancelled()

    def test_a_display_transform_change_leaves_thumbnails_current(self) -> None:
        self.session.asset_model.visible_actual_indices_ordered.return_value = [0, 1, 2]
        self._save("other", self._current("other"))
        self._save("third", self._current("third"))

        self.controller.state.monitor_icc_bytes = b"another-screen"
        self.controller.state.soft_proof_enabled = not self.controller.state.soft_proof_enabled
        self.controller.request_thumbnail_refresh("roll")

        assert self.tasks == []

    def test_roll_scope_treats_quick_and_unfingerprinted_thumbnails_as_stale(self) -> None:
        from negpy.services.assets.thumbnail_fingerprint import QUICK

        self.session.asset_model.visible_actual_indices_ordered.return_value = [0, 1, 2]
        self._save("other", QUICK)
        self._save("third", None)

        self.controller.request_thumbnail_refresh("roll")

        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["other", "third"]
        self.controller._on_thumbnail_render_cancelled()

    def test_a_setting_change_makes_a_matching_thumbnail_stale(self) -> None:
        self.session.asset_model.visible_actual_indices_ordered.return_value = [1]
        self._save("other", self._current("other"))
        base = WorkspaceConfig()
        self.session.config_for_asset.return_value = replace(base, exposure=replace(base.exposure, density=base.exposure.density + 0.2))

        self.controller.request_thumbnail_refresh("roll")

        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["other"]
        self.controller._on_thumbnail_render_cancelled()

    def test_this_sessions_stale_flag_wins_over_a_matching_fingerprint(self) -> None:
        from negpy.services.assets.thumbnails import asset_thumbnail_key

        self.session.asset_model.visible_actual_indices_ordered.return_value = [1]
        self._save("other", self._current("other"))
        self.controller.state.stale_thumbnails.add(asset_thumbnail_key(self.files[1]))

        self.controller.request_thumbnail_refresh("roll")

        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["other"]
        self.controller._on_thumbnail_render_cancelled()

    def test_roll_scope_with_every_thumbnail_current_dispatches_nothing(self) -> None:
        self.session.asset_model.visible_actual_indices_ordered.return_value = [0, 1, 2]
        self._save("other", self._current("other"))
        self._save("third", self._current("third"))
        statuses = []
        self.controller.status_message_requested.connect(lambda msg, *_a, **_k: statuses.append(msg))

        self.controller.request_thumbnail_refresh("roll")

        assert self.tasks == []
        assert statuses == ["All thumbnails are up to date"]

    def test_roll_scope_does_not_count_a_diptych_row_as_stale(self) -> None:
        self.session.asset_model.visible_actual_indices_ordered.return_value = [1, 2]
        self._save("third", self._current("third"))
        self.files[1]["diptych"] = True
        statuses = []
        self.controller.status_message_requested.connect(lambda msg, *_a, **_k: statuses.append(msg))

        with patch.object(
            self.controller, "diptych_pair", side_effect=lambda f: (WorkspaceConfig(), WorkspaceConfig()) if f["diptych"] else None
        ):
            self.controller.request_thumbnail_refresh("roll")

        assert self.tasks == []
        assert statuses == ["All thumbnails are up to date"]

    def test_selection_scope_forces_a_render_even_when_current(self) -> None:
        self._save("other", self._current("other"))
        self.controller.state.selected_indices = [1]

        self.controller.request_thumbnail_refresh("selection")

        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["other"]
        self.controller._on_thumbnail_render_cancelled()

    def test_background_render_carries_the_fingerprint_it_was_rendered_from(self) -> None:
        self.controller.refresh_thumbnails_for(["other"])

        self.controller._on_thumbnail_rendered(self.tasks[0].frames[0], np.zeros((2, 2, 3), dtype=np.float32))

        assert self.thumbnail_updates[0].fingerprint == self._current("other")
        self.controller._on_thumbnail_render_cancelled()

    def _live_render(self, identity, *, splash: bool = False) -> None:
        with self.controller.state.metrics_lock:
            self.controller.state.last_metrics.update(
                {
                    "base_positive": np.zeros((2, 2, 3), dtype=np.float32),
                    "source_hash": "active",
                    "splash": splash,
                    "proof": True,
                    "render_identity": identity,
                }
            )

    def test_a_frame_edited_since_its_render_is_flagged_not_written(self) -> None:
        from negpy.services.assets.thumbnails import asset_thumbnail_key

        rendered = WorkspaceConfig()
        self.controller.state.config = replace(rendered, exposure=replace(rendered.exposure, density=rendered.exposure.density + 0.3))
        self._live_render(("active", rendered))

        self.controller._update_thumbnail_from_state(persist=True)

        assert self.thumbnail_updates == []
        assert asset_thumbnail_key(self.files[0]) in self.controller.state.stale_thumbnails

    def test_an_edit_that_leaves_the_pixels_alone_still_files_the_render(self) -> None:
        rendered = WorkspaceConfig()
        self.controller.state.config = replace(rendered, metadata=replace(rendered.metadata, camera_id="cam"))
        self._live_render(("active", rendered))

        self.controller._update_thumbnail_from_state(persist=True)

        assert self.thumbnail_updates[-1].fingerprint == self.controller.thumbnail_fingerprint_for(rendered)

    def test_pixels_that_are_not_this_frames_render_never_reach_disk(self) -> None:
        self._live_render(("someone-else", WorkspaceConfig()))
        self.controller._update_thumbnail_from_state(persist=True)
        self._live_render(None)
        self.controller._update_thumbnail_from_state(persist=True)
        self._live_render(("active", WorkspaceConfig()), splash=True)
        self.controller._update_thumbnail_from_state(persist=True)

        assert self.thumbnail_updates == []

    def test_an_in_memory_thumbnail_carries_no_fingerprint(self) -> None:
        self._live_render(("active", WorkspaceConfig()))
        self.controller._update_thumbnail_from_state(persist=False)
        assert self.thumbnail_updates[-1].fingerprint is None

    def test_a_skipped_write_flags_a_thumbnail_the_edit_made_stale(self) -> None:
        from negpy.services.assets.thumbnails import asset_thumbnail_key

        self._save("active", "some-older-render")
        self._live_render(None)

        self.controller._update_thumbnail_from_state(persist=True)

        assert asset_thumbnail_key(self.files[0]) in self.controller.state.stale_thumbnails

    def test_negative_peek_drops_the_render_identity(self) -> None:
        self._live_render(("active", self.controller.state.config))
        self.controller.state.preview_raw = np.full((4, 4, 3), 0.5, dtype=np.float32)

        self.controller._paint_negative_peek()
        self.controller._update_thumbnail_from_state(persist=True)

        assert "render_identity" not in self.controller.state.last_metrics
        assert self.thumbnail_updates == []

    def _file_it_here(self) -> None:
        """The test writes the JPEG itself; the worker thread writing the same file would race it."""
        self.controller.thumbnail_update_requested.disconnect(self.controller.thumb_worker.update_rendered)

    def _leave_after_bounds_writeback(self) -> None:
        """Render the active frame, land its measured bounds, file its thumbnail on the
        way out, then make it a background frame that reads its config back from storage."""
        from types import SimpleNamespace

        def update_config(config, **_kw):
            self.controller.state.config = config

        self._file_it_here()
        self.session.update_config.side_effect = update_config
        rendered = self.controller.state.config
        self._live_render(("active", rendered))
        self.controller._on_metrics_updated(
            {"source_hash": "active", "log_bounds": SimpleNamespace(floors=(-1.0, -1.1, -1.2), ceils=(-0.9, -0.95, -1.0))}
        )
        saved = self.controller.state.config
        assert saved.process.local_floors != rendered.process.local_floors

        self.controller._update_thumbnail_from_state(persist=True)
        task = self.thumbnail_updates[-1]
        self.controller.asset_store.save_thumbnail(task.file_hash, Image.new("RGB", (4, 4)), fingerprint=task.fingerprint)

        self.controller.state.current_file_hash = "third"
        self.session.config_for_asset.side_effect = lambda a: (
            WorkspaceConfig.from_flat_dict(saved.to_dict()) if a["hash"] == "active" else WorkspaceConfig()
        )

    def test_a_canvas_render_reads_current_after_its_bounds_are_written_back(self) -> None:
        self._leave_after_bounds_writeback()
        assert self.controller.thumbnail_is_stale(self.files[0]) is False

    def test_an_edit_after_the_render_breaks_the_carried_identity(self) -> None:
        base = self.controller.state.config
        self._live_render(("active", base))
        self.controller.state.config = replace(base, exposure=replace(base.exposure, density=base.exposure.density + 0.3))
        before = self.controller.state.config
        self.controller.state.config = replace(before, process=replace(before.process, local_floors=(-1.0, -1.0, -1.0)))

        self.controller._carry_render_identity(before)

        assert self.controller.state.last_metrics["render_identity"] == ("active", base)

    def test_switching_away_skips_the_write_when_the_disk_already_holds_the_render(self) -> None:
        self._file_it_here()
        self._live_render(("active", self.controller.state.config))
        self.controller._update_thumbnail_from_state(persist=True)
        task = self.thumbnail_updates[-1]
        self.controller.asset_store.save_thumbnail(task.file_hash, Image.new("RGB", (4, 4)), fingerprint=task.fingerprint)
        count = len(self.thumbnail_updates)

        self.controller._update_thumbnail_from_state(persist=True)

        assert len(self.thumbnail_updates) == count

    # A batch turn keeps a current thumbnail current.

    def _batch_turn(self, turned: WorkspaceConfig) -> dict:
        from negpy.services.assets.thumbnails import asset_thumbnail_key

        self.controller.state.selected_indices = [0, 1, 2]
        before = self.controller.thumbnail_turn_snapshot()
        self.session.config_for_asset.return_value = turned
        keys = [asset_thumbnail_key(f) for f in self.files[1:]]
        self.controller.state.stale_thumbnails.update(keys)  # as push_external_history leaves them
        self.controller.rotate_thumbnails(keys, 1, before)
        return dict(zip(("other", "third"), keys))

    def test_a_batch_turn_carries_a_current_fingerprint_to_the_turned_settings(self) -> None:
        self._save("other", self._current("other"))
        self._save("third", "some-older-render")
        base = WorkspaceConfig()

        keys = self._batch_turn(replace(base, geometry=replace(base.geometry, rotation=1)))

        assert self.controller.asset_store.get_thumbnail_fingerprint(keys["other"]) == self._current("other")
        assert keys["other"] not in self.controller.state.stale_thumbnails
        assert self.controller.asset_store.get_thumbnail_fingerprint(keys["third"]) is None
        assert keys["third"] in self.controller.state.stale_thumbnails

    def test_a_batch_turn_does_not_vouch_for_a_keystoned_frame(self) -> None:
        base = WorkspaceConfig()
        keystoned = replace(base, geometry=replace(base.geometry, converge_v=4.0))
        self.session.config_for_asset.return_value = keystoned
        self._save("other", self._current("other"))

        keys = self._batch_turn(replace(keystoned, geometry=replace(keystoned.geometry, rotation=1)))

        assert self.controller.asset_store.get_thumbnail_fingerprint(keys["other"]) is None

    # Seeding the Film Strip's stale dot from fingerprints when a roll opens.

    def _seed(self, assets=None) -> set:
        from negpy.services.assets.thumbnails import asset_thumbnail_key

        with patch("negpy.desktop.controller.QTimer.singleShot", side_effect=lambda _ms, fn: fn()):
            self.controller._seed_stale_thumbnails(assets if assets is not None else self.files, restart=True)
        keys = {asset_thumbnail_key(f): f["hash"] for f in self.files}
        return {keys[k] for k in self.controller.state.stale_thumbnails if k in keys}

    def test_seeding_flags_only_a_real_mismatch(self) -> None:
        from negpy.services.assets.thumbnail_fingerprint import QUICK

        self._save("other", "some-older-render")
        self._save("third", self._current("third"))
        assert self._seed() == {"other"}

        self._save("other", QUICK)
        self._save("third", None)
        self.controller.state.stale_thumbnails.clear()
        assert self._seed() == set()

    def test_seeding_skips_the_active_frame_and_diptych_rows(self) -> None:
        self._save("active", "some-older-render")
        self._save("other", "some-older-render")
        self.files[1]["diptych"] = True
        with patch.object(
            self.controller, "diptych_pair", side_effect=lambda f: (WorkspaceConfig(), WorkspaceConfig()) if f["diptych"] else None
        ):
            assert self._seed() == set()

    def test_seeding_keeps_this_sessions_flags(self) -> None:
        from negpy.services.assets.thumbnails import asset_thumbnail_key

        self._save("other", self._current("other"))
        self.controller.state.stale_thumbnails.add(asset_thumbnail_key(self.files[1]))
        assert self._seed() == {"other"}

    def test_seeding_works_through_a_roll_larger_than_one_chunk(self) -> None:
        from negpy.desktop.controller import _STALE_SEED_CHUNK

        self.files.extend(
            {"name": f"f{i}.dng", "path": f"/roll/f{i}.dng", "hash": f"f{i}", "diptych": False} for i in range(_STALE_SEED_CHUNK * 2)
        )
        self._save(f"f{_STALE_SEED_CHUNK * 2 - 1}", "some-older-render")
        assert self._seed() == {f"f{_STALE_SEED_CHUNK * 2 - 1}"}
        assert self.controller._stale_seed_pending == []

    def test_seeding_ignores_a_frame_no_longer_loaded(self) -> None:
        self._save("other", "some-older-render")
        gone = self.files.pop(1)
        assert self._seed([gone]) == set()


def test_progress_text_has_no_time_left_after_one_frame() -> None:
    assert thumbnail_refresh_progress_text(1, 49, 1.0, 1.0) == "Thumbnails 1/49"


def test_progress_text_formats_seconds_and_minutes() -> None:
    assert thumbnail_refresh_progress_text(2, 5, 1.0, 1.0) == "Thumbnails 2/5 · ~6 s left"
    assert thumbnail_refresh_progress_text(2, 49, 1.0, 1.0) == "Thumbnails 2/49 · ~2 min left"


def test_progress_text_has_no_time_left_on_the_last_frame() -> None:
    assert thumbnail_refresh_progress_text(5, 5, 1.0, 1.0) == "Thumbnails 5/5"


def test_progress_text_flags_a_read_bound_refresh_only() -> None:
    assert thumbnail_refresh_progress_text(2, 49, 17.0, 1.0) == "Thumbnails 2/49 · ~14 min left · reading 17 s/frame"
    assert "reading" not in thumbnail_refresh_progress_text(2, 49, 4.0, 3.0)
    assert "reading" not in thumbnail_refresh_progress_text(2, 49, 2.5, 0.1)
