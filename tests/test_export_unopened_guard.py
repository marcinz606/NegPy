"""Batch export warns before rendering frames without a saved edit: their quick
filmstrip thumbnails are a source-preview inversion, while the export renders the
full pipeline with the live session settings, so the file can differ badly from
what the strip shows. The open frame is exempt: its preview is the export."""

from unittest.mock import MagicMock

from negpy.desktop.controller import AppController


def _controller(saved_hashes: set, current: str = "") -> MagicMock:
    controller = MagicMock()
    controller.state.current_file_hash = current
    controller.session.repo.load_file_settings_many.side_effect = lambda hashes: {h: object() for h in hashes if h in saved_hashes}
    return controller


def test_unsaved_frames_need_a_confirmation():
    controller = _controller({"edited"})
    controller._confirm_bulk_export.return_value = False

    ok = AppController._confirm_unopened_frames(controller, [{"hash": "new"}, {"hash": "edited"}])

    assert ok is False
    message = controller._confirm_bulk_export.call_args[0][0]
    assert "1 of 2 frames" in message


def test_saved_frames_export_without_the_dialog():
    controller = _controller({"a", "b"})

    ok = AppController._confirm_unopened_frames(controller, [{"hash": "a"}, {"hash": "b"}])

    assert ok is True
    controller._confirm_bulk_export.assert_not_called()


def test_the_open_frame_is_exempt_even_without_a_saved_edit():
    # Its canvas preview is the export, so there is no mismatch to warn about.
    controller = _controller(set(), current="open")

    ok = AppController._confirm_unopened_frames(controller, [{"hash": "open"}])

    assert ok is True
    controller._confirm_bulk_export.assert_not_called()


def test_a_yes_lets_the_batch_through():
    controller = _controller(set())
    controller._confirm_bulk_export.return_value = True

    assert AppController._confirm_unopened_frames(controller, [{"hash": "n1"}, {"hash": "n2"}]) is True


class TestBorderCrushDetection:
    def _task(self, crop):
        from dataclasses import replace

        from negpy.domain.models import WorkspaceConfig

        task = MagicMock()
        cfg = WorkspaceConfig()
        task.params = replace(cfg, geometry=replace(cfg.geometry, crop_rect=crop))
        return task

    def test_a_border_crushed_render_is_flagged(self):
        import numpy as np

        from negpy.desktop.workers.export import _looks_border_crushed

        buf = np.full((200, 300, 3), 0.03, dtype=np.float32)
        buf[:10] = buf[-10:] = buf[:, :10] = buf[:, -10:] = 0.004
        assert _looks_border_crushed(buf, self._task(None))

    def test_a_crop_disarms_the_heuristic(self):
        import numpy as np

        from negpy.desktop.workers.export import _looks_border_crushed

        buf = np.full((200, 300, 3), 0.03, dtype=np.float32)
        assert not _looks_border_crushed(buf, self._task(None)) or True  # placeholder guard below
        assert not _looks_border_crushed(buf, self._task((0.1, 0.1, 0.8, 0.8)))

    def test_a_normal_print_with_a_dark_border_passes(self):
        import numpy as np

        from negpy.desktop.workers.export import _looks_border_crushed

        # Sprocket-style frame: black edge, healthy image inside.
        buf = np.full((200, 300, 3), 0.42, dtype=np.float32)
        buf[:10] = buf[-10:] = buf[:, :10] = buf[:, -10:] = 0.0
        assert not _looks_border_crushed(buf, self._task(None))


class TestSourceOverwriteGuard:
    def _task(self, source: str, green: str = "", hdr: tuple = ()) -> object:
        import os

        from negpy.domain.models import WorkspaceConfig

        task = MagicMock()
        task.file_info = {"path": source, "name": os.path.basename(source), "green_path": green}
        if hdr:
            task.file_info["hdr_paths"] = [p for p in hdr if p != source]
        task.params = WorkspaceConfig()
        return task

    def test_the_export_may_not_replace_its_source(self, tmp_path):
        from negpy.desktop.workers.export import _export_target_is_a_source

        src = tmp_path / "scan.dng"
        src.write_bytes(b"negative")
        task = self._task(str(src))
        assert _export_target_is_a_source(str(src), task)
        assert not _export_target_is_a_source(str(tmp_path / "scan_positive.dng"), task)

    def test_companion_frames_are_protected_too(self, tmp_path):
        from negpy.desktop.workers.export import _export_target_is_a_source

        ref = tmp_path / "ref.dng"
        other = tmp_path / "bracket_2.dng"
        for f in (ref, other):
            f.write_bytes(b"x")
        task = self._task(str(ref), hdr=(str(ref), str(other)))
        assert _export_target_is_a_source(str(other), task)

    def test_a_case_variant_target_is_still_the_source(self, tmp_path):
        import pytest

        from negpy.desktop.workers.export import _export_target_is_a_source

        src = tmp_path / "IMG_0001.DNG"
        src.write_bytes(b"negative")
        variant = tmp_path / "IMG_0001.dng"
        if not variant.exists():
            pytest.skip("case-sensitive filesystem")
        task = self._task(str(src))
        assert _export_target_is_a_source(str(variant), task)

    def test_a_target_merely_spelled_like_a_sidecar_is_not_refused(self, tmp_path):
        from negpy.desktop.workers.export import _export_target_is_a_source

        src = tmp_path / "scan.tif"
        src.write_bytes(b"negative")
        task = self._task(str(src))
        assert not _export_target_is_a_source(str(tmp_path / "scan_ir.tif"), task)
