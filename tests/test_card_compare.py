"""A held card eye shows the frame without that card, then gives the compare view back."""

from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _controller(**state) -> MagicMock:
    ctrl = MagicMock()
    ctrl.state = AppState(preview_raw=np.zeros((2, 2, 3), np.float32), **state)
    ctrl._card_compare_saved = None
    ctrl._compare_before_key.return_value = "k"
    for name in ("begin_compare_hold", "end_compare_hold", "_restore_card_compare", "_drop_compare_before", "exit_compare"):
        setattr(ctrl, name, getattr(AppController, name).__get__(ctrl))
    return ctrl


def test_holding_a_card_borrows_the_reference_and_gives_it_back():
    step = replace(DEFAULT_WORKSPACE_CONFIG)
    ctrl = _controller(compare_reference=step, compare_reference_label="STEP 2")
    card_off = replace(DEFAULT_WORKSPACE_CONFIG)

    AppController.begin_card_compare(ctrl, card_off, "WITHOUT TONE")
    assert ctrl.state.compare_hold is True
    assert ctrl.state.compare_reference is card_off
    assert ctrl.state.compare_reference_label == "WITHOUT TONE"
    ctrl._request_compare_baseline.assert_called_once()

    ctrl.state.compare_before = np.zeros((2, 2, 3), np.float32)
    AppController.end_card_compare(ctrl)
    assert ctrl.state.compare_hold is False
    assert ctrl.state.compare_reference is step
    assert ctrl.state.compare_reference_label == "STEP 2"
    assert ctrl.state.compare_before is None


def test_a_refused_hold_leaves_the_reference_alone():
    ctrl = _controller(negative_peek=True)
    AppController.begin_card_compare(ctrl, replace(DEFAULT_WORKSPACE_CONFIG), "WITHOUT TONE")
    assert ctrl.state.compare_hold is False
    assert ctrl.state.compare_reference is None
    assert ctrl._card_compare_saved is None
    AppController.end_card_compare(ctrl)
    assert ctrl.state.compare_reference is None


def test_ending_a_card_hold_in_the_split_renders_its_baseline_again():
    ctrl = _controller(compare_mode=True)
    AppController.begin_card_compare(ctrl, replace(DEFAULT_WORKSPACE_CONFIG), "WITHOUT TONE")
    ctrl._request_compare_baseline.reset_mock()
    AppController.end_card_compare(ctrl)
    ctrl._request_compare_baseline.assert_called_once()


def test_a_card_eye_turns_off_its_own_card_alone(qapp):
    from negpy.desktop.view.sidebar.controls_panel import ControlsPanel

    controller = MagicMock()
    controller.state = AppState()
    controller.active_diptych.return_value = None
    panel = ControlsPanel(controller)
    cfg = DEFAULT_WORKSPACE_CONFIG
    controller.state.config = replace(
        cfg,
        exposure=replace(cfg.exposure, density=cfg.exposure.density + 0.3, grade=cfg.exposure.grade + 1.0),
        lab=replace(cfg.lab, sharpen=cfg.lab.sharpen + 0.5),
    )
    edit = controller.state.config

    tone_off = panel._card_off_config("tone")
    assert tone_off.lab == edit.lab
    assert (tone_off.exposure.density, tone_off.exposure.grade) == (cfg.exposure.density, cfg.exposure.grade)

    lab_off = panel._card_off_config("lab")
    assert lab_off.lab == cfg.lab
    assert lab_off.exposure == edit.exposure

    eye = panel.lab_section.compare_btn
    eye.pressed.emit()
    controller.begin_card_compare.assert_called_once()
    assert controller.begin_card_compare.call_args.args[0].lab == cfg.lab
    eye.released.emit()
    controller.end_card_compare.assert_called_once()


def test_gpu_card_hold_detaches_with_hold_lowered_before_destroying_vram_texture():
    from unittest.mock import patch

    class _FakeGPUTex:
        def __init__(self):
            self.destroyed = False

        def destroy(self):
            self.destroyed = True

    step = replace(DEFAULT_WORKSPACE_CONFIG, lab=replace(DEFAULT_WORKSPACE_CONFIG.lab, sharpen=0.4))
    card_off = replace(DEFAULT_WORKSPACE_CONFIG)
    ctrl = _controller(compare_mode=True, compare_reference=step, compare_reference_label="STEP 2")

    with patch("negpy.desktop.controller.GPUTexture", _FakeGPUTex):
        split_tex = _FakeGPUTex()
        ctrl.state.compare_before = split_tex
        ctrl.state.compare_before_key = "old"

        AppController.begin_card_compare(ctrl, card_off, "WITHOUT TONE")
        assert split_tex.destroyed is True

        card_tex = _FakeGPUTex()
        ctrl.state.compare_before = card_tex
        ctrl.state.compare_before_rect = (0, 0, 100, 80)

        seen_on_detach: list[tuple[bool, object, bool, str]] = []
        ctrl.compare_frame_ready.emit.side_effect = lambda: seen_on_detach.append(
            (ctrl.state.compare_hold, ctrl.state.compare_before, card_tex.destroyed, ctrl.state.compare_reference_label)
        )

        AppController.end_card_compare(ctrl)

        # Hold is already False and compare_before is None when compare_frame_ready fires,
        # so the canvas neither flashes a mode-1 split of card_tex nor holds a destroyed view.
        assert seen_on_detach == [(False, None, False, "STEP 2")]
        assert card_tex.destroyed is True
        assert ctrl.state.compare_before is None
        assert ctrl.state.compare_before_rect is None


def test_exit_compare_during_card_hold_clears_borrowed_reference():
    step = replace(DEFAULT_WORKSPACE_CONFIG)
    ctrl = _controller(compare_mode=True, compare_reference=step, compare_reference_label="STEP 2")

    AppController.begin_card_compare(ctrl, replace(DEFAULT_WORKSPACE_CONFIG), "WITHOUT TONE")
    assert ctrl._card_compare_saved is not None

    ctrl.exit_compare()
    assert ctrl._card_compare_saved is None
    assert ctrl.state.compare_reference is None

    AppController.end_card_compare(ctrl)
    assert ctrl.state.compare_reference is None
    assert ctrl.state.compare_reference_label == ""


def test_superseded_card_compare_render_is_not_captured_into_restored_split():
    from unittest.mock import patch

    class _FakeGPUTex:
        def __init__(self):
            self.width, self.height = 100, 80
            self.clones = 0

        def clone_gpu(self):
            self.clones += 1
            return _FakeGPUTex()

    ctrl = _controller(compare_mode=True, gpu_enabled=True, gpu_viewport_failed="")
    ctrl._is_rendering = True
    ctrl._pending_render_task = None
    ctrl._renders_another_frame.return_value = False
    ctrl._compare_before_key.return_value = "split_baseline_key"
    setattr(ctrl, "_capture_compare_before", AppController._capture_compare_before.__get__(ctrl))

    stale_pool = _FakeGPUTex()
    with patch("negpy.desktop.controller.GPUTexture", _FakeGPUTex):
        AppController._on_render_finished(
            ctrl,
            None,
            {"base_positive": stale_pool, "compare": True, "compare_key": "card_off_key"},
        )

    assert stale_pool.clones == 0
    assert ctrl.state.compare_before is None
    ctrl.compare_before_staged.emit.assert_called_once_with(True)
    ctrl._request_compare_baseline.assert_called_once()


def test_gpu_widget_redraws_after_silent_detach_clears_active_split(qapp):
    from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget

    class _FakeTex:
        width, height = 100, 80
        view = object()

    widget = GPUCanvasWidget()

    draws: list[object] = []
    widget.device = object()
    widget.context = object()
    widget.canvas = MagicMock()
    widget.canvas.request_draw.side_effect = draws.append

    widget.set_compare_state(_FakeTex(), None, None, 0.5, 1, redraw=True)
    assert len(draws) == 1
    assert widget._drawn_compare_mode == 1

    # Silent detach (redraw=False) drops the texture view without drawing yet.
    widget.set_compare_state(None, None, None, 0.5, 0, redraw=False)
    assert len(draws) == 1
    assert widget.before_texture_view is None
    assert widget._drawn_compare_mode == 1

    # Subsequent redraw=True still repaints to clear the split off the GPU surface.
    widget.set_compare_state(None, None, None, 0.5, 0, redraw=True)
    assert len(draws) == 2
    assert widget._drawn_compare_mode == 0


def test_gpu_hold_paints_card_badge_without_texture_readback(qapp):
    from unittest.mock import patch
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QPainter, QPixmap
    from PyQt6.QtWidgets import QWidget
    from negpy.desktop.view.canvas.overlay import CanvasOverlay

    class _FakeGPU(QWidget):
        def isVisible(self):
            return True

    class _FakeParent(QWidget):
        def __init__(self):
            super().__init__()
            self.gpu_widget = _FakeGPU(self)
            self._compare_awaiting_edit = False

    class _FakeTex:
        width, height = 200, 200

        def readback(self):
            raise AssertionError("GPU card hold must not read back baseline texture")

    parent = _FakeParent()
    state = AppState(compare_hold=True, compare_reference_label="WITHOUT TONE")
    overlay = CanvasOverlay(state)
    overlay.setParent(parent)
    overlay.resize(200, 200)
    overlay._view_rect = QRectF(0, 0, 200, 200)
    overlay._current_size = (200, 200)

    with patch("negpy.desktop.view.canvas.overlay.GPUTexture", _FakeTex):
        state.compare_before = _FakeTex()
        pm = QPixmap(200, 200)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        overlay._draw_ui(painter)
        painter.end()

    img = pm.toImage()
    badge_pixels = sum(1 for y in range(12, 34) for x in range(12, 160) if img.pixelColor(x, y).alpha() > 0)
    assert badge_pixels > 100


def test_backslash_hold_and_tap_while_card_eye_held_never_leak_card_into_split():
    step = replace(DEFAULT_WORKSPACE_CONFIG, lab=replace(DEFAULT_WORKSPACE_CONFIG.lab, sharpen=0.4))
    card_off = replace(DEFAULT_WORKSPACE_CONFIG)
    ctrl = _controller(compare_mode=True, compare_reference=step, compare_reference_label="STEP 2")
    setattr(ctrl, "begin_card_compare", AppController.begin_card_compare.__get__(ctrl))
    setattr(ctrl, "end_card_compare", AppController.end_card_compare.__get__(ctrl))
    setattr(ctrl, "toggle_compare", AppController.toggle_compare.__get__(ctrl))

    ctrl.begin_card_compare(card_off, "WITHOUT TONE")
    assert ctrl.state.compare_hold is True

    # Releasing keyboard `\` while the card eye is still held keeps the card hold active.
    ctrl.end_compare_hold()
    assert ctrl.state.compare_hold is True
    assert ctrl.state.compare_reference is card_off

    # Tapping `\` while the card eye is held ends card compare cleanly before toggling split.
    ctrl2 = _controller(compare_mode=False)
    for name in ("begin_card_compare", "end_card_compare", "toggle_compare"):
        setattr(ctrl2, name, getattr(AppController, name).__get__(ctrl2))
    ctrl2.begin_card_compare(card_off, "WITHOUT TONE")
    ctrl2.toggle_compare()
    assert ctrl2.state.compare_mode is True
    assert ctrl2.state.compare_hold is False
    assert ctrl2.state.compare_reference is None
    assert ctrl2.state.compare_reference_label == ""


def test_preserve_gpu_edit_keeps_canvas_live_across_compare_and_card_release(qapp):
    from unittest.mock import patch
    from negpy.desktop.view.canvas.widget import ImageCanvas

    class _FakeTex:
        def __init__(self, val: float = 0.5):
            self.width, self.height = 100, 80
            self.view = object()
            self.val = val
            self.destroyed = False

        def clone_gpu(self):
            return _FakeTex(self.val)

        def readback_region(self, *_a):
            return np.full((1, 1, 4), self.val, dtype=np.float32)

        def destroy(self):
            self.destroyed = True

    state = AppState(gpu_enabled=True, gpu_viewport_failed="")
    with patch("negpy.desktop.view.canvas.widget.GPUDevice") as mock_dev:
        mock_dev.get.return_value.is_available = False
        canvas = ImageCanvas(state)

    modes: list[tuple[int, bool]] = []
    canvas.gpu_widget.set_compare_state = lambda _b, _br, _ar, _s, mode, *, redraw=True: modes.append((mode, redraw))

    with (
        patch("negpy.desktop.view.canvas.widget.GPUTexture", _FakeTex),
        patch("negpy.desktop.view.canvas.overlay.GPUTexture", _FakeTex),
    ):
        pool_tex = _FakeTex(0.75)
        canvas.update_buffer(pool_tex, "sRGB", (0, 0, 100, 80))
        canvas.preserve_gpu_edit()
        preserved = canvas._preserved_edit_tex
        assert isinstance(preserved, _FakeTex)
        assert preserved is not pool_tex
        assert canvas.gpu_widget.current_texture_view is preserved.view

        # Pool is overwritten by compare=True, but preserved edit lets split & probe stay live immediately.
        pool_tex.val = 0.1
        state.compare_mode = True
        state.compare_before = _FakeTex(0.1)
        canvas.stage_compare_before(True)
        assert canvas._compare_awaiting_edit is False
        assert modes[-1] == (1, True)
        assert canvas.get_pixel_rgb(0.5, 0.5) == (0.75, 0.75, 0.75)

        # Next update_buffer swaps in the fresh edit and destroys the preserved clone.
        canvas.update_buffer(_FakeTex(0.75), "sRGB", (0, 0, 100, 80))
        assert preserved.destroyed is True
        assert canvas._preserved_edit_tex is None

        # When pool is marked dirty/invalid (e.g. after flat_peek or TestStrip), preserve_gpu_edit
        # refuses to clone _last_buffer and sets _compare_awaiting_edit=True instead.
        canvas.preserve_gpu_edit(can_clone_pool=False, can_keep_preserved=False)
        assert canvas._preserved_edit_tex is None
        assert canvas._compare_awaiting_edit is True

        # When an existing preserved texture becomes stale across an edit before update_buffer,
        # current_texture_view stays non-None (preventing black swapchain clears) while awaiting edit.
        canvas.update_buffer(_FakeTex(0.8), "sRGB", (0, 0, 100, 80))
        canvas.preserve_gpu_edit(can_clone_pool=True, can_keep_preserved=True)
        stale_preserved = canvas._preserved_edit_tex
        assert stale_preserved is not None
        canvas.preserve_gpu_edit(can_clone_pool=False, can_keep_preserved=False)
        assert canvas._compare_awaiting_edit is True
        assert canvas.gpu_widget.current_texture_view is stale_preserved.view
        assert stale_preserved.destroyed is False
        canvas.update_buffer(_FakeTex(0.9), "sRGB", (0, 0, 100, 80))
        assert stale_preserved.destroyed is True

        # Switching to CPU update_buffer (e.g. negative peek or disabling GPU) detaches current_texture_view
        # BEFORE destroying _preserved_edit_tex.
        canvas.preserve_gpu_edit(can_clone_pool=True, can_keep_preserved=True)
        cpu_drop = canvas._preserved_edit_tex
        assert cpu_drop is not None
        canvas.update_buffer(np.zeros((80, 100, 3), dtype=np.float32), "sRGB", None)
        assert cpu_drop.destroyed is True
        assert canvas.gpu_widget.current_texture_view is None


def test_gpu_fallback_during_compare_releases_dead_pool_textures_and_notifies():
    from unittest.mock import patch

    class _FakeGPUTex:
        def __init__(self):
            self.destroyed = False

        def destroy(self):
            self.destroyed = True

    ctrl = _controller(compare_hold=True, gpu_enabled=False, gpu_viewport_failed="err")
    ctrl._gpu_fallback_notified = False
    ctrl._pending_render_task = None
    ctrl._in_flight_compare_key = "k"
    ctrl._last_render_identity = ("h", "m", None)
    ctrl._renders_another_frame.return_value = False
    setattr(ctrl, "_capture_compare_before", AppController._capture_compare_before.__get__(ctrl))

    old_gpu_before = _FakeGPUTex()
    ctrl.state.compare_before = old_gpu_before
    ctrl.state.last_metrics["base_positive"] = _FakeGPUTex()
    ctrl.state.last_metrics["normalized_log"] = _FakeGPUTex()
    ctrl.state.last_metrics["render_identity"] = ("h", ctrl.state.config)

    with patch("negpy.desktop.controller.GPUTexture", _FakeGPUTex):
        AppController._on_render_finished(
            ctrl,
            None,
            {
                "base_positive": np.zeros((4, 4, 3), np.float32),
                "compare": True,
                "gpu_fallback": True,
            },
        )

    assert old_gpu_before.destroyed is True
    assert ctrl._last_render_identity is None
    assert "base_positive" not in ctrl.state.last_metrics
    assert "normalized_log" not in ctrl.state.last_metrics
    assert "render_identity" not in ctrl.state.last_metrics
    ctrl.gpu_textures_released.emit.assert_called_once()
    ctrl.set_status.assert_called_once()
    assert isinstance(ctrl.state.compare_before, np.ndarray)
    ctrl.request_render.assert_called_once_with(keep_overlays=True)


def test_compare_recaptures_during_standalone_hold_and_preserves_zone_pins_and_expected_key():
    from types import SimpleNamespace
    from unittest.mock import patch

    with patch("negpy.desktop.controller.PreviewManager") as mock_pm:
        mock_pm.return_value.load_linear_preview.return_value = (None, (0, 0), {})
        from negpy.desktop.session import DesktopSessionManager

        sm = MagicMock(spec=DesktopSessionManager)
        sm.state = AppState(preview_raw=np.zeros((8, 8, 3), np.float32))
        sm.repo = MagicMock()
        sm.asset_model = MagicMock()
        ctrl = AppController(sm)
    try:
        tasks = []
        ctrl.render_requested.connect(tasks.append)

        # average_peek blocks begin_compare_hold and is cleared by toggle_compare.
        ctrl.state.average_peek = True
        ctrl.begin_compare_hold()
        assert ctrl.state.compare_hold is False
        ctrl.toggle_compare()
        assert ctrl.state.average_peek is False
        assert ctrl.state.compare_mode is True

        # Zone pins survive post-compare edit restoration when state is unchanged,
        # and are dropped if a real edit happened while compare=True was in flight.
        ctrl.exit_compare()
        ctrl.request_render()
        ctrl._is_rendering = False
        ctrl.state.zone_pins = [(0.5, 0.5, 5)]
        ctrl.toggle_compare()
        ctrl._is_rendering = False
        ctrl._on_render_finished(None, {"base_positive": np.zeros((4, 4, 3), np.float32), "compare": True})
        assert ctrl.state.zone_pins == [(0.5, 0.5, 5)]

        ctrl.state.config = replace(ctrl.state.config, exposure=replace(ctrl.state.config.exposure, density=1.5))
        ctrl._is_rendering = False
        ctrl._pending_render_task = None
        ctrl._on_render_finished(None, {"base_positive": np.zeros((4, 4, 3), np.float32), "compare": True})
        assert ctrl.state.zone_pins == []

        # compare_capture during an uncropped preview tool still renders cropped full-res,
        # and never overwrites a pending plain edit render in _pending_render_task.
        from negpy.desktop.session import ToolMode

        ctrl.state.active_tool = ToolMode.CROP_MANUAL
        ctrl._is_rendering = False
        ctrl._pending_render_task = None
        tasks.clear()
        ctrl._request_compare_baseline()
        assert tasks[-1].crop_preview_full is False
        ctrl.state.active_tool = ToolMode.NONE

        ctrl._is_rendering = True
        ctrl.request_render()  # queues plain edit in _pending_render_task
        queued_edit = ctrl._pending_render_task
        assert queued_edit is not None and queued_edit.compare is False
        ctrl._request_compare_baseline()
        assert ctrl._pending_render_task is queued_edit
        ctrl._pending_render_task = None
        ctrl._is_rendering = False

        # _request_compare_baseline does not clobber _expected_render_key, and clears _last_render_identity
        # at GPU compare dispatch so an errored compare render cannot file a dirty pool texture in _render_memo.
        ctrl._expected_render_key = "settled_edit_key"
        ctrl._last_render_identity = ("h1", "m1", None)
        ctrl.state.gpu_enabled = True
        ctrl._request_compare_baseline()
        assert ctrl._expected_render_key == "settled_edit_key"
        assert ctrl._last_render_identity is None

        # Standalone compare_hold recaptures when geometry invalidates compare_before_key.
        ctrl.exit_compare()
        ctrl.begin_compare_hold()
        ctrl._is_rendering = False
        ctrl._on_render_finished(None, {"base_positive": np.zeros((4, 4, 3), np.float32), "compare": True})
        ctrl.state.config = replace(ctrl.state.config, geometry=replace(ctrl.state.config.geometry, rotation=90))
        tasks.clear()
        ctrl._is_rendering = False
        ctrl._pending_render_task = None
        ctrl._on_render_finished(None, {"base_positive": np.zeros((4, 4, 3), np.float32), "compare": False})
        assert tasks and tasks[-1].compare is True

        # Bounds writeback in _on_metrics_updated rekeys _in_flight_compare_key so first baseline render matches.
        ctrl.state.current_file_hash = "h1"
        sm.update_config.side_effect = lambda cfg, **_kw: setattr(ctrl.state, "config", cfg)
        ctrl._in_flight_compare_key = ctrl._compare_before_key()
        bounds = SimpleNamespace(floors=(0.1, 0.1, 0.1), ceils=(0.9, 0.9, 0.9))
        ctrl._on_metrics_updated({"source_hash": "h1", "log_bounds": bounds})
        assert ctrl._in_flight_compare_key == ctrl._compare_before_key()
    finally:
        for thread in (
            ctrl.render_thread,
            ctrl.export_thread,
            ctrl.thumb_thread,
            ctrl.norm_thread,
            ctrl.discovery_thread,
            ctrl.preview_load_thread,
            ctrl.prefetch_load_thread,
            ctrl.scan_thread,
        ):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()


def test_hiding_or_deactivating_held_card_eye_releases_hold_once(qapp):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QWidget
    from negpy.desktop.view.widgets.collapsible import CollapsibleSection

    host = QWidget()
    section = CollapsibleSection("Tone", parent=host)
    eye = section.add_compare_button("Hold")
    section.set_modified(1)
    host.show()
    qapp.processEvents()

    releases: list[int] = []
    eye.released.connect(lambda: releases.append(1))

    # Case 1: Undo resets modified count to 0 while eye is held down -> Hide triggers release once.
    eye.setDown(True)
    eye.pressed.emit()
    section.set_modified(0)
    assert len(releases) == 1
    assert eye.isDown() is False

    # Case 2: Hiding the ancestor sidebar/dock while eye is held down -> HideToParent triggers release once.
    section.set_modified(1)
    qapp.processEvents()
    eye.setDown(True)
    eye.pressed.emit()
    host.hide()
    assert len(releases) == 2
    assert eye.isDown() is False

    # Case 3: Application deactivation while eye is held down -> triggers release once.
    host.show()
    qapp.processEvents()
    eye.setDown(True)
    eye.pressed.emit()
    qapp.sendEvent(qapp, QEvent(QEvent.Type.ApplicationDeactivate))
    assert len(releases) == 3
    assert eye.isDown() is False

    # Case 4: Modal WindowBlocked or Popup menu Show (e.g. while Controls dock is floated) releases hold,
    # whereas a ToolTip Show does not.
    eye.setDown(True)
    eye.pressed.emit()
    qapp.sendEvent(eye, QEvent(QEvent.Type.WindowBlocked))
    assert len(releases) == 4
    assert eye.isDown() is False

    from PyQt6.QtCore import Qt

    tip = QWidget(None, Qt.WindowType.ToolTip)
    popup = QWidget(None, Qt.WindowType.Popup)
    eye.setDown(True)
    eye.pressed.emit()
    qapp.sendEvent(tip, QEvent(QEvent.Type.Show))
    assert len(releases) == 4
    qapp.sendEvent(popup, QEvent(QEvent.Type.Show))
    assert len(releases) == 5
    assert eye.isDown() is False
    host.close()


def test_gpu_hold_badge_tracks_before_geometry_while_awaiting_edit_and_resize_draws_mode2(qapp):
    from unittest.mock import patch
    from PyQt6.QtCore import QRectF
    from PyQt6.QtWidgets import QWidget
    from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget
    from negpy.desktop.view.canvas.overlay import CanvasOverlay

    class _FakeGPU(QWidget):
        def isVisible(self):
            return True

    class _FakeParent(QWidget):
        def __init__(self):
            super().__init__()
            self.gpu_widget = _FakeGPU(self)
            self._compare_awaiting_edit = True

    class _FakeTex:
        width, height = 800, 1000
        view = object()

    parent = _FakeParent()
    state = AppState(compare_hold=True, compare_reference_label="WITHOUT TONE")
    overlay = CanvasOverlay(state)
    overlay.setParent(parent)
    overlay.resize(1000, 1000)
    overlay.fit_margin = 12.0
    # Prior stale edit was landscape 1000x800 (or empty); GPU hold uses portrait 800x1000 before_tex.
    overlay._current_size = (1000, 800)
    overlay._recalc_view_rect()

    with patch("negpy.desktop.view.canvas.overlay.GPUTexture", _FakeTex):
        state.compare_before = _FakeTex()
        target = overlay._compare_hold_target_rect()
        expected = overlay._view_rect_for_size(800, 1000)
        assert target == expected
        assert target != QRectF(overlay._view_rect)

    # GPUCanvasWidget._perform_resize requests draw when Mode 2 before_texture_view is active even with no current_texture_view.
    gpu_widget = GPUCanvasWidget()
    gpu_widget.device = object()
    gpu_widget.context = object()
    gpu_widget._configure_context = lambda: None
    draws: list[object] = []
    gpu_widget.canvas = MagicMock()
    gpu_widget.canvas.request_draw.side_effect = draws.append
    gpu_widget.set_compare_state(_FakeTex(), None, None, 0.5, 2, redraw=False)
    assert gpu_widget.current_texture_view is None
    assert gpu_widget._has_drawable_texture() is True
    gpu_widget._perform_resize()
    assert len(draws) == 1

    # Mode 2 with _compare_awaiting_edit=True binds before_texture_view to binding 0 as well,
    # so border/mat pixels outside before_content_rect never sample a stale/dirty current_texture_view.
    gpu_widget.setParent(parent)
    stale_view = object()
    before_tex = _FakeTex()
    before_tex.view = object()
    gpu_widget.current_texture_view = stale_view
    gpu_widget.set_compare_state(before_tex, (20, 20, 760, 960), None, 0.5, 2, redraw=False)
    gpu_widget.render_pipeline = object()
    gpu_widget.bind_group_layout = object()
    gpu_widget.uniform_buffer = object()
    gpu_widget.lut_view = object()
    gpu_widget.lut_sampler = object()
    fake_device = MagicMock()
    bound_entries: list[list[dict]] = []
    fake_device.create_bind_group.side_effect = lambda **kw: bound_entries.append(kw["entries"]) or MagicMock()
    gpu_widget.device = fake_device
    fake_ctx = MagicMock()
    fake_ctx.get_current_texture.return_value.width = 1000
    fake_ctx.get_current_texture.return_value.height = 1000
    gpu_widget.context = fake_ctx
    gpu_widget._draw_frame()
    assert len(bound_entries) == 1
    assert bound_entries[0][0]["resource"] is before_tex.view
    assert bound_entries[0][4]["resource"] is before_tex.view
