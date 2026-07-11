"""Regression test for the capture→import seam (AppController._on_capture_finished).

A finished capture must set the `rgbscan_mode` global correctly — on for an R/G/B
triplet (so NegPy merges it), off for a single frame — and hand the paths to asset
discovery. This guards that seam against an upstream rename of `rgbscan_mode` /
`request_asset_discovery`: it fails in a fast unit test instead of only showing up
as a gray frame at a real hardware scan.

Calls the capture and load seams against a mock controller (no full AppController /
GPU needed), with an AppState standing in for session hydration.
"""

import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.features.process.models import ProcessMode


def _run(paths, **req_kw):
    """Invoke _on_capture_finished on a mock controller with a fake capture request."""
    fields = {"white_mode": False, "rgb_mode": True, "white_process_mode": "auto", **req_kw}
    req = SimpleNamespace(**fields)
    controller = MagicMock()
    controller.state = AppState()
    controller._pending_capture_imports = {}
    controller._last_capture_req = req
    AppController._on_capture_finished(controller, paths)
    return controller


def _hydrate_and_load(controller, path, process_mode, *, autodetect=True):
    """Model the session selection step, then enter the controller's public load seam."""
    controller.state.config = replace(
        controller.state.config,
        process=replace(controller.state.config.process, process_mode=process_mode),
    )
    controller.state.current_file_is_new = True
    controller.state.autodetect_enabled = autodetect
    controller.state.hq_preview = False
    controller.state.workspace_color_space = "Adobe RGB"
    controller.state.current_file_path = path
    controller._prefetch_gen = 0
    controller._file_hash_for_path.return_value = None

    AppController.load_file(controller, path)
    return controller.preview_load_requested.emit.call_args.args[0]


def test_rgb_triplet_enables_merge_and_discovers():
    c = _run(["r.ARW", "g.ARW", "b.ARW"], rgb_mode=True, white_mode=False)
    c.session.repo.save_global_setting.assert_any_call("rgbscan_mode", True)  # triplet → merge ON
    c.request_asset_discovery.assert_called_once_with(["r.ARW", "g.ARW", "b.ARW"])
    assert c._pending_scanned_file == "r.ARW"  # red is primary → auto-selected after discovery


def test_rgb_triplet_import_defaults_to_c41_without_autodetect():
    c = _run(["r.ARW", "g.ARW", "b.ARW"], rgb_mode=True, white_mode=False)

    task = _hydrate_and_load(c, "r.ARW", ProcessMode.E6, autodetect=True)

    assert c.state.config.process.process_mode == ProcessMode.C41
    assert task.detect_mode is False


def test_normal_single_scan_leaves_merge_off():
    c = _run(["frame.ARW"], rgb_mode=False)
    c.session.repo.save_global_setting.assert_any_call("rgbscan_mode", False)  # single RAW → no merge
    c.request_asset_discovery.assert_called_once_with(["frame.ARW"])


def test_white_slide_leaves_merge_off():
    c = _run(["slide.ARW"], rgb_mode=True, white_mode=True, white_process_mode="auto")
    c.session.repo.save_global_setting.assert_any_call("rgbscan_mode", False)  # one white exposure → no merge
    c.request_asset_discovery.assert_called_once_with(["slide.ARW"])


def test_explicit_e6_applies_to_import_after_hydration_without_detection():
    c = _run(["slide.ARW"], rgb_mode=True, white_mode=True, white_process_mode="E-6")

    # Capture completion must not mutate whichever asset happened to be open before import.
    assert c.state.config.process.process_mode == ProcessMode.C41

    task = _hydrate_and_load(c, "slide.ARW", ProcessMode.C41, autodetect=True)

    assert c.state.config.process.process_mode == ProcessMode.E6
    assert task.detect_mode is False


def test_explicit_bw_applies_to_import_after_hydration_without_detection():
    c = _run(["mono.ARW"], rgb_mode=True, white_mode=True, white_process_mode="B&W")

    assert c.state.config.process.process_mode == ProcessMode.C41

    task = _hydrate_and_load(c, "mono.ARW", ProcessMode.E6, autodetect=True)

    assert c.state.config.process.process_mode == ProcessMode.BW
    assert task.detect_mode is False


def test_automatic_white_import_requests_detection_without_forcing_mode():
    c = _run(["auto.ARW"], rgb_mode=True, white_mode=True, white_process_mode="auto")

    task = _hydrate_and_load(c, "auto.ARW", ProcessMode.E6, autodetect=False)

    assert c.state.config.process.process_mode == ProcessMode.E6
    assert task.detect_mode is True


def test_failed_discovery_discards_capture_intent():
    c = _run(["missing.ARW"], rgb_mode=True, white_mode=True, white_process_mode="E-6")
    c._auto_open_after_discovery = False
    c._replace_after_discovery = False
    c._reselect_after_discovery = None
    c._active_discovery_keys = frozenset({os.path.normcase(os.path.abspath("missing.ARW"))})

    AppController._on_discovery_finished(c, [])

    assert c._pending_capture_imports == {}
    assert c._pending_scanned_file is None


def test_capture_intent_is_scoped_to_captured_primary_path():
    c = _run(["slide.ARW"], rgb_mode=True, white_mode=True, white_process_mode="E-6")

    unrelated_task = _hydrate_and_load(c, "other.ARW", ProcessMode.BW, autodetect=True)

    assert c.state.config.process.process_mode == ProcessMode.BW
    assert unrelated_task.detect_mode is True
    assert len(c._pending_capture_imports) == 1

    captured_task = _hydrate_and_load(c, "slide.ARW", ProcessMode.C41, autodetect=True)
    assert c.state.config.process.process_mode == ProcessMode.E6
    assert captured_task.detect_mode is False
    assert c._pending_capture_imports == {}


def test_empty_paths_is_a_noop():
    c = _run([])
    c.request_asset_discovery.assert_not_called()  # nothing captured → no discovery
