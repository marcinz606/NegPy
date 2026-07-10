"""Regression test for the capture→import seam (AppController._on_capture_finished).

A finished capture must set the `rgbscan_mode` global correctly — on for an R/G/B
triplet (so NegPy merges it), off for a single frame — and hand the paths to asset
discovery. This guards that seam against an upstream rename of `rgbscan_mode` /
`request_asset_discovery`: it fails in a fast unit test instead of only showing up
as a gray frame at a real hardware scan.

Calls the method against a mock controller (no full AppController / GPU needed) —
none of the exercised paths touch `self.state.config`.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController


def _run(paths, **req_kw):
    """Invoke _on_capture_finished on a mock controller with a fake capture request."""
    fields = {"white_mode": False, "rgb_mode": True, "white_process_mode": "auto", **req_kw}
    req = SimpleNamespace(**fields)
    controller = MagicMock()
    controller._last_capture_req = req
    AppController._on_capture_finished(controller, paths)
    return controller


def test_rgb_triplet_enables_merge_and_discovers():
    c = _run(["r.ARW", "g.ARW", "b.ARW"], rgb_mode=True, white_mode=False)
    c.session.repo.save_global_setting.assert_any_call("rgbscan_mode", True)  # triplet → merge ON
    c.request_asset_discovery.assert_called_once_with(["r.ARW", "g.ARW", "b.ARW"])
    assert c._pending_scanned_file == "r.ARW"  # red is primary → auto-selected after discovery


def test_normal_single_scan_leaves_merge_off():
    c = _run(["frame.ARW"], rgb_mode=False)
    c.session.repo.save_global_setting.assert_any_call("rgbscan_mode", False)  # single RAW → no merge
    c.request_asset_discovery.assert_called_once_with(["frame.ARW"])


def test_white_slide_leaves_merge_off():
    c = _run(["slide.ARW"], rgb_mode=True, white_mode=True, white_process_mode="auto")
    c.session.repo.save_global_setting.assert_any_call("rgbscan_mode", False)  # one white exposure → no merge
    c.request_asset_discovery.assert_called_once_with(["slide.ARW"])


def test_empty_paths_is_a_noop():
    c = _run([])
    c.request_asset_discovery.assert_not_called()  # nothing captured → no discovery
