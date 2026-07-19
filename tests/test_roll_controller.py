"""Regression test for the roll-scan→import seam (AppController._on_roll_scan_finished).

Mirrors test_capture_controller.py's shape: call the seam directly against a mock
controller, no real worker/thread/session involved. RollFrameOutput's shape (slot,
rgb_path, ir_path, receipt_path) is stood in with SimpleNamespace -- this seam only
reads .rgb_path, so nothing coolscanpy-shaped needs to be constructed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController


def _output(slot: int, rgb_path: str):
    return SimpleNamespace(slot=slot, rgb_path=rgb_path, ir_path=None, receipt_path=f"{rgb_path}_receipt.json")


def test_writes_are_discovered_in_batch_order():
    controller = MagicMock()
    outputs = [_output(1, "a.tif"), _output(2, "b.tif"), _output(3, "c.tif")]

    AppController._on_roll_scan_finished(controller, outputs)

    controller.roll_finished.emit.assert_called_once_with(outputs)
    controller.request_asset_discovery.assert_called_once_with(["a.tif", "b.tif", "c.tif"])


def test_empty_batch_still_emits_but_does_not_request_discovery():
    controller = MagicMock()

    AppController._on_roll_scan_finished(controller, [])

    controller.roll_finished.emit.assert_called_once_with([])
    controller.request_asset_discovery.assert_not_called()


def test_ir_and_receipt_paths_are_not_sent_to_discovery():
    """Only the RGB TIFF is a NegPy-openable asset; the _IR sidecar and the JSON
    receipt are not scannable frames and must not be handed to discovery."""
    controller = MagicMock()
    outputs = [SimpleNamespace(slot=1, rgb_path="frame.tif", ir_path="frame_IR.tif", receipt_path="frame_receipt.json")]

    AppController._on_roll_scan_finished(controller, outputs)

    (discovered,), _kwargs = controller.request_asset_discovery.call_args
    assert discovered == ["frame.tif"]
