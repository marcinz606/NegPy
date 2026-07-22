"""Regression test for the roll-scan→import seam (AppController._on_roll_scan_finished).

Mirrors test_capture_controller.py's shape: call the seam directly against a mock
controller, no real worker/thread/session involved. RollFrameOutput's shape (slot,
rgb_path, ir_path, receipt_path) is stood in with SimpleNamespace -- this seam only
reads .rgb_path, so nothing coolscanpy-shaped needs to be constructed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.controller import AppController
from negpy.services.roll.service import RollScanningError


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


def test_outputs_with_no_rgb_path_are_skipped_for_discovery():
    """The roll sidebar's output-tier setting can leave Tier 1 (rgb_path)
    unwritten -- e.g. a positive-only scan -- in which case rgb_path is None.
    Those frames have nothing NegPy can open and must not reach discovery,
    but a frame that did write Tier 1 in the same batch still should."""
    controller = MagicMock()
    outputs = [
        SimpleNamespace(slot=1, rgb_path=None, ir_path=None, receipt_path="a_receipt.json"),
        SimpleNamespace(slot=2, rgb_path="b.tif", ir_path=None, receipt_path="b_receipt.json"),
    ]

    AppController._on_roll_scan_finished(controller, outputs)

    controller.roll_finished.emit.assert_called_once_with(outputs)
    controller.request_asset_discovery.assert_called_once_with(["b.tif"])


def test_batch_with_no_rgb_paths_at_all_does_not_request_discovery():
    controller = MagicMock()
    outputs = [SimpleNamespace(slot=1, rgb_path=None, ir_path=None, receipt_path="a_receipt.json")]

    AppController._on_roll_scan_finished(controller, outputs)

    controller.request_asset_discovery.assert_not_called()


def test_request_shutdown_blocks_when_scanner_close_is_unresolved():
    worker = MagicMock()
    worker.shutdown.side_effect = RuntimeError("ownership remains uncertain")
    controller = SimpleNamespace(
        _roll_shutdown_complete=False,
        _shutdown_block_reason=None,
        roll_worker=worker,
        roll_status=MagicMock(),
    )

    assert AppController.request_shutdown(controller) is False

    assert controller._roll_shutdown_complete is False
    assert controller._shutdown_block_reason == "ownership remains uncertain"
    controller.roll_status.emit.assert_called_once()


def test_request_shutdown_is_idempotent_after_verified_close():
    worker = MagicMock()
    controller = SimpleNamespace(
        _roll_shutdown_complete=False,
        _shutdown_block_reason="old failure",
        roll_worker=worker,
        roll_status=MagicMock(),
    )

    assert AppController.request_shutdown(controller) is True
    assert AppController.request_shutdown(controller) is True

    worker.shutdown.assert_called_once_with()
    assert controller._shutdown_block_reason is None


def test_cleanup_refuses_to_quit_worker_threads_before_scanner_close():
    roll_thread = MagicMock()
    controller = SimpleNamespace(
        _cleaned_up=False,
        _shutdown_block_reason="ownership remains uncertain",
        request_shutdown=lambda: False,
        roll_thread=roll_thread,
    )

    with pytest.raises(RollScanningError, match="ownership remains uncertain"):
        AppController.cleanup(controller)

    assert controller._cleaned_up is False
    roll_thread.quit.assert_not_called()
