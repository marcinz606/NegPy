"""
Guards PreviewLoadWorker._detect_mode after the use_camera_wb rename.

The worker previously read task.linear_raw and passed linear_raw=True to
load_linear_preview; both were renamed to use_camera_wb during the #210 merge.
A wrong reference here is a silent runtime crash (not a merge conflict), so these
tests pin the wiring:
  - camera-WB preview (use_camera_wb=True): a lean no-WB decode (decode_for_detection)
    before classifying, since the C41 orange mask is hidden by camera WB.
  - no-WB preview (use_camera_wb=False): classify the buffer we already have.
"""

from unittest.mock import MagicMock, patch

import numpy as np


def _task(use_camera_wb: bool):
    from negpy.desktop.workers.render import PreviewLoadTask

    return PreviewLoadTask(
        file_path="/fake/path.dng",
        workspace_color_space="Adobe RGB",
        use_camera_wb=use_camera_wb,
        detect_mode=True,
    )


def test_detect_mode_camera_wb_redecodes_no_wb(qapp):
    from negpy.desktop.workers.render import PreviewLoadWorker

    service = MagicMock()
    rescan = np.zeros((4, 4, 3), dtype=np.float32)
    service.decode_for_detection.return_value = rescan
    worker = PreviewLoadWorker(service)

    camera_wb_buf = np.ones((4, 4, 3), dtype=np.float32)
    with patch("negpy.features.process.logic.detect_process_mode", return_value="c41") as dpm:
        result = worker._detect_mode(_task(use_camera_wb=True), camera_wb_buf)

    assert result == "c41"
    # Camera WB hides the C41 mask → a lean no-WB decode is run for detection.
    service.decode_for_detection.assert_called_once_with("/fake/path.dng")
    service.load_linear_preview.assert_not_called()
    # Classified the freshly re-decoded no-WB buffer, not the camera-WB one.
    assert dpm.call_args[0][0] is rescan


def test_detect_mode_no_wb_uses_existing_buffer(qapp):
    from negpy.desktop.workers.render import PreviewLoadWorker

    service = MagicMock()
    worker = PreviewLoadWorker(service)

    no_wb_buf = np.ones((4, 4, 3), dtype=np.float32)
    with patch("negpy.features.process.logic.detect_process_mode", return_value="bw") as dpm:
        result = worker._detect_mode(_task(use_camera_wb=False), no_wb_buf)

    assert result == "bw"
    service.load_linear_preview.assert_not_called()
    assert dpm.call_args[0][0] is no_wb_buf


def test_rgb_import_with_detection_disabled_never_calls_classifier(qapp):
    from negpy.desktop.workers.render import PreviewLoadTask, PreviewLoadWorker

    service = MagicMock()
    raw = np.ones((4, 4, 3), dtype=np.float32)
    service.load_linear_preview_rgb.return_value = (raw, (4, 4), {})
    worker = PreviewLoadWorker(service)
    finished = []
    worker.finished.connect(lambda *args: finished.append(args))
    task = PreviewLoadTask(
        file_path="r.ARW",
        green_path="g.ARW",
        blue_path="b.ARW",
        workspace_color_space="Adobe RGB",
        use_camera_wb=False,
        detect_mode=False,
    )

    with patch("negpy.features.process.logic.detect_process_mode") as dpm:
        worker.process(task)

    dpm.assert_not_called()
    assert finished[0][-1] == ""


def test_automatic_import_returns_classifier_result_from_public_process(qapp):
    from negpy.desktop.workers.render import PreviewLoadTask, PreviewLoadWorker

    service = MagicMock()
    raw = np.ones((4, 4, 3), dtype=np.float32)
    service.load_linear_preview.return_value = (raw, (4, 4), {})
    worker = PreviewLoadWorker(service)
    finished = []
    worker.finished.connect(lambda *args: finished.append(args))
    task = PreviewLoadTask(
        file_path="auto.ARW",
        workspace_color_space="Adobe RGB",
        use_camera_wb=False,
        use_splash=False,
        detect_mode=True,
    )

    with patch("negpy.features.process.logic.detect_process_mode", return_value="E-6") as dpm:
        worker.process(task)

    dpm.assert_called_once_with(raw)
    assert finished[0][-1] == "E-6"
