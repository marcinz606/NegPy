"""Every exposure of an RGB-scan triplet must decode on one neutral white balance,
never its own as-shot multipliers.

A triplet exposure is a single narrowband channel: only one raw channel carries real
signal, so an as-shot white balance has no scene to correct for, and a camera left on
auto WB records a different one per frame anyway. Left unpinned, each of the three
files' own multipliers get baked into the one channel the merge keeps from it, which is
exactly the failure the bracket merge was already pinned against (see
test_hdr_solve_white_balance.py) — RGB scan never got the same treatment.
"""

from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.services.rendering.preview_manager import PreviewManager


def test_triplet_render_decode_pins_every_exposure_neutral(tmp_path):
    red = str(tmp_path / "r.arw")
    green = str(tmp_path / "g.arw")
    blue = str(tmp_path / "b.arw")
    for p in (red, green, blue):
        open(p, "wb").close()

    processor = ImageProcessor()
    calls: list = []

    def fake_decode(path, linear_raw, fast=False, wb_override=None):
        calls.append((path, wb_override))
        return np.zeros((4, 4, 3), dtype=np.uint16), {"cam_xyz": None, "camera_wb": [1.9, 1.0, 1.55]}

    processor._decode_sensor_rgb = fake_decode

    cfg = replace(WorkspaceConfig(), rgbscan=RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=False))
    processor._decode_oriented_f32(red, cfg)

    assert {path for path, _wb in calls} == {red, green, blue}
    assert all(wb == (1.0, 1.0, 1.0, 1.0) for _path, wb in calls)


def test_triplet_render_ignores_the_primarys_own_camera_wb_downstream(tmp_path):
    """The primary exposure's as-shot camera_wb must not survive into the per-path memo
    a triplet render reads back later (e.g. the capture-matrix fold)."""
    red = str(tmp_path / "r.arw")
    green = str(tmp_path / "g.arw")
    blue = str(tmp_path / "b.arw")
    for p in (red, green, blue):
        open(p, "wb").close()

    processor = ImageProcessor()
    processor._decode_sensor_rgb = lambda path, linear_raw, fast=False, wb_override=None: (
        np.zeros((4, 4, 3), dtype=np.uint16),
        {"cam_xyz": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "camera_wb": [1.9, 1.0, 1.55]},
    )

    cfg = replace(WorkspaceConfig(), rgbscan=RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=False))
    processor._decode_oriented_f32(red, cfg)

    assert processor.camera_wb_for(red) is None


def test_preview_merge_decodes_every_exposure_neutral(tmp_path):
    red = str(tmp_path / "r.arw")
    green = str(tmp_path / "g.arw")
    blue = str(tmp_path / "b.arw")

    pm = PreviewManager()
    calls: list = []

    def fake_preview(path, color_space=None, use_camera_wb=False, full_resolution=False, file_hash=None):
        calls.append((path, use_camera_wb))
        return np.zeros((4, 4, 3), dtype=np.float32), (4, 4), {"camera_wb": [1.9, 1.0, 1.55]}

    pm.load_linear_preview = fake_preview

    rgbscan = RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=False)
    _out, _dims, meta = pm.load_linear_preview_rgb(red, rgbscan, "Adobe RGB", use_camera_wb=True)

    assert {path for path, _wb in calls} == {red, green, blue}
    assert all(wb is False for _path, wb in calls), "caller asked for as-shot WB, the triplet merge must refuse it per-file"
    assert meta["camera_wb"] is None, "the red file's own as-shot metadata must not ride along as the merge's"
