"""Every exposure of an RGB-scan triplet must decode on no white balance at all, never
its own as-shot multipliers — and never a value shared by all three, either.

A triplet exposure is a single narrowband channel: only one raw channel carries real
signal, so a WB gain applied to it corrects nothing, there being no full-spectrum scene
for it to describe. That makes this a sharper case than the bracket merge fixed in
test_hdr_solve_white_balance.py: a bracket's frames share one real scene white balance to
agree on, so pinning them to one frame's gain is the right fix; a triplet has no such
value to agree on even if every exposure's own gain happened to match. Left unfixed, each
of the three files' own multiplier got baked into the one channel the merge keeps from it.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rawpy

from negpy.domain.models import WorkspaceConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.services.rendering.preview_manager import PreviewManager


class _SpyRaw:
    """Records the kwargs its own postprocess() call received. One instance per file,
    so a per-path decode can be told apart from its siblings'."""

    raw_type = rawpy.RawType.Flat
    raw_pattern = np.zeros((2, 2), dtype=np.uint8)
    sizes = SimpleNamespace(raw_height=8, raw_width=8, iheight=8, iwidth=8)

    def __init__(self) -> None:
        self.seen: dict = {}

    def __enter__(self) -> "_SpyRaw":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def postprocess(self, **kwargs: object) -> np.ndarray:
        self.seen.update(kwargs)
        return np.zeros((8, 8, 3), dtype=np.uint16)


def test_triplet_render_decode_pins_every_exposure_neutral(tmp_path):
    red = str(tmp_path / "r.arw")
    green = str(tmp_path / "g.arw")
    blue = str(tmp_path / "b.arw")
    for p in (red, green, blue):
        open(p, "wb").close()

    processor = ImageProcessor()
    calls: list = []

    def fake_decode(path, linear_raw, fast=False, wb_override=None, demosaic="Auto"):
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
    processor._decode_sensor_rgb = lambda path, linear_raw, fast=False, wb_override=None, demosaic="Auto": (
        np.zeros((4, 4, 3), dtype=np.uint16),
        {"cam_xyz": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "camera_wb": [1.9, 1.0, 1.55]},
    )

    cfg = replace(WorkspaceConfig(), rgbscan=RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=False))
    processor._decode_oriented_f32(red, cfg)

    assert processor.camera_wb_for(red) is None


def test_triplet_with_linear_raw_off_still_decodes_neutral_at_the_rawpy_call(tmp_path):
    """Reproduces the bug as a user hits it without ever touching Scanning Setup: add
    three files, assemble them into a triplet (Trichrome Scan toggle or Edit RGB Triplet…),
    and render. Nothing about that path sets Linear RAW — it stays at its default,
    False — so `use_camera_wb` would read True for every exposure if the triplet
    branch did not override it. Goes through the real `_decode_sensor_rgb`, not a
    mock of it, so a regression in its `use_camera_wb`/`user_wb` computation itself
    would be caught here even if the plumbing above it still looked right.
    """
    red = str(tmp_path / "r.arw")
    green = str(tmp_path / "g.arw")
    blue = str(tmp_path / "b.arw")
    for p in (red, green, blue):
        open(p, "wb").close()

    spies = {path: _SpyRaw() for path in (red, green, blue)}

    def fake_get_loader(path, linear_raw=False):
        return spies[path], {}

    processor = ImageProcessor()
    cfg = replace(
        WorkspaceConfig(),
        process=replace(WorkspaceConfig().process, linear_raw=False),
        rgbscan=RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=False),
    )
    with patch("negpy.services.rendering.image_processor.loader_factory") as lf:
        lf.get_loader.side_effect = fake_get_loader
        processor._decode_oriented_f32(red, cfg)

    for path, spy in spies.items():
        assert spy.seen["use_camera_wb"] is False, f"{path} decoded on its own as-shot white balance"
        assert spy.seen["user_wb"] == [1.0, 1.0, 1.0, 1.0], f"{path} did not decode neutral"


def test_preview_merge_decodes_every_exposure_neutral(tmp_path):
    red = str(tmp_path / "r.arw")
    green = str(tmp_path / "g.arw")
    blue = str(tmp_path / "b.arw")

    pm = PreviewManager()
    calls: list = []

    def fake_preview(path, color_space=None, use_camera_wb=False, full_resolution=False, file_hash=None, demosaic="Auto"):
        calls.append((path, use_camera_wb))
        return np.zeros((4, 4, 3), dtype=np.float32), (4, 4), {"camera_wb": [1.9, 1.0, 1.55]}

    pm.load_linear_preview = fake_preview

    rgbscan = RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=False)
    _out, _dims, meta = pm.load_linear_preview_rgb(red, rgbscan, "Adobe RGB", use_camera_wb=True)

    assert {path for path, _wb in calls} == {red, green, blue}
    assert all(wb is False for _path, wb in calls), "caller asked for as-shot WB, the triplet merge must refuse it per-file"
    assert meta["camera_wb"] is None, "the red file's own as-shot metadata must not ride along as the merge's"
