"""The bracket preview decodes every sibling frame on the reference's white balance, the
same pin the export merge and the HDR solve already apply (see
`test_hdr_solve_white_balance.py`).

Left unpinned, an active reconstruction bakes the reference's real white balance into its
own decode (`highlight_reconstruction_bakes_wb`) while `should_fold_camera_wb` skips the
downstream camera-matrix fold to match — correct only if every frame the merge reads is on
that same real white balance. A sibling decoded neutral instead renders with no white
balance applied at all: a raw camera-primaries cast, not merely a mismatched one.
"""

from unittest.mock import MagicMock

import numpy as np

from negpy.features.hdr.models import HdrConfig
from negpy.services.rendering.preview_manager import PreviewManager

_REF_WB = [1.9, 1.0, 1.55]


def _bracket(tmp_path, n=3):
    paths = [tmp_path / f"{i}.dng" for i in range(n)]
    for path in paths:
        path.write_bytes(b"x")
    ratios = tuple(2.0**i for i in range(n))
    hdr = HdrConfig(hdr_enabled=True, hdr_paths=tuple(str(p) for p in paths[1:]), hdr_ratios=ratios, hdr_align=False)
    return paths, hdr


def _spy_manager(paths):
    pm = PreviewManager()
    calls = []

    def decode(path, *_args, **kwargs):
        calls.append((path, kwargs.get("wb_override")))
        meta = {"camera_wb": _REF_WB} if path == str(paths[0]) else {}
        return np.full((4, 4, 3), 0.3, dtype=np.float32), (4, 4), meta

    pm.load_linear_preview = MagicMock(side_effect=decode)
    return pm, calls


def test_a_reconstruction_bracket_pins_every_sibling_to_the_reference(tmp_path):
    paths, hdr = _bracket(tmp_path)
    pm, calls = _spy_manager(paths)

    pm.load_linear_preview_hdr(str(paths[0]), hdr, "Adobe RGB", bake_camera_wb=True)

    assert calls[0] == (str(paths[0]), None), "the reference supplies the pin, it cannot take one"
    assert calls[1:] == [(str(paths[1]), _REF_WB), (str(paths[2]), _REF_WB)]


def test_a_bracket_without_reconstruction_pins_nothing(tmp_path):
    paths, hdr = _bracket(tmp_path)
    pm, calls = _spy_manager(paths)

    pm.load_linear_preview_hdr(str(paths[0]), hdr, "Adobe RGB", bake_camera_wb=False)

    assert [wb for _path, wb in calls] == [None, None, None]
