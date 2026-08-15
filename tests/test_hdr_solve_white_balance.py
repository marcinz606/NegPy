"""The bracket solve decodes every frame on one white balance, not each frame's own.

`merge_bracket` pins the merge at render (`wb_override=bracket_wb`), but the *solve* runs
before it, one frame at a time with `hdr` cleared, so it never reaches that branch. Left
unpinned, `use_camera_wb` reads each file's as-shot multipliers — a camera on auto records
different ones per frame — and `pair_ratio` absorbs the spread into the exposure ratio
rather than measuring it.

It stayed hidden because the default slide path is a transparency transfer, which forces a
neutral decode for every frame anyway. Turning Normalize on for a bracket leaves that path
and puts the solve straight onto per-file auto white balance, with nothing said about it.

How badly that reads out depends on which multipliers moved (see WhenItMatters). The pin
is not justified by the worst case: it is justified by the solve and the render having to
stand on the same footing, and the render already pins.
"""

import unittest
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.workers.hdr import HdrTask, HdrWorker
from negpy.domain.models import WorkspaceConfig
from negpy.features.hdr.logic import pair_ratio
from negpy.features.process.models import ProcessMode

_WB = [1.9, 1.0, 1.55, 0.0]


def _slide(**process) -> WorkspaceConfig:
    cfg = WorkspaceConfig()
    return replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, **process))


class _Processor:
    """Records the white balance each frame was asked to decode on."""

    def __init__(self):
        self.overrides = []

    def _decode_oriented_f32(self, path, params, fast_decode=False, wb_override=None):
        self.overrides.append(wb_override)
        return np.full((8, 8, 3), 0.5, dtype=np.float32), None, "sRGB"

    def camera_wb_for(self, path):
        return list(_WB)

    def cleanup(self, **kw):
        pass


def _run(params: WorkspaceConfig):
    worker = HdrWorker.__new__(HdrWorker)
    worker._processor = _Processor()
    worker._cancel = MagicMock()
    worker._cancel.is_set.return_value = False
    for signal in ("progress", "solved", "cancelled", "error"):
        setattr(worker, signal, MagicMock())
    files = tuple({"path": f"/x/{i}.nef", "name": f"{i}.nef"} for i in range(3))
    worker.run(HdrTask(files=files, params_by_path={f["path"]: params for f in files}))
    worker.error.emit.assert_not_called()
    return worker._processor.overrides


class SolvePin(unittest.TestCase):
    def test_a_camera_wb_bracket_is_pinned_to_the_first_frame(self):
        """Normalize on leaves the transfer path, so the decode carries as-shot gains —
        the case that has to be pinned."""
        overrides = _run(_slide(e6_normalize=True, linear_raw=False))

        self.assertIsNone(overrides[0], "the first frame supplies the pin, it cannot take one")
        self.assertEqual(overrides[1:], [list(_WB), list(_WB)])

    def test_a_neutral_bracket_pins_nothing(self):
        """Linear RAW decodes with unity multipliers, so there are no as-shot gains to
        share and every frame is already on one basis."""
        self.assertEqual(_run(_slide(e6_normalize=True, linear_raw=True)), [None, None, None])

    def test_the_transfer_path_needs_no_pin_either(self):
        """Normalize off forces a neutral decode whatever Linear RAW says, which is why
        this went unnoticed: the default slide bracket was never exposed to it."""
        self.assertEqual(_run(_slide(e6_normalize=False, linear_raw=False)), [None, None, None])


class WhenItMatters(unittest.TestCase):
    """What an unshared white balance actually does to a solved ratio.

    Not "always corrupts it": `pair_ratio` takes one median over the pooled channels, so a
    multiplier change confined to a single channel, or one that moves red and blue in
    opposite directions, leaves the median sitting in the untouched majority. It skews when
    red and blue move the same way, which is the shift that has no channel left in the
    middle. Recorded here so the pin is not defended by a claim wider than the truth — its
    real justification is that the render already pins, and a solve on a different footing
    from the render is wrong whether or not this median happens to survive.
    """

    def _scene(self):
        rng = np.random.default_rng(4)
        scene = rng.uniform(0.05, 0.4, (128, 128, 3)).astype(np.float32)
        return scene, scene * 2.0  # a true one-stop difference

    def _solved(self, gains):
        short, long_ = self._scene()
        return pair_ratio(short * np.array(gains, dtype=np.float32), long_)

    def test_a_shared_basis_measures_the_true_stop(self):
        short, long_ = self._scene()
        self.assertAlmostEqual(pair_ratio(short, long_), 2.0, places=2)

    def test_red_and_blue_moving_together_skews_the_ratio(self):
        self.assertLess(self._solved([1.2, 1.0, 1.2]), 1.75)

    def test_the_median_absorbs_a_one_channel_shift(self):
        self.assertAlmostEqual(self._solved([1.0, 1.0, 1 / 1.4]), 2.0, places=2)
        self.assertAlmostEqual(self._solved([1.2, 1.0, 1 / 1.2]), 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
