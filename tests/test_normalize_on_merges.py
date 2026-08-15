"""A merged bracket never carries the Normalize stretch.

The two decide the same thing. Normalize meters the frame and stretches the measured range
to full, which divides the merge's render exposure straight back out — below the point
where its specular stops clipping, moving the anchor changes nothing at all. They do not
want each other either: Normalize rescues faded film, fading compresses density range, and
a frame whose range collapsed is not one that needed a bracket.

The invariant is held in WorkspaceConfig rather than at the render, because e6_normalize is
read from `is_transparency_transfer` down through both engines and the sidebars — a rule
applied at only some of those is the hidden-but-live trap the Calibration panel already
learned once.
"""

import sys
import unittest
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.process import ProcessSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.normalization import analyze_log_exposure_bounds
from negpy.features.exposure.transfer import is_transparency_transfer
from negpy.features.hdr.models import HdrConfig
from negpy.features.process.models import ProcessMode

if not QApplication.instance():
    _app = QApplication(sys.argv)

_MERGE = HdrConfig(hdr_enabled=True, hdr_paths=("/x/b.nef",))


def _slide(normalize=True, **hdr) -> WorkspaceConfig:
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=normalize))
    return replace(cfg, hdr=HdrConfig(**hdr)) if hdr else cfg


class Invariant(unittest.TestCase):
    def test_a_merge_drops_the_stretch(self):
        merged = replace(_slide(normalize=True), hdr=_MERGE)
        self.assertFalse(merged.process.e6_normalize)

    def test_an_unmerged_frame_keeps_it(self):
        self.assertTrue(_slide(normalize=True).process.e6_normalize)

    def test_it_holds_however_the_config_arrives(self):
        """A merge created now, a composite loaded from the DB, and a replace that turns an
        ordinary frame into one all have to land in the same place."""
        merged = replace(_slide(normalize=True), hdr=_MERGE)
        self.assertFalse(WorkspaceConfig.from_flat_dict({**merged.to_dict(), "e6_normalize": True}).process.e6_normalize)
        self.assertFalse(replace(_slide(normalize=True), hdr=_MERGE).process.e6_normalize)

    def test_an_inactive_bracket_is_not_a_merge(self):
        """Paths without the enable flag are a dissolved merge, not a live one."""
        seeded = replace(_slide(normalize=True), hdr=HdrConfig(hdr_enabled=False, hdr_paths=("/x/b.nef",)))
        self.assertTrue(seeded.process.e6_normalize)

    def test_the_render_follows_the_config(self):
        """The whole point of holding it here: every reader of e6_normalize sees the
        resolved value, so a merged slide stays on the transfer path."""
        merged = replace(_slide(normalize=True), hdr=_MERGE)
        self.assertTrue(is_transparency_transfer(merged.process.process_mode, merged.process.e6_normalize))


class WhyItIsRefused(unittest.TestCase):
    def test_the_stretch_cancels_the_render_exposure(self):
        """Metering and stretching divides the anchor's scale back out: the span the print
        is built from comes out identical at every anchor that does not clip."""
        rng = np.random.default_rng(0)
        scene = rng.uniform(0.002, 0.9, (128, 128, 3)).astype(np.float32)

        spans = []
        for k in (0.5, 0.25):
            bounds = analyze_log_exposure_bounds(np.clip(scene * k, 0.0, 1.0), process_mode=ProcessMode.E6, e6_normalize=True)
            spans.append(np.asarray(bounds.ceils) - np.asarray(bounds.floors))

        np.testing.assert_allclose(spans[0], spans[1], rtol=1e-6)


class Panel(unittest.TestCase):
    def _sidebar(self, hdr: HdrConfig):
        ctrl = MagicMock()
        cfg = replace(_slide(normalize=False), hdr=hdr)
        ctrl.state.config = cfg
        ctrl.state.autodetect_enabled = False
        w = ProcessSidebar(ctrl)
        w.sync_ui()
        return w

    def test_greyed_with_a_reason_on_a_merge(self):
        w = self._sidebar(_MERGE)
        # isHidden, not isVisible: the sidebar is never shown here.
        self.assertFalse(w.normalize_e6_btn.isHidden(), "hiding it teaches nothing about why")
        self.assertFalse(w.normalize_e6_btn.isEnabled())
        self.assertFalse(w.normalize_merged_hint.isHidden())

    def test_live_on_an_ordinary_slide(self):
        w = self._sidebar(HdrConfig())
        self.assertTrue(w.normalize_e6_btn.isEnabled())
        self.assertTrue(w.normalize_merged_hint.isHidden())


if __name__ == "__main__":
    unittest.main()
