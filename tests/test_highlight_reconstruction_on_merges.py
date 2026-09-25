"""A merged bracket never carries Highlight Reconstruction.

Libraw's own reconstruction fills a clipped pixel with a per-frame guess, so the sample no
longer reads near the sensor ceiling -- the exact thing the merge's own clip detection
(`clipped_fraction`, `_accumulate`'s roll-off) trusts to tell a genuine highlight from a
frame that still has signal there. Blending independently-guessed values from different
frames prints as inconsistent color where the merge should recover a real highlight from a
shorter, unclipped exposure instead.

The invariant is held in WorkspaceConfig, the same place and for the same reason as the
Auto Density/Auto Grade exclusion (see test_autos_on_merges.py).
"""

import sys
import unittest
from dataclasses import replace
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.demosaic import DemosaicSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.features.hdr.models import HdrConfig
from negpy.features.process.models import ProcessMode

if not QApplication.instance():
    _app = QApplication(sys.argv)

_MERGE = HdrConfig(hdr_enabled=True, hdr_paths=("/x/b.nef",))


def _slide(level=5, **hdr) -> WorkspaceConfig:
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, highlight_reconstruction=level))
    return replace(cfg, hdr=HdrConfig(**hdr)) if hdr else cfg


class Invariant(unittest.TestCase):
    def test_a_merge_drops_reconstruction(self):
        merged = replace(_slide(5), hdr=_MERGE)
        self.assertEqual(merged.process.highlight_reconstruction, 0)

    def test_an_unmerged_frame_keeps_it(self):
        self.assertEqual(_slide(5).process.highlight_reconstruction, 5)

    def test_it_holds_however_the_config_arrives(self):
        """A merge created now, a composite loaded from the DB, and a replace that turns an
        ordinary frame into one all have to land in the same place."""
        merged = replace(_slide(5), hdr=_MERGE)
        self.assertEqual(
            WorkspaceConfig.from_flat_dict({**merged.to_dict(), "highlight_reconstruction": 5}).process.highlight_reconstruction, 0
        )
        self.assertEqual(replace(_slide(5), hdr=_MERGE).process.highlight_reconstruction, 0)

    def test_an_inactive_bracket_is_not_a_merge(self):
        """Paths without the enable flag are a dissolved merge, not a live one."""
        seeded = replace(_slide(5), hdr=HdrConfig(hdr_enabled=False, hdr_paths=("/x/b.nef",)))
        self.assertEqual(seeded.process.highlight_reconstruction, 5)


class Panel(unittest.TestCase):
    def _sidebar(self, hdr: HdrConfig, level: int = 5):
        ctrl = MagicMock()
        cfg = replace(_slide(level), hdr=hdr)
        ctrl.state.config = cfg
        ctrl.state.autodetect_enabled = False
        ctrl.state.preview_cam_xyz = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        w = DemosaicSidebar(ctrl)
        w.sync_ui()
        return w

    def test_greyed_with_a_reason_on_a_merge(self):
        w = self._sidebar(_MERGE)
        # isHidden, not isVisible: the sidebar is never shown here.
        self.assertFalse(w.highlight_combo.isHidden(), "hiding it teaches nothing about why")
        self.assertFalse(w.highlight_combo.isEnabled())
        self.assertFalse(w.highlight_merged_hint.isHidden())

    def test_live_on_an_ordinary_slide(self):
        w = self._sidebar(HdrConfig())
        self.assertTrue(w.highlight_combo.isEnabled())
        self.assertTrue(w.highlight_merged_hint.isHidden())


if __name__ == "__main__":
    unittest.main()
