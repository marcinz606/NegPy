"""A merged bracket never carries Auto Density or Auto Grade.

The merge's render exposure already places the tones. The autos meter the merged frame and
place them again, which divides that choice straight back out.

The invariant is held in WorkspaceConfig rather than at the render, because the toggles are
read down through both engines and the sidebars.
"""

import sys
import unittest
from dataclasses import replace
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QApplication

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.tone import ToneSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.features.hdr.models import HdrConfig
from negpy.features.process.models import ProcessMode

if not QApplication.instance():
    _app = QApplication(sys.argv)

_MERGE = HdrConfig(hdr_enabled=True, hdr_paths=("/x/b.nef",))


def _metered(mode=ProcessMode.C41) -> WorkspaceConfig:
    cfg = WorkspaceConfig()
    return replace(
        cfg,
        process=replace(cfg.process, process_mode=mode),
        exposure=replace(cfg.exposure, auto_exposure=True, auto_normalize_contrast=True),
    )


def _autos(cfg: WorkspaceConfig) -> tuple[bool, bool]:
    return cfg.exposure.auto_exposure, cfg.exposure.auto_normalize_contrast


class Invariant(unittest.TestCase):
    def test_a_merge_drops_the_autos(self):
        for mode in (ProcessMode.C41, ProcessMode.E6):
            with self.subTest(mode=mode):
                self.assertEqual(_autos(replace(_metered(mode), hdr=_MERGE)), (False, False))

    def test_an_unmerged_frame_keeps_them(self):
        self.assertEqual(_autos(_metered()), (True, True))

    def test_it_holds_however_the_config_arrives(self):
        """A merge created now, a composite loaded from the DB, and a replace that turns an
        ordinary frame into one all have to land in the same place."""
        merged = replace(_metered(), hdr=_MERGE)
        loaded = WorkspaceConfig.from_flat_dict({**merged.to_dict(), "auto_exposure": True, "auto_normalize_contrast": True})
        self.assertEqual(_autos(loaded), (False, False))

    def test_an_inactive_bracket_is_not_a_merge(self):
        """Paths without the enable flag are a dissolved merge, not a live one."""
        seeded = replace(_metered(), hdr=HdrConfig(hdr_enabled=False, hdr_paths=("/x/b.nef",)))
        self.assertEqual(_autos(seeded), (True, True))

    def test_the_autos_and_reconstruction_drop_together(self):
        cfg = _metered(ProcessMode.E6)
        merged = replace(replace(cfg, process=replace(cfg.process, highlight_reconstruction=5)), hdr=_MERGE)
        self.assertEqual(_autos(merged), (False, False))
        self.assertEqual(merged.process.highlight_reconstruction, 0)


class Panel(unittest.TestCase):
    def _sidebar(self, hdr: HdrConfig):
        ctrl = MagicMock()
        ctrl.state = AppState()
        ctrl.state.config = replace(_metered(ProcessMode.E6), hdr=hdr)
        w = ToneSidebar(ctrl)
        w.sync_ui()
        return w

    def test_greyed_with_a_reason_on_a_merge(self):
        w = self._sidebar(_MERGE)
        # isHidden, not isVisible: the sidebar is never shown here.
        self.assertFalse(w.auto_btn.isHidden(), "hiding it teaches nothing about why")
        for action in (w.auto_density_action, w.auto_grade_action):
            self.assertFalse(action.isEnabled())
        self.assertFalse(w.auto_merged_hint.isHidden())

    def test_live_on_an_ordinary_slide(self):
        w = self._sidebar(HdrConfig())
        self.assertTrue(w.auto_density_action.isEnabled())
        self.assertTrue(w.auto_grade_action.isEnabled())
        self.assertTrue(w.auto_merged_hint.isHidden())


if __name__ == "__main__":
    unittest.main()
