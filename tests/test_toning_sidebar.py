from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.toning import ToningSidebar
from negpy.features.altprocess.models import AltProcess
from negpy.features.process.models import ProcessMode


def _sidebar(alt: AltProcess) -> ToningSidebar:
    controller = MagicMock()
    controller.state = AppState()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg, process=replace(cfg.process, process_mode=ProcessMode.BW), altproc=replace(cfg.altproc, alt_process=alt)
    )
    sidebar = ToningSidebar(controller)
    sidebar.sync_ui()
    return sidebar


def test_an_alternative_process_says_why_it_grays_the_toners(qapp):
    """Disabled sliders get no hover, so the reason has to be on screen."""
    assert _sidebar(AltProcess.NONE).alt_process_hint.isHidden()

    lith = _sidebar(AltProcess.LITH)
    assert not lith.alt_process_hint.isHidden()
    assert "Selenium and Gold" in lith.alt_process_hint.text()
    assert not lith.sepia_slider.isEnabled() and lith.selenium_slider.isEnabled()

    cyano = _sidebar(AltProcess.CYANOTYPE)
    assert "Bleach and Tannin" in cyano.alt_process_hint.text()
    assert not cyano.selenium_slider.isEnabled()
