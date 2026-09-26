from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.finish import FinishSidebar


def _sidebar(match_paper: bool) -> tuple[MagicMock, FinishSidebar]:
    controller = MagicMock()
    controller.state = AppState()
    cfg = controller.state.config
    controller.state.config = replace(cfg, finish=replace(cfg.finish, border_match_paper=match_paper))
    sidebar = FinishSidebar(controller)
    sidebar.sync_ui()
    return controller, sidebar


def test_border_color_is_a_choice_and_the_swatch_is_live_only_for_custom(qapp):
    _, paper = _sidebar(match_paper=True)
    assert paper.border_color_btn.currentIndex() == 0
    assert not paper.color_btn.isEnabled()

    controller, custom = _sidebar(match_paper=False)
    assert custom.border_color_btn.currentIndex() == 1
    assert custom.color_btn.isEnabled()

    custom.border_color_btn.choice_menu.actions()[0].trigger()
    assert controller.apply_config.call_args.args[0].finish.border_match_paper is True
