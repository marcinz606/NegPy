from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.domain.models import WorkspaceConfig


def _controller() -> MagicMock:
    controller = MagicMock()
    controller.state = AppState()
    controller._render_state = lambda: AppController._render_state(controller)
    controller._dispatched_render_state = controller._render_state()
    return controller


def test_a_selection_change_does_not_render_again() -> None:
    controller = _controller()
    AppController._render_if_state_moved(controller)
    controller._render_debounce.start.assert_not_called()


def test_an_edit_renders() -> None:
    controller = _controller()
    controller.state.config = WorkspaceConfig()
    AppController._render_if_state_moved(controller)
    controller._render_debounce.start.assert_called_once()


def test_turning_hq_preview_on_renders() -> None:
    controller = _controller()
    controller.state.hq_preview = not controller.state.hq_preview
    AppController._render_if_state_moved(controller)
    controller._render_debounce.start.assert_called_once()
