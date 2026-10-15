from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.geometry import GeometrySidebar


def test_fine_rotation_commit_ends_the_peeks(qapp):
    controller = MagicMock()
    controller.state = AppState()
    sidebar = GeometrySidebar(controller)

    sidebar.fine_rot_slider.valueChanged.emit(1.0)
    controller.reset_all_peeks.assert_not_called()

    sidebar.fine_rot_slider.valueCommitted.emit(1.0)
    controller.reset_all_peeks.assert_called_once_with()
