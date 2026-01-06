from src.presentation.state.view_models import ExposureViewModel
from src.presentation.components.exposure_view import render_exposure_view


def render_exposure_section() -> None:
    """
    Renders the 'Exposure & Tonality' section of the sidebar.
    Delegates to the new Clean Architecture component.
    """
    # Create the ViewModel (which wraps the current session state)
    vm = ExposureViewModel()

    # Render the pure view
    render_exposure_view(vm)
