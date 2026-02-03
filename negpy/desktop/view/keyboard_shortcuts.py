from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import Qt


def setup_keyboard_shortcuts(window) -> None:
    """Defines global application hotkeys for the main window."""
    controller = window.controller
    toolbar = window.toolbar
    controls = window.controls_panel

    # Navigation
    QShortcut(QKeySequence(Qt.Key.Key_Left), window, controller.session.prev_file)
    QShortcut(QKeySequence(Qt.Key.Key_Right), window, controller.session.next_file)

    # Geometry
    QShortcut(QKeySequence("["), window, lambda: toolbar.rotate(1))
    QShortcut(QKeySequence("]"), window, lambda: toolbar.rotate(-1))
    QShortcut(QKeySequence("H"), window, lambda: toolbar.flip("horizontal"))
    QShortcut(QKeySequence("V"), window, lambda: toolbar.flip("vertical"))

    # Tools
    QShortcut(
        QKeySequence("Shift+W"),
        window,
        lambda: controls.exposure_sidebar.pick_wb_btn.toggle(),
    )
    QShortcut(
        QKeySequence("Shift+C"),
        window,
        lambda: controls.geometry_sidebar.manual_crop_btn.toggle(),
    )
    QShortcut(
        QKeySequence("Shift+D"),
        window,
        lambda: controls.retouch_sidebar.pick_dust_btn.toggle(),
    )

    # Sliders (QA: Density, WS: Grade, ED: Magenta, RF: Yellow, ZC: Offset)
    def adjust(slider, delta):
        slider.setValue(slider.value() + delta)
        slider.valueChanged.emit(slider.value())

    QShortcut(
        QKeySequence("Q"),
        window,
        lambda: adjust(controls.exposure_sidebar.density_slider, 0.01),
    )
    QShortcut(
        QKeySequence("A"),
        window,
        lambda: adjust(controls.exposure_sidebar.density_slider, -0.01),
    )

    QShortcut(
        QKeySequence("W"),
        window,
        lambda: adjust(controls.exposure_sidebar.grade_slider, 0.01),
    )
    QShortcut(
        QKeySequence("S"),
        window,
        lambda: adjust(controls.exposure_sidebar.grade_slider, -0.01),
    )

    QShortcut(
        QKeySequence("E"),
        window,
        lambda: adjust(controls.exposure_sidebar.magenta_slider, 0.01),
    )
    QShortcut(
        QKeySequence("D"),
        window,
        lambda: adjust(controls.exposure_sidebar.magenta_slider, -0.01),
    )

    QShortcut(
        QKeySequence("R"),
        window,
        lambda: adjust(controls.exposure_sidebar.yellow_slider, 0.01),
    )
    QShortcut(
        QKeySequence("F"),
        window,
        lambda: adjust(controls.exposure_sidebar.yellow_slider, -0.01),
    )

    QShortcut(
        QKeySequence("Z"),
        window,
        lambda: adjust(controls.geometry_sidebar.offset_slider, -1.0),
    )
    QShortcut(
        QKeySequence("X"),
        window,
        lambda: adjust(controls.geometry_sidebar.offset_slider, 1.0),
    )

    # Actions
    QShortcut(QKeySequence("Ctrl+E"), window, controller.request_export)
    QShortcut(QKeySequence("Ctrl+C"), window, controller.session.copy_settings)
    QShortcut(QKeySequence("Ctrl+V"), window, controller.session.paste_settings)
