import qtawesome as qta
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from negpy.desktop.view.styles.templates import labeled_action, labeled_toggle, load_stylesheet


def test_labeled_buttons_fit_the_shared_fixed_height(qapp):
    root = QWidget()
    root.setStyleSheet(load_stylesheet())
    layout = QVBoxLayout(root)
    ref = QPushButton(" Ref")
    ref.setIcon(qta.icon("fa5s.circle"))
    toggle = labeled_toggle("fa5s.sun", " Highlights", False, "tip")
    action = labeled_action("fa5s.times", " Clear Region", "tip")
    for w in (ref, toggle, action):
        layout.addWidget(w)
    root.ensurePolished()
    for w in (ref, toggle, action):
        w.ensurePolished()

    assert toggle.sizeHint().height() <= ref.sizeHint().height()
    assert action.sizeHint().height() <= ref.sizeHint().height()
