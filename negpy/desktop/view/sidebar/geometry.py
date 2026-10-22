from negpy.kernel.system.i18n import tr
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.session import ToolMode
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import header_row, section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.fold import rail_fold
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.geometry.models import FINE_ROTATION_LIMIT


def _keystone_reference_icon() -> QIcon:
    icon = QIcon()
    modes = (
        (QIcon.Mode.Normal, THEME.text_hint, THEME.text_primary),
        (QIcon.Mode.Active, THEME.text_hint, THEME.text_primary),
        (QIcon.Mode.Disabled, THEME.text_muted, THEME.text_muted),
    )
    for mode, off_color, on_color in modes:
        for state, color in ((QIcon.State.Off, off_color), (QIcon.State.On, on_color)):
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor(color), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for y in (3, 15, 27):
                painter.drawLine(2, y, 29, 21)
            for x in (4, 15, 27):
                painter.drawLine(21, 3, x, 29)
            painter.setBrush(QColor(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(29.0, 21.0), 1.5, 1.5)
            painter.drawEllipse(QPointF(21.0, 3.0), 1.5, 1.5)
            painter.end()
            icon.addPixmap(pixmap, mode, state)
    return icon


class GeometrySidebar(BaseSidebar):
    """Straighten: how this frame lies, turned, flipped and leveled."""

    def _init_ui(self) -> None:
        conf = self.state.config.geometry

        # The toolbar's buttons, here too; MainWindow wires them to the same rotate and flip.
        self.rot_l_btn = self._icon_action("mdi6.file-rotate-left", tr("Rotate CCW"))
        self.rot_r_btn = self._icon_action("mdi6.file-rotate-right", tr("Rotate CW"))
        self.flip_h_btn = self._icon_action("fa5s.arrows-alt-h", tr("Flip Horizontal"))
        self.flip_v_btn = self._icon_action("fa5s.arrows-alt-v", tr("Flip Vertical"))
        self.layout.addLayout(
            header_row(section_subheader(tr("ROTATE AND FLIP")), self.rot_l_btn, self.rot_r_btn, self.flip_h_btn, self.flip_v_btn)
        )

        self.straighten_btn = self._tool_toggle("fa5s.ruler", tr("Level"), tr("Draw a line along a horizon or edge to level the frame"))
        self.auto_skew_btn = self._labeled_action(
            "fa5s.ruler-combined", tr("Square to Film"), tr("Square to Film: line the picture up with the edges of the film.")
        )
        self.keystone_lines_btn = self._tool_toggle("", tr("Draw Lines"), tr("Tilt and swing with reference lines"))
        self.keystone_lines_btn.setIcon(_keystone_reference_icon())
        tool_row = QHBoxLayout()
        for btn in (self.straighten_btn, self.auto_skew_btn, self.keystone_lines_btn):
            tool_row.addWidget(btn, 1)
        self.layout.addLayout(tool_row)

        # The slider shows the photographer's convention, where positive is clockwise on screen.
        # Internally geometry.fine_rotation keeps the cv2/warp convention, where positive is
        # counter-clockwise and flip-independent because flips apply before fine rotation, so
        # saved edits keep their meaning: display = -stored at this boundary.
        self.fine_rot_slider = CompactSlider(
            tr("Fine Rotation"), -FINE_ROTATION_LIMIT, FINE_ROTATION_LIMIT, -conf.fine_rotation, step=0.1, unit="°"
        )
        self.layout.addWidget(self.fine_rot_slider)

        self.converge_v_slider = CompactSlider(tr("Tilt"), -15.0, 15.0, conf.converge_v, step=0.1, unit="%")
        self.converge_v_slider.setToolTip(
            tr("Tilt: straighten buildings that lean back because the camera pointed up, where positive stretches the top.")
        )
        self.converge_h_slider = CompactSlider(tr("Swing"), -15.0, 15.0, conf.converge_h, step=0.1, unit="%")
        self.converge_h_slider.setToolTip(tr("Swing: straighten a wall that narrows to one side, where positive stretches the left side."))
        self.layout.addWidget(rail_fold("geometry_easel", self.fine_rot_slider, self.converge_v_slider, self.converge_h_slider))

        self.crop_to_valid_btn = self._small_toggle(
            "fa5s.crop",
            tr("Crop Empty Corners"),
            conf.crop_to_valid,
            tr("Crop Empty Corners: crop away the empty corners that rotation, Tilt and Swing leave."),
            align_left=True,
        )
        self.layout.addWidget(self.crop_to_valid_btn)

    def _connect_signals(self) -> None:
        self.auto_skew_btn.clicked.connect(self.controller.auto_skew_frame)
        self.straighten_btn.toggled.connect(self._on_straighten_toggled)
        self.keystone_lines_btn.toggled.connect(self._on_keystone_lines_toggled)

        # Display convention is CW-positive; negate crossing into the stored convention.
        self.fine_rot_slider.valueChanged.connect(
            lambda v: self.update_config_section("geometry", render=True, persist=False, readback_metrics=False, fine_rotation=-v)
        )
        self.fine_rot_slider.valueChanged.connect(lambda _v: self.controller.show_rotation_guide())
        self.fine_rot_slider.valueCommitted.connect(lambda _v: self.controller.reset_all_peeks())
        self.fine_rot_slider.valueCommitted.connect(
            lambda v: self.update_config_section("geometry", render=True, persist=True, readback_metrics=True, fine_rotation=-v)
        )

        self.crop_to_valid_btn.toggled.connect(self._on_crop_to_valid_toggled)

        for slider, field in (
            (self.converge_v_slider, "converge_v"),
            (self.converge_h_slider, "converge_h"),
        ):
            slider.valueChanged.connect(
                lambda v, f=field: self.update_config_section("geometry", render=True, persist=False, readback_metrics=False, **{f: v})
            )
            slider.valueCommitted.connect(
                lambda v, f=field: self.update_config_section("geometry", render=True, persist=True, readback_metrics=True, **{f: v})
            )

        for slider in (self.converge_v_slider, self.converge_h_slider):
            slider.valueChanged.connect(lambda _v: self.controller.show_rotation_guide())

    def _on_straighten_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.STRAIGHTEN if checked else ToolMode.NONE)

    def _on_keystone_lines_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.KEYSTONE_LINES if checked else ToolMode.NONE)

    def _on_crop_to_valid_toggled(self, checked: bool) -> None:
        self.update_config_section("geometry", render=True, persist=True, readback_metrics=True, crop_to_valid=checked)
        self.controller.show_rotation_guide()

    def sync_ui(self) -> None:
        conf = self.state.config.geometry

        self.block_signals(True)
        try:
            self.fine_rot_slider.setValue(-conf.fine_rotation)
            self.converge_v_slider.setValue(conf.converge_v)
            self.converge_h_slider.setValue(conf.converge_h)

            self.straighten_btn.setChecked(self.state.active_tool == ToolMode.STRAIGHTEN)
            self.keystone_lines_btn.setChecked(self.state.active_tool == ToolMode.KEYSTONE_LINES)
            self.crop_to_valid_btn.setChecked(conf.crop_to_valid)
            self.crop_to_valid_btn.edited_dot.set_active(conf.crop_to_valid)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.fine_rot_slider,
            self.converge_v_slider,
            self.converge_h_slider,
            self.straighten_btn,
            self.keystone_lines_btn,
            self.crop_to_valid_btn,
        ):
            w.blockSignals(blocked)
