from unittest.mock import MagicMock

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QStackedWidget, QWidget

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.analysis_plate import AnalysisPlate
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.charts import ON_PLATE, ChartWell, VectorscopeWidget, WaveformWidget


def _stack() -> QStackedWidget:
    stack = QStackedWidget()
    stack.addWidget(WaveformWidget())
    stack.addWidget(VectorscopeWidget())
    return stack


def test_the_plate_shows_only_while_it_holds_charts_and_no_tool_blocks_it(qapp) -> None:
    plate = AnalysisPlate()
    stack = _stack()
    assert plate.isHidden()
    plate.hold(stack)
    assert not plate.isHidden() and stack.parent() is plate
    plate.set_blocked(True)
    assert plate.isHidden()
    plate.set_blocked(False)
    plate.release()
    assert plate.isHidden() and not plate.holds_charts()


def test_a_chart_on_the_plate_paints_no_ground(qapp) -> None:
    plate = AnalysisPlate()
    stack = _stack()
    chart = stack.widget(0)
    chart.resize(120, 80)
    ground = chart.grab().toImage().pixelColor(60, 40).name()
    assert ground.upper() == THEME.bg_input.upper()

    plate.hold(stack)
    assert chart.property(ON_PLATE)
    plate.release()
    assert not chart.property(ON_PLATE)


def test_the_plate_at_rest_lets_the_picture_through_under_the_app_sheet(qapp) -> None:
    canvas = QWidget()
    canvas.setStyleSheet("QWidget { background-color: #ffffff; }")
    canvas.resize(400, 300)
    plate = AnalysisPlate(canvas)
    plate.move(0, 0)
    plate.hold(_stack())
    lightness = canvas.grab().toImage().pixelColor(8, 8).lightness()
    assert QColor(THEME.bg_input).lightness() + 20 < lightness < 235


def test_a_click_on_the_plate_asks_for_the_next_chart(qapp) -> None:
    plate = AnalysisPlate()
    plate.hold(_stack())
    asked = []
    plate.cycle_requested.connect(lambda: asked.append(1))
    QTest.mouseClick(plate, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    QTest.mouseClick(plate, Qt.MouseButton.RightButton, pos=QPoint(20, 20))
    assert asked == [1]


def test_the_well_takes_its_chart_back_in_first_place(qapp) -> None:
    stack, wedge = _stack(), WaveformWidget()
    well = ChartWell(stack, wedge)
    plate = AnalysisPlate()
    plate.hold(stack)
    plate.release()
    well.take_back()
    assert stack.parent() is well and well.layout().indexOf(stack) == 0


def _panel(saved: dict):
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    controller = MagicMock()
    controller.state = AppState()
    controller.session.repo.get_global_setting.side_effect = lambda key, default=None: saved.get(key, default)
    return RightPanel(controller)


def test_the_overlay_is_wanted_only_while_the_card_is_out_of_view(qapp) -> None:
    panel = _panel({"right_panel_group": "frame"})
    panel.analysis_section.set_expanded(True)
    assert not panel.analysis_overlay_wanted()
    panel.analysis_overlay_btn.setChecked(True)
    assert not panel.analysis_overlay_wanted()
    panel.analysis_section.set_expanded(False)
    assert panel.analysis_overlay_wanted()
    panel.analysis_section.set_expanded(True)
    panel._switch_group(panel._group_keys.index("library"))
    assert panel.analysis_overlay_wanted()


def test_the_panel_cycles_through_every_chart(qapp) -> None:
    panel = _panel({})
    seen = []
    for _ in range(panel.view_btn.count()):
        seen.append(panel.view_btn.currentData())
        panel.cycle_analysis_view()
    assert seen == ["curve", "waveform", "parade", "vectorscope"]
    assert panel.view_btn.currentData() == "curve"
