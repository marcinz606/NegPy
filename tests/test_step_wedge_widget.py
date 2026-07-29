"""Sidebar side of the step wedge: the Analysis strip under the curve chart. It shows the same
curve the chart plots, as tones, so it is fed from that chart's own solved slope and pivot and
hides whenever there is no print curve to wedge."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PyQt6.QtGui import QPainter, QPixmap

from negpy.domain.models import WorkspaceConfig
from negpy.desktop.view.sidebar.right_panel import RightPanel
from negpy.desktop.view.widgets.charts import StepWedgeWidget
from negpy.features.exposure.analysis import WEDGE_STEPS, wedge_step_density, wedge_vals
from negpy.features.exposure.logic import print_curve, print_curve_output
from negpy.features.exposure.models import ExposureConfig


def _wedge(width: int = 400, step_d: float = 0.15) -> StepWedgeWidget:
    widget = StepWedgeWidget()
    widget.resize(width, widget.height())
    config = ExposureConfig()
    enc = print_curve_output(print_curve(config, 3.026, 0.224, None), wedge_vals())
    widget.update_data(enc, step_d, "sRGB", None)
    return widget


def test_patch_bounds_tile_the_full_width_exactly() -> None:
    widget = _wedge()
    bounds = widget._patch_bounds()

    assert len(bounds) == WEDGE_STEPS
    # Integer-aligned and gapless: each patch starts where the previous one ended.
    for (_, a_end), (b_start, _) in zip(bounds, bounds[1:]):
        assert a_end == b_start
    assert bounds[0][0] == 0
    assert bounds[-1][1] == widget.width()


def test_patch_bounds_tile_a_width_that_does_not_divide_evenly() -> None:
    widget = _wedge(width=41)  # 41 / 21 is not an integer
    bounds = widget._patch_bounds()

    assert len(bounds) == WEDGE_STEPS
    assert all(end - start >= 1 for start, end in bounds)
    assert sum(end - start for start, end in bounds) == 41


def test_the_patches_use_the_canvas_display_transform() -> None:
    """Guards the double-proof trap: under a soft proof the render worker already baked the
    proof into the buffer, so the wedge must reuse the canvas's transform, not apply a second."""
    widget = StepWedgeWidget()
    with patch("negpy.desktop.converters.ImageConverter.to_qimage") as to_qimage:
        to_qimage.return_value = QPixmap(WEDGE_STEPS, 1).toImage()
        widget.update_data(np.linspace(0.0, 1.0, WEDGE_STEPS), 0.15, "Display P3", b"fake-icc")

    _buf, cs, icc = to_qimage.call_args[0]
    assert cs == "Display P3"
    assert icc == b"fake-icc"


def test_no_data_paints_an_empty_strip_rather_than_raising() -> None:
    widget = StepWedgeWidget()
    widget.resize(400, widget.height())
    widget.grab()  # a real paint pass; before the first render there is nothing to print
    assert widget._colors == []


def test_a_strip_narrower_than_the_step_count_draws_no_patches() -> None:
    widget = _wedge(width=12)  # 21 patches can't tile 12 px
    with patch.object(widget, "_patch_bounds") as spy:
        widget.grab()
    assert not spy.called


def test_the_density_axis_is_labelled_at_a_real_sidebar_width() -> None:
    """21 patches in a ~300 px sidebar are ~14 px each, so labelling every fifth step gated
    the axis off entirely and the strip lost its density units."""
    widget = _wedge(width=300, step_d=0.15)
    with patch.object(QPainter, "drawText") as draw_text:
        widget.grab()

    texts = [call[0][-1] for call in draw_text.call_args_list]
    assert texts == ["0.00", "1.50", "3.00"]


def test_every_label_box_lies_inside_the_strip() -> None:
    """drawText clips to the rect it is given, so a box hanging off the end silently loses
    characters. The end labels align to the strip edge the way an axis does; sliding their box
    inward instead would centre them over the wrong step."""
    widget = _wedge()
    with patch.object(QPainter, "drawText") as draw_text:
        widget.grab()

    boxes = [call[0][0] for call in draw_text.call_args_list]
    assert len(boxes) == 3
    for box in boxes:
        assert box.left() >= 0, f"label box starts off the left edge: {box}"
        assert box.right() <= widget.width(), f"label box runs past the right edge: {box}"
    # Still ordered left to right and not overlapping, so no label sits over another's step.
    for a, b in zip(boxes, boxes[1:]):
        assert a.right() < b.left()


def test_the_tooltip_reads_the_step_and_its_density() -> None:
    widget = _wedge(step_d=0.15)
    event = SimpleNamespace(position=lambda: SimpleNamespace(x=lambda: widget.width() * 2.5 / WEDGE_STEPS))
    with patch.object(StepWedgeWidget, "setToolTip") as set_tip:
        with patch("PyQt6.QtWidgets.QWidget.mouseMoveEvent"):
            widget.mouseMoveEvent(event)
    assert set_tip.call_args[0][0] == "Step 2 — density 0.30"


def _panel_stub(flat_peek: bool) -> MagicMock:
    panel = MagicMock()
    panel._clip_fracs = (None, None)
    panel.controller.state.flat_peek = flat_peek
    panel.controller.session.state.config = WorkspaceConfig()
    panel.controller.session.state.last_metrics = {}
    panel.controller.display_transform_params.return_value = ("sRGB", None)
    panel.step_wedge = StepWedgeWidget()
    return panel


def test_a_flat_peek_hides_the_wedge() -> None:
    """The flat intent bypasses the print curve, so there is nothing to print the wedge through
    and a stale strip would describe tones the canvas isn't showing."""
    panel = _panel_stub(flat_peek=True)
    RightPanel._update_analysis(panel)
    assert not panel.step_wedge.isVisibleTo(panel.step_wedge)


def test_the_wedge_is_fed_the_curve_the_chart_just_plotted() -> None:
    """One solve, two consumers: the chart and the wedge must not resolve slope and pivot
    separately, or Auto Grade could move one and not the other."""
    panel = _panel_stub(flat_peek=False)
    RightPanel._update_analysis(panel)

    config, mode, slope, pivot, metrics = panel._update_step_wedge.call_args[0]
    plotted = panel.curve_widget.update_curve.call_args
    assert slope == plotted.kwargs["slope"]
    assert pivot == plotted.kwargs["pivot"]
    assert mode is plotted.kwargs["process_mode"]
    assert metrics is panel.controller.session.state.last_metrics
    assert config is panel.controller.session.state.config.exposure


def test_the_wedge_labels_read_in_the_scans_own_density_units() -> None:
    """The geometry is a fixed 21 steps across val 0..1, so the physical truth rides in the
    labels: the step density is the scan's range spread over the 20 intervals."""
    panel = _panel_stub(flat_peek=False)
    panel.controller.session.state.last_metrics = {"norm_density_range": 3.0}
    with patch.object(StepWedgeWidget, "update_data") as update_data:
        RightPanel._update_step_wedge(
            panel,
            ExposureConfig(),
            None,
            3.026,
            0.224,
            panel.controller.session.state.last_metrics,
        )
    assert update_data.call_args[0][1] == wedge_step_density(3.0)
