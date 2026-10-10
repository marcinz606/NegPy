"""The Analysis step wedge printed through the render's own alternative process and toning."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.altprocess.models import AltProcess
from negpy.features.exposure.analysis import wedge_vals
from negpy.features.exposure.logic import curve_params_from_metrics, print_curve, print_curve_output
from negpy.features.process.models import ProcessMode
from negpy.services.rendering import engine
from negpy.services.rendering.engine import DarkroomEngine, print_wedge


def _bw(**altproc) -> WorkspaceConfig:
    base = WorkspaceConfig()
    cfg = replace(base, process=replace(base.process, process_mode=ProcessMode.BW))
    return replace(cfg, altproc=replace(cfg.altproc, **altproc)) if altproc else cfg


def _curve(cfg: WorkspaceConfig) -> np.ndarray:
    mode = cfg.process.process_mode
    slopes, pivots, _ = curve_params_from_metrics(cfg.exposure, mode, {})
    return print_curve_output(print_curve(cfg.exposure, slopes[1], pivots[1], mode), wedge_vals())


def test_untoned_wedge_is_the_gray_curve() -> None:
    for cfg in (WorkspaceConfig(), _bw()):
        curve = _curve(cfg)
        rgb = print_wedge(cfg, curve)
        assert rgb.shape == (len(curve), 3)
        np.testing.assert_allclose(rgb, np.repeat(np.clip(curve, 0, 1)[:, None], 3, axis=1), atol=1e-4)


def test_cyanotype_prints_a_blue_wedge() -> None:
    cfg = _bw(alt_process=AltProcess.CYANOTYPE)
    mid = print_wedge(cfg, _curve(cfg))[5:15]
    assert np.all(mid[:, 2] > mid[:, 0])


def test_split_toning_tints_the_ends_apart() -> None:
    cfg = _bw()
    cfg = replace(
        cfg,
        toning=replace(cfg.toning, shadow_tint_hue=250.0, shadow_tint_strength=0.8, highlight_tint_hue=60.0, highlight_tint_strength=0.6),
    )
    rgb = print_wedge(cfg, _curve(cfg))
    assert rgb[2, 2] > rgb[2, 0]  # cool shadows
    assert rgb[-3, 0] > rgb[-3, 2]  # warm highlights


def test_sabattier_folds_the_light_end_back() -> None:
    cfg = _bw(alt_process=AltProcess.SABATTIER)
    luma = print_wedge(cfg, _curve(cfg)).mean(axis=1)
    assert luma.argmax() < len(luma) - 1


def test_alt_process_is_inert_on_color_film() -> None:
    base = WorkspaceConfig()
    cfg = replace(base, altproc=replace(base.altproc, alt_process=AltProcess.CYANOTYPE))
    curve = _curve(cfg)
    np.testing.assert_allclose(print_wedge(cfg, curve), print_wedge(base, curve), atol=1e-6)


def test_the_render_runs_the_wedges_print_stages() -> None:
    frame = np.repeat(np.linspace(0.05, 0.9, 32, dtype=np.float32)[None, :, None], 32, axis=0).repeat(3, axis=2)
    with patch.object(engine, "print_stages", wraps=engine.print_stages) as spy:
        DarkroomEngine().process(np.ascontiguousarray(frame), _bw(), source_hash="print-stages")
    assert spy.called


def test_the_wedge_strip_stays_on_the_serial_kernels() -> None:
    """The wedge runs on the GUI thread: a strip at or over SERIAL_MAX_ELEMENTS would take the
    parallel kernels' lock and wait behind the render thread's full-frame kernels."""
    from negpy.features.exposure.analysis import WEDGE_STEPS
    from negpy.kernel.system.parallel import SERIAL_MAX_ELEMENTS

    assert engine._WEDGE_PATCH_PX**2 * WEDGE_STEPS * 3 < SERIAL_MAX_ELEMENTS
