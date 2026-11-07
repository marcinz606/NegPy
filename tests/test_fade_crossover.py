"""The Fade Restoration crossover table, checked on films from `fade_crossover_scene`, whose per-layer
characteristic curves are independent of the table's band model."""

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from fade_crossover_scene import (
    CROSSOVER,
    REFERENCE,
    build_scene,
    chroma,
    estimate,
    hue,
    linear_fade,
    ramp_chroma,
    region_lab,
    render,
    scan_image,
    survival_changes,
    to_lab,
)
from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.fade import (
    CROSSOVER_LIMIT,
    CROSSOVER_WEIGHTS,
    crossover_bands,
    crossover_offsets,
    measure_crossover,
)
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.services.rendering.engine import base_processor


@lru_cache(maxsize=None)
def _case(film_key: str, subject: bool = False, groups: bool = False):
    film = {"crossover": CROSSOVER, "linear": linear_fade(REFERENCE, (1.0, 0.7, 0.5)), "reference": REFERENCE}[film_key]
    scene = build_scene(subject=subject, opposite_groups=groups)
    image = scan_image(film, scene)
    return scene, image, estimate(image)


@lru_cache(maxsize=None)
def _lab(film_key: str, crossover: float, subject: bool = False, groups: bool = False) -> np.ndarray:
    scene, image, est = _case(film_key, subject, groups)
    return to_lab(render(image, survival_changes(est, crossover)))


def test_leans_and_offsets_cancel_under_the_weights():
    _, _, est = _case("crossover")
    rows = crossover_bands(est.crossover)
    w = np.asarray(CROSSOVER_WEIGHTS)
    assert np.abs(rows[:, 2:] @ w).max() < 1e-4
    for strength in (1.0, 0.5):
        offsets = crossover_offsets(est.crossover, strength)
        assert offsets is not None
        assert np.abs(offsets.bins.astype(np.float64) @ w).max() < 1e-6
        assert np.abs(offsets.bins).max() <= CROSSOVER_LIMIT + 1e-6


def test_the_table_keeps_each_pixels_mean_density():
    _, _, est = _case("crossover")
    from negpy.features.process.fade import apply_crossover

    rng = np.random.default_rng(1)
    img = rng.uniform(-3.0, 0.0, (32, 32, 3)).astype(np.float32)
    out = apply_crossover(img, crossover_offsets(est.crossover, 1.0))
    w = np.asarray(CROSSOVER_WEIGHTS, dtype=np.float32)
    assert np.abs(out @ w - img @ w).max() < 1e-5
    assert np.abs(out - img).max() > 0.05


def test_a_linear_fade_leaves_the_table_near_identity():
    """The survivals fit a fade that is linear in density exactly, so no residual remains to fix."""
    _, _, est = _case("linear")
    assert est.green == pytest.approx(0.7, abs=0.01) and est.blue == pytest.approx(0.5, abs=0.01)
    assert np.abs(crossover_bands(est.crossover)[:, 2:]).max() < 0.015
    assert np.abs(_lab("linear", 1.0) - _lab("linear", 0.0)).max() < 1.5


def test_the_table_removes_a_density_dependent_crossover():
    scene, _, _ = _case("crossover")
    worst = {x: max(m for m, _ in ramp_chroma(_lab("crossover", x), scene)) for x in (0.0, 0.5, 1.0)}
    assert worst[1.0] < 0.35 * worst[0.0]
    assert worst[1.0] < worst[0.5] < worst[0.0]


def test_the_table_is_zero_where_the_survivals_were_fit():
    """The reference is the lean at the Estimate's midtone, so the offset row at that mean log is zero:
    the survivals own the cast there."""
    from negpy.features.process.fade import crossover_zero

    _, _, est = _case("crossover")
    offsets = crossover_offsets(est.crossover, 1.0)
    pos = (crossover_zero(est.crossover) - offsets.lo) / (offsets.hi - offsets.lo) * (len(offsets.bins) - 1)
    i = int(pos)
    row = offsets.bins[i] * (1 - (pos - i)) + offsets.bins[i + 1] * (pos - i)
    assert np.abs(row).max() < 0.01


def test_a_crossover_under_the_floor_applies_nothing():
    from negpy.features.process.fade import CROSSOVER_FLOOR, crossover_amount

    _, _, est = _case("linear")
    assert crossover_amount(est.crossover) < CROSSOVER_FLOOR
    assert crossover_offsets(est.crossover, 1.0) is None


def test_the_table_acts_at_crossover_s_strength():
    from negpy.features.process.fade import crossover_strength

    _, _, est = _case("crossover")
    negpy = replace(ProcessConfig(), process_mode=ProcessMode.E6, **survival_changes(est, 1.0))
    assert crossover_strength(negpy) == 1.0
    assert crossover_strength(replace(negpy, fade_crossover=0.0)) == 0.0


def test_a_large_saturated_subject_is_not_neutralized():
    """The red subject fills much of the frame and one density band; its low neutral weight and the
    mean-shift keep the band's lean on the grays."""
    _, _, plain = _case("crossover")
    scene, _, with_subject = _case("crossover", subject=True)
    a, b = crossover_bands(plain.crossover), crossover_bands(with_subject.crossover)
    assert a.shape == b.shape
    assert np.abs(a[:, 2:] - b[:, 2:]).max() < 0.05
    before = region_lab(_lab("crossover", 0.0, subject=True), scene.subject)
    after = region_lab(_lab("crossover", 1.0, subject=True), scene.subject)
    assert abs(chroma(after) - chroma(before)) < 6.0
    assert abs(hue(after) - hue(before)) < 3.0


def test_opposite_groups_at_one_density_follow_the_larger():
    """Two near-neutral groups lean opposite ways in one band: the mean-shift settles the band's centre
    on the larger (magenta: green below the band's mean log)."""
    from negpy.features.process.fade import neutral_bands

    scene, image, _ = _case("reference", groups=True)
    big, small = scene.groups
    grid = np.log10(np.clip(image, 1e-6, 1.0))
    bands = neutral_bands(grid)
    big_level = float(np.log10(image[big]).mean())
    k = int(np.argmin(np.abs(bands.level - big_level)))
    lean = bands.centre[k] - bands.level[k]
    assert lean[1] < -0.02


def test_a_frame_with_no_fade_takes_no_table():
    """Estimate finds nothing to restore on the unfaded film, so the survivals stay at 1 and the table
    stays off whatever the strength: the groups keep their colour."""
    scene, _, est = _case("reference", groups=True)
    assert est.red is None
    big, small = scene.groups
    lab0, lab1 = _lab("reference", 0.0, groups=True), _lab("reference", 1.0, groups=True)
    assert np.array_equal(lab0, lab1)


def _base(process: ProcessConfig, image: np.ndarray):
    settings = replace(WorkspaceConfig(), process=process)
    h, w = image.shape[:2]
    ctx = PipelineContext(original_size=(h, w), scale_factor=1.0, process_mode=ProcessMode.E6)
    out = base_processor(settings).process(image, ctx)
    return np.asarray(out), ctx


def test_a_render_measures_a_missing_table_as_the_estimate_does():
    _, image, est = _case("crossover")
    tagged = replace(ProcessConfig(), process_mode=ProcessMode.E6, **survival_changes(est, 1.0))
    _, ctx = _base(replace(tagged, fade_crossover_table=None), image)
    table, space = ctx.metrics["fade_crossover"]
    assert space == tagged.fade_crossover_space
    # The render reads the zero from its own grays, the Estimate from the survivals' own midtone.
    assert abs(table[0] - est.crossover[0]) < 0.03
    assert np.abs(np.asarray(table[1:]) - np.asarray(est.crossover[1:])).max() < 2e-3
    _, ctx = _base(tagged, image)
    assert "fade_crossover" not in ctx.metrics


def test_crossover_at_zero_reads_nothing_and_changes_nothing():
    _, image, est = _case("crossover")
    process = replace(ProcessConfig(), process_mode=ProcessMode.E6, **survival_changes(est, 0.0))
    stored, ctx = _base(process, image)
    bare, _ = _base(replace(process, fade_crossover_table=None, fade_crossover_space=""), image)
    assert "fade_crossover" not in ctx.metrics
    assert np.array_equal(stored, bare)


def test_a_print_takes_no_table():
    _, image, est = _case("crossover")
    process = replace(ProcessConfig(), process_mode=ProcessMode.C41, fade_crossover=1.0)
    _, ctx = _base(process, image)
    assert "fade_crossover" not in ctx.metrics


def test_measure_needs_cells():
    assert measure_crossover(np.zeros((4, 4, 3), np.float32) - 1.0) is None


def test_the_table_survives_a_flat_round_trip():
    _, _, est = _case("crossover")
    config = replace(
        WorkspaceConfig(),
        process=replace(ProcessConfig(), fade_crossover=0.5, fade_crossover_table=est.crossover, fade_crossover_space="abc"),
    )
    back = WorkspaceConfig.from_flat_dict(config.to_dict())
    assert back.process.fade_crossover_table == est.crossover
    assert back.process.fade_crossover == 0.5 and back.process.fade_crossover_space == "abc"


def test_one_engine_follows_a_strength_change():
    """The base stage is cached per config: the crossover fields are in its key."""
    from negpy.services.rendering.engine import DarkroomEngine

    _, image, est = _case("crossover")
    engine = DarkroomEngine()
    renders = []
    for strength in (0.0, 1.0):
        process = replace(ProcessConfig(), process_mode=ProcessMode.E6, **survival_changes(est, strength))
        renders.append(np.asarray(engine.process(image, replace(WorkspaceConfig(), process=process), source_hash="same")))
    assert np.abs(renders[1] - renders[0]).max() > 0.05


def test_a_new_meter_area_measures_the_table_again():
    from negpy.features.process.fade import crossover_space, fade_density

    process = replace(ProcessConfig(), process_mode=ProcessMode.E6, fade_ratio_g=0.8, fade_crossover=1.0)
    density = fade_density(process, ProcessMode.E6, "")
    tag = crossover_space(density, "", process)
    assert crossover_space(density, "", replace(process, analysis_rect=(0.1, 0.1, 0.5, 0.5))) != tag
    assert crossover_space(density, "", replace(process, analysis_buffer=process.analysis_buffer + 0.05)) != tag
    assert crossover_space(density, "", replace(process, meter_rect=(0.2, 0.2, 0.3, 0.3))) != tag


def test_paste_carries_the_table_with_the_strength():
    from negpy.desktop.settings_catalog import CATALOG, apply_selected_fields

    rows = [r for _, section in CATALOG for r in section if "fade_ratio_r" in r.fields]
    source = replace(
        WorkspaceConfig(),
        process=replace(ProcessConfig(), fade_crossover=0.7, fade_crossover_table=(1.0, 2.0, 3.0, 4.0, 5.0), fade_crossover_space="src"),
    )
    target = replace(
        WorkspaceConfig(), process=replace(ProcessConfig(), fade_crossover_table=(6.0, 7.0, 8.0, 9.0, 10.0), fade_crossover_space="dst")
    )
    out = apply_selected_fields(source, target, rows).process
    assert out.fade_crossover == 0.7
    assert (out.fade_crossover_table, out.fade_crossover_space) == ((1.0, 2.0, 3.0, 4.0, 5.0), "src")


def test_roll_pushes_the_table_with_the_strength():
    from negpy.services.assets.rolls import ROLL_DEFAULT_FIELDS

    fields = ROLL_DEFAULT_FIELDS["fade"][1]
    assert {"fade_crossover", "fade_crossover_table", "fade_crossover_space"} <= set(fields)


def test_a_stored_table_applies_under_another_tag():
    from negpy.features.process.fade import resolve_crossover

    def grid():
        raise AssertionError("measured")

    process = replace(
        ProcessConfig(process_mode=ProcessMode.E6),
        fade_ratio_g=0.9,
        fade_crossover=1.0,
        fade_crossover_table=(0.1, 0.2),
        fade_crossover_space="a",
    )
    assert resolve_crossover(process, "b", grid) == ((0.1, 0.2), None)


def test_a_survival_set_by_hand_keeps_the_estimates_grays():
    """With the Estimate's survivals the table is the Estimate's own; with NC's grays it keeps their zero,
    where a render's measurement would take NegPy's."""
    from fade_crossover_scene import estimate_grid

    from fade_crossover_scene import estimate_fade
    from negpy.features.process.fade import gray_refs, survival_crossover

    _, image, _ = _case("crossover")
    grid = estimate_grid(image)
    for mix in (0.0, 1.0):
        est = estimate_fade(grid, mix=mix)
        table = survival_crossover(grid, lambda g, m=mix: gray_refs(g, m), (est.red, est.green, est.blue))
        assert table == est.crossover
    lowered = (0.8 * est.red, est.green, est.blue)
    nc = survival_crossover(grid, lambda g: gray_refs(g, 1.0), lowered)
    negpy = survival_crossover(grid, lambda g: gray_refs(g, 0.0), lowered)
    assert nc[1:] == negpy[1:] and nc[0] != negpy[0]
