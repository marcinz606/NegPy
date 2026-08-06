"""The printing record's text: stops are written in the exposure domain a printer uses,
so a burn (which adds exposure) reads +, a dodge reads −."""

from dataclasses import replace

from negpy.features.exposure.models import ExposureConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.local.models import LocalAdjustmentsConfig, PolygonMask
from negpy.services.view.printing_notes import mask_notes, recipe_lines, stops_label

SQUARE = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))


def _local(*stops: float) -> LocalAdjustmentsConfig:
    """Masks by print exposure: positive burns, negative dodges."""
    return LocalAdjustmentsConfig(masks=tuple(PolygonMask(vertices=SQUARE, stops=s) for s in stops))


def test_stops_are_written_as_darkroom_fractions() -> None:
    assert stops_label(1.0) == "+1"
    assert stops_label(0.5) == "+½"
    assert stops_label(-1.0 / 3.0) == "−⅓"
    assert stops_label(1.25) == "+1¼"
    assert stops_label(0.0) == "0"


def test_an_unfractional_value_falls_back_to_decimals() -> None:
    assert stops_label(0.20) == "+0.20"
    assert stops_label(-0.15) == "−0.15"
    # A slider step of 0.05 never lands on an exact third, so near-thirds still get the glyph.
    assert stops_label(0.35) == "+⅓"


def test_a_burn_reads_plus_and_a_dodge_reads_minus() -> None:
    burn, dodge = mask_notes(_local(0.5, -0.5))

    assert (burn.is_burn, burn.kind, burn.stops) == (True, "Burn", "+½")
    assert (dodge.is_burn, dodge.kind, dodge.stops) == (False, "Dodge", "−½")
    assert (burn.number, dodge.number) == (1, 2)


def test_the_recipe_keeps_quiet_about_defaults() -> None:
    lines = recipe_lines(ExposureConfig(), LocalAdjustmentsConfig(), FinishConfig(), frame="roll1_04.tif")

    assert lines[0] == "roll1_04.tif"
    assert any("Neutral" in line for line in lines)
    assert any(line.startswith("Print Density 1.00") for line in lines)
    assert any(line.startswith("Grade ISO-R 115") for line in lines)
    assert not [line for line in lines if line.startswith(("Filtration", "Toe", "Snap", "Edge burn", "Dodge & burn", "Zone density"))]


def test_the_recipe_reports_every_decision_that_moved() -> None:
    exposure = replace(
        ExposureConfig(),
        density=1.2,
        grade=95.0,
        shadow_grade=-12.0,
        highlight_grade=8.0,
        shadow_density=0.3,
        wb_magenta=0.1,
        toe=0.4,
        midtone_gamma=0.2,
        auto_exposure=False,
        auto_normalize_contrast=False,
    )
    lines = recipe_lines(exposure, _local(1.0, -0.25), replace(FinishConfig(), vignette_stops=0.5))
    joined = "\n".join(lines)

    assert "Print Density 1.20" in joined and "(auto)" not in joined
    assert "Grade ISO-R 95 · split -12/+8" in joined
    assert "Zone density: shadows +0.30 · highlights +0.00" in joined
    assert "Filtration C+0.00 M+0.10 Y+0.00" in joined
    assert "Toe +0.40 · Shoulder +0.00" in joined
    assert "Snap +0.20" in joined
    assert "Edge burn +½ stop" in joined
    assert "Dodge & burn: 1 Burn +1 · 2 Dodge −¼" in joined


def test_auto_flags_are_marked() -> None:
    lines = recipe_lines(ExposureConfig(), LocalAdjustmentsConfig(), FinishConfig())

    assert "Print Density 1.00 (auto)" in lines
    assert "Grade ISO-R 115 (auto)" in lines


def test_a_local_grade_is_written_as_the_grade_it_prints_at() -> None:
    local = LocalAdjustmentsConfig(masks=(PolygonMask(vertices=SQUARE, stops=1.0, grade=-20.0),))
    (note,) = mask_notes(local, grade=115.0)

    assert note.local_r == "R95"
    assert note.badge == "1 +1 R95"
    assert note.summary == "1 Burn +1 @ R95"


def test_a_grade_only_mask_is_neither_a_dodge_nor_a_burn() -> None:
    local = LocalAdjustmentsConfig(masks=(PolygonMask(vertices=SQUARE, stops=0.0, grade=30.0),))
    (note,) = mask_notes(local, grade=115.0)

    assert note.kind == "Grade"
    assert note.badge == "1 R145"
    assert note.summary == "1 Grade @ R145"


def test_a_local_grade_off_the_ladder_is_written_clamped() -> None:
    local = LocalAdjustmentsConfig(masks=(PolygonMask(vertices=SQUARE, stops=0.0, grade=-90.0),))
    (note,) = mask_notes(local, grade=115.0)

    assert note.local_r == "R50"


def test_masks_at_the_frame_grade_carry_no_grade_note() -> None:
    (note,) = mask_notes(_local(1.0), grade=115.0)

    assert note.local_r == ""
    assert note.badge == "1 +1"
    assert note.summary == "1 Burn +1"


def test_the_record_names_each_mask_grade() -> None:
    local = LocalAdjustmentsConfig(
        masks=(
            PolygonMask(vertices=SQUARE, stops=1.0, grade=-20.0),
            PolygonMask(vertices=SQUARE, stops=-0.25),
        )
    )
    lines = recipe_lines(replace(ExposureConfig(), grade=115.0), local, FinishConfig())

    assert "Dodge & burn: 1 Burn +1 @ R95 · 2 Dodge −¼" in "\n".join(lines)
