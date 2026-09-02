"""A size reads in the unit that actually keeps it under 1024."""

import pytest

from negpy.kernel.system.text import human_bytes


@pytest.mark.parametrize(
    "count,expected",
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1048576, "1.0 MB"),
        (1073741824, "1.0 GB"),
    ],
)
def test_each_unit_starts_at_its_own_threshold(count, expected):
    assert human_bytes(count) == expected


@pytest.mark.parametrize(
    "count,expected",
    [
        (1048575, "1.0 MB"),
        (1073741823, "1.0 GB"),
    ],
)
def test_a_size_just_under_the_threshold_moves_up(count, expected):
    """One byte short of the next unit still rounds to 1024.0 at one decimal
    place, so the displayed number has to decide the unit, not the raw value."""
    assert human_bytes(count) == expected


def test_gb_is_the_last_unit():
    """There is no TB, so a value past 1024 GB stays in GB rather than
    silently losing its magnitude."""
    assert human_bytes(1024 * 1024 * 1024 * 1024) == "1024.0 GB"
