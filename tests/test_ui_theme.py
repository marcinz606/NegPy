import dataclasses

import pytest

from negpy.desktop.view.styles import theme
from negpy.desktop.view.styles.theme import THEME, UI_THEMES, ThemeConfig, apply_theme


@pytest.fixture(autouse=True)
def _restore_default_theme():
    yield
    apply_theme("black")


def test_black_leaves_every_token_at_its_default():
    apply_theme("mid_gray")
    apply_theme("black")
    assert dataclasses.asdict(THEME) == dataclasses.asdict(ThemeConfig())


def test_an_unknown_key_falls_back_to_black():
    apply_theme("light_gray")
    apply_theme("no_such_theme")
    assert dataclasses.asdict(THEME) == dataclasses.asdict(ThemeConfig())


@pytest.mark.parametrize("key", ["mid_gray", "light_gray"])
def test_a_gray_theme_retints_the_surfaces_in_place(key: str):
    before = THEME
    apply_theme(key)
    assert THEME is before
    assert THEME.bg_dark != ThemeConfig().bg_dark
    assert THEME.bg_dark == theme._THEME_OVERRIDES[key]["bg_dark"]


def test_switching_themes_does_not_leak_the_previous_override():
    apply_theme("light_gray")
    assert THEME.text_secondary == "#FFFFFF"
    apply_theme("mid_gray")
    assert THEME.text_secondary == ThemeConfig().text_secondary


def test_every_override_names_a_real_token():
    tokens = {f.name for f in dataclasses.fields(ThemeConfig)}
    for key, overrides in theme._THEME_OVERRIDES.items():
        assert set(overrides) <= tokens, key


def test_every_listed_theme_resolves():
    assert {key for key, _ in UI_THEMES} == {"black", *theme._THEME_OVERRIDES}
