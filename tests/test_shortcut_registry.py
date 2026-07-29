from negpy.desktop.view.shortcut_registry import (
    REGISTRY,
    EditorRowSlider,
    category_editor_rows,
    default_bindings,
    default_slider_steps,
    load_bindings,
    load_slider_steps,
    merge_bindings,
    save_bindings,
    save_slider_steps,
    tooltip_with_shortcut,
)
from negpy.desktop.view.slider_shortcut_groups import SLIDER_GROUP_BY_ACTION, SLIDER_GROUPS


class _Repo:
    def __init__(self):
        self.data = {}

    def get_global_setting(self, key, default=None):
        return self.data.get(key, default)

    def save_global_setting(self, key, value):
        self.data[key] = value


def test_merge_bindings_applies_known_overrides_only():
    bindings = merge_bindings({"density_up": "Ctrl+Alt+D", "unknown": "Ctrl+U"})

    assert bindings["density_up"] == "Ctrl+Alt+D"
    assert "unknown" not in bindings


def test_save_bindings_only_persists_overrides():
    repo = _Repo()
    bindings = default_bindings()
    bindings["density_up"] = "Ctrl+Alt+D"

    save_bindings(repo, bindings)

    assert repo.data["shortcut_bindings"] == {"density_up": "Ctrl+Alt+D"}


def test_load_bindings_merges_saved_overrides():
    repo = _Repo()
    repo.data["shortcut_bindings"] = {"density_up": "Ctrl+Alt+D"}

    bindings = load_bindings(repo)

    assert bindings["density_up"] == "Ctrl+Alt+D"
    assert bindings["grade_up"] == default_bindings()["grade_up"]


def test_cyan_defaults_empty_but_bindable():
    # #406: Cyan ships with no default binding, yet stays assignable via the editor.
    defaults = default_bindings()
    assert defaults["cyan_inc"] == ""
    assert defaults["cyan_dec"] == ""

    bindings = merge_bindings({"cyan_inc": "Alt+C", "cyan_dec": "Alt+Shift+C"})
    assert bindings["cyan_inc"] == "Alt+C"
    assert bindings["cyan_dec"] == "Alt+Shift+C"


def test_tooltip_with_multiple_shortcuts_renders_all_keys():
    tooltip = tooltip_with_shortcut("Density", ["density_up", "density_down"], {"density_up": "Q", "density_down": "A"})

    assert "Density" in tooltip
    assert "Q" in tooltip
    assert "A" in tooltip


def test_tooltip_places_shortcut_on_its_own_right_aligned_line():
    tooltip = tooltip_with_shortcut("Density up", "density_up", {"density_up": "Q"})

    # Shortcut sits on its own right-aligned line below the text, as a bordered keycap.
    text_part, sep, shortcut_part = tooltip.partition('<table align="right"')
    assert text_part == "Density up"
    assert sep == '<table align="right"'
    # The key renders as a bordered <td> so it reads like a physical keyboard key.
    assert "border:1px solid" in shortcut_part
    assert "<td" in shortcut_part
    assert "Q" in shortcut_part


def test_tooltip_joins_two_shortcuts_with_ampersand():
    tooltip = tooltip_with_shortcut("Density", ["density_up", "density_down"], {"density_up": "Q", "density_down": "A"})

    assert '<table align="right"' in tooltip
    assert "&amp;" in tooltip


def test_tooltip_without_binding_returns_plain_text():
    tooltip = tooltip_with_shortcut("Cyan up", "cyan_inc", {"cyan_inc": ""})

    assert tooltip == "Cyan up"
    assert "<div" not in tooltip


def test_default_slider_steps_match_current_keyboard_behavior():
    steps = default_slider_steps()
    assert steps["density"] == 0.01
    assert steps["grade"] == 10.0
    assert steps["offset"] == 1.0
    assert steps["temperature"] == 50.0
    assert len(steps) == len(SLIDER_GROUPS)


def test_save_slider_steps_only_persists_overrides():
    repo = _Repo()
    steps = default_slider_steps()
    steps["density"] = 0.05

    save_slider_steps(repo, steps)

    assert repo.data["shortcut_slider_steps"] == {"density": 0.05}


def test_load_slider_steps_merges_saved_overrides():
    repo = _Repo()
    repo.data["shortcut_slider_steps"] = {"grade": 5.0}

    steps = load_slider_steps(repo)

    assert steps["grade"] == 5.0
    assert steps["density"] == 0.01


def test_category_editor_rows_merge_slider_pairs():
    exposure_items = [(action_id, entry) for action_id, entry in REGISTRY.items() if entry.category == "Exposure"]
    rows = category_editor_rows(exposure_items)
    labels = [row.group.label if isinstance(row, EditorRowSlider) else row.entry.description for row in rows]

    assert "Density ↑/↓" in labels
    assert "Density up" not in labels
    assert "Density down" not in labels
    assert "Magenta ↑/↓" in labels


def test_no_two_actions_claim_the_same_default_key():
    # Two actions on one key makes Qt fire activatedAmbiguously and both go dead.
    keys = [key for key in default_bindings().values() if key]
    assert len(keys) == len(set(keys))


def test_every_slider_action_has_a_group():
    slider_actions = {group.inc_action for group in SLIDER_GROUPS} | {group.dec_action for group in SLIDER_GROUPS}
    for action_id in slider_actions:
        assert action_id in REGISTRY
        assert action_id in SLIDER_GROUP_BY_ACTION
