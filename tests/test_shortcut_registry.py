from negpy.desktop.view.shortcut_registry import default_bindings, load_bindings, merge_bindings, save_bindings, tooltip_with_shortcut


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
