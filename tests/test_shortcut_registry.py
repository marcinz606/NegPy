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
