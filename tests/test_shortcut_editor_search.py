from negpy.desktop.view.shortcut_editor_search import (
    action_id_for_binding,
    build_shortcut_editor_targets,
    filter_targets,
    target_ids_for_binding,
)
from negpy.desktop.view.shortcut_registry import default_bindings


def test_build_targets_includes_action_description():
    targets = build_shortcut_editor_targets()
    export = next(t for t in targets if t.target_id == "export")
    assert export.label == "Export"
    assert "export" in export.search_text


def test_build_targets_includes_slider_group_label():
    targets = build_shortcut_editor_targets()
    density = next(t for t in targets if t.target_id == "density")
    assert density.label == "Print Density ↑/↓"
    assert density.row_kind == "slider"
    assert "density" in density.search_text


def test_search_matches_current_key_binding():
    targets = build_shortcut_editor_targets({"density_up": "Q", "density_down": "A"})
    matches = filter_targets(targets, "q")
    assert any(t.target_id == "density" for t in matches)


def test_search_g_matches_binding_without_action_id_noise():
    targets = build_shortcut_editor_targets(default_bindings())
    matches = filter_targets(targets, "g")
    assert [t.target_id for t in matches] == ["temperature"]


def test_search_alt_t_matches_toe_binding():
    targets = build_shortcut_editor_targets(default_bindings())
    matches = filter_targets(targets, "alt+t")
    assert [t.target_id for t in matches] == ["toe"]


def test_search_matches_category_name():
    targets = build_shortcut_editor_targets()
    matches = filter_targets(targets, "finishing")
    assert any(t.target_id == "border_size" for t in matches)


def test_search_matches_ctrl_binding():
    targets = build_shortcut_editor_targets({"export": "Ctrl+E"})
    matches = filter_targets(targets, "ctrl+e")
    assert any(t.target_id == "export" for t in matches)


def test_target_ids_for_binding_matches_slider_row():
    targets = build_shortcut_editor_targets({"toe_inc": "Alt+T", "toe_dec": "Alt+Shift+T"})
    assert target_ids_for_binding(targets, "Alt+T") == ["toe"]


def test_action_id_for_binding_resolves_inc_action():
    bindings = {"toe_inc": "Alt+T", "toe_dec": "Alt+Shift+T"}
    assert action_id_for_binding(bindings, "Alt+T") == "toe_inc"
