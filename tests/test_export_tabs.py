"""Export's sub-tabs: the form, the proof, the presets and the side jobs each on a tab, with
the Export button pinned under all of them."""

from negpy.desktop.view.sidebar.export import ExportSidebar
from tests.conftest import FakeController, FakeRepo


def _sidebar() -> ExportSidebar:
    return ExportSidebar(FakeController(repo=FakeRepo()))


def _page(sidebar: ExportSidebar, key: str):
    return sidebar.tabs.stack.widget(sidebar.tabs.keys.index(key))


def test_two_named_tabs_then_two_icon_tabs(qapp) -> None:
    sidebar = _sidebar()

    assert [btn.text() for btn in sidebar.tabs.buttons] == ["File", "Proof", "Presets", "Alongside Exports"]
    assert [btn._icon_only for btn in sidebar.tabs.buttons] == [False, False, True, True]


def test_each_card_sits_on_its_tab(qapp) -> None:
    sidebar = _sidebar()

    assert _page(sidebar, "file").isAncestorOf(sidebar.form)
    assert _page(sidebar, "proof").isAncestorOf(sidebar._soft_proof_section)
    assert _page(sidebar, "presets").isAncestorOf(sidebar._presets_section)
    assert _page(sidebar, "alongside").isAncestorOf(sidebar.contact_sheet_section)
    assert _page(sidebar, "alongside").isAncestorOf(sidebar.alongside_section)


def test_the_export_button_is_pinned_outside_the_tabs(qapp) -> None:
    sidebar = _sidebar()

    assert sidebar.isAncestorOf(sidebar.export_main_btn)
    assert not sidebar.tabs.isAncestorOf(sidebar.export_main_btn)


def test_the_proof_tab_is_marked_while_the_proof_is_on(qapp) -> None:
    sidebar = _sidebar()
    sidebar.state.soft_proof_enabled = True
    sidebar.sync_ui()
    assert sidebar.tabs.button("proof").is_marked()

    sidebar.state.soft_proof_enabled = False
    sidebar.sync_ui()
    assert not sidebar.tabs.button("proof").is_marked()


def test_linear_grays_out_presets_and_hides_bake_and_sidecars(qapp) -> None:
    sidebar = _sidebar()
    sidebar.intent_btn.setCurrentIndex(2)

    assert not sidebar.manage_presets_btn.isEnabled()
    assert not sidebar.export_presets_group.isEnabled()
    assert sidebar.presets_linear_hint.isVisibleTo(sidebar._presets_section)
    assert not sidebar.bake_on_export_btn.isVisibleTo(sidebar.alongside_section)
    assert not sidebar.sidecars_enabled_btn.isVisibleTo(sidebar.alongside_section)
    assert sidebar.printing_notes_btn.isVisibleTo(sidebar.alongside_section)

    sidebar.intent_btn.setCurrentIndex(0)
    assert sidebar.manage_presets_btn.isEnabled()
    assert not sidebar.presets_linear_hint.isVisibleTo(sidebar._presets_section)
    assert sidebar.bake_on_export_btn.isVisibleTo(sidebar.alongside_section)
