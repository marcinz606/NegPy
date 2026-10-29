"""Language selection, format fallback and localized UI configuration boundaries."""

import ast
import hashlib
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
import subprocess
import sys

import pytest

from negpy.kernel.system import i18n
from negpy.kernel.system.text import count_of
from scripts.extract_translations import CATALOGS, ENGLISH_GUIDE, ROOT, extract


@pytest.fixture
def chinese() -> Iterator[None]:
    previous = i18n.language()
    i18n.configure("zh_CN")
    yield
    i18n.configure(previous)


@pytest.mark.parametrize(
    "choice,system,expected",
    [
        ("en", ["zh-CN"], "en"),
        ("zh_CN", ["en-US"], "zh_CN"),
        ("system", ["zh-Hans-CN"], "zh_CN"),
        ("system", ["en-US"], "en"),
        ("system", ["fr-FR"], "en"),
        ("system", ["zh-Hant-TW"], "en"),
        ("ja", ["en-US"], "ja"),
        ("ja_JP", ["en-US"], "ja"),
        ("system", ["ja-Jpan-JP"], "ja"),
        ("ko", ["en-US"], "ko"),
        ("KO_KR", ["en-US"], "ko"),
        ("system", ["ko-KR"], "ko"),
        ("de", ["en-US"], "de"),
        ("DE_AT", ["en-US"], "de"),
        ("system", ["de-DE"], "de"),
        ("system", [], "en"),
        ("invalid", ["zh-CN"], "en"),
    ],
)
def test_language_resolution(choice: str, system: list[str], expected: str) -> None:
    assert i18n.resolve_language(choice, system) == expected


_FIX = "Run `make translations`, then fill every registered catalog (.claude/skills/update-translations/SKILL.md)."


@pytest.mark.parametrize("locale", CATALOGS)
def test_catalog_covers_every_marked_message_and_preserves_formats(locale: str) -> None:
    catalog = json.loads(CATALOGS[locale].read_text(encoding="utf-8"))
    missing, obsolete = sorted(extract() - catalog.keys()), sorted(catalog.keys() - extract())
    assert not missing and not obsolete, f"{locale}.json misses {missing} and keeps obsolete {obsolete}. {_FIX}"
    for source, translated in catalog.items():
        assert translated, f"{locale}.json has no translation for {source!r}. {_FIX}"
        assert i18n.fields(source) == i18n.fields(translated), f"{locale}.json changes the placeholders of {source!r}"
        tags = r"</?[A-Za-z][^>]*>"
        assert re.findall(tags, source) == re.findall(tags, translated), f"{locale}.json changes the HTML tags of {source!r}"
        assert not re.search(r"[ZX]XQ(?:MSG|PH).*QXZ", translated), source
    assert ENGLISH_GUIDE.read_bytes() == (ROOT / "docs" / "USER_GUIDE.md").read_bytes(), (
        "USER_GUIDE.en.md is stale. Run `make translations`."
    )


# Display-text arguments of Qt calls and the templates factories; None means every positional argument.
_DISPLAY_ARGS: dict[str, tuple[int, ...] | None] = {
    **dict.fromkeys(
        (
            "setText", "setToolTip", "setWindowTitle", "setPlaceholderText", "setStatusTip", "setTitle", "showMessage",
            "setLabelText", "addAction", "addMenu", "QAction", "QMenu", "QLabel", "QPushButton", "QToolButton", "QCheckBox",
            "QGroupBox",
        ),
        None,
    ),
    "addItem": (0,), "setTabText": (1,), "addTab": (1,), "setItemText": (1,),
    "hint_label": (0,), "section_subheader": (0,), "field_label": (0,), "field_row": (0,), "make_section": (1,),
    "tool_toggle": (1, 2), "clearable_tool_toggle": (1, 2, 3), "labeled_toggle": (1, 3), "labeled_action": (1, 2),
    "icon_button": (1,), "header_button": (1,), "CompactSlider": (0,),
    "information": (1, 2), "warning": (1, 2), "critical": (1, 2), "question": (1, 2),
}  # fmt: skip
_MESSAGE_BOX = {"information", "warning", "critical", "question"}


def _untranslated(node: ast.AST) -> str | None:
    """The English in a bare literal or f-string; None for tr(...), variables and text without words."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
    elif isinstance(node, ast.JoinedStr):
        text = "".join(str(part.value) for part in node.values if isinstance(part, ast.Constant))
    else:
        return None
    return text if re.search(r"[A-Za-z]{2}", text) else None


def test_display_text_is_marked_for_translation() -> None:
    # ponytail: known calls only; text routed through a variable or a private helper passes unseen.
    offenders = []
    for path in sorted((ROOT / "negpy" / "desktop").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in _DISPLAY_ARGS:
                continue
            if name in _MESSAGE_BOX and not (isinstance(func, ast.Attribute) and getattr(func.value, "id", None) == "QMessageBox"):
                continue
            positions = _DISPLAY_ARGS[name]
            args = [arg for i, arg in enumerate(node.args) if positions is None or i in positions]
            args += [kw.value for kw in node.keywords if kw.arg in {"tooltip", "label", "title", "text"}]
            offenders += [f"{path.relative_to(ROOT)}:{node.lineno} {text!r}" for arg in args if (text := _untranslated(arg))]
    assert not offenders, f"Wrap this display text in tr(...). {_FIX}\n" + "\n".join(offenders)


def _guide_source(english: str) -> str:
    return hashlib.sha256(english.encode("utf-8")).hexdigest()[:8]


@pytest.mark.parametrize("locale", CATALOGS)
def test_localized_guides_record_their_current_english_source(locale: str) -> None:
    from negpy.desktop.view.widgets.section_help_dialog import _localized_guides

    path = CATALOGS[locale].with_name(f"USER_GUIDE.{locale}.md")
    recorded = dict(re.findall(r"^<!--\s*panel:([a-z_]+)\s+src:([0-9a-f]{8})\s*-->$", path.read_text(encoding="utf-8"), re.M))
    stale = [
        f"<!-- panel:{key} src:{_guide_source(english)} -->"
        for key, english in _localized_guides("en").items()
        if recorded.get(key) != _guide_source(english)
    ]
    assert not stale, (
        f"{path.name}: update these panels from their docs/USER_GUIDE.md sections, then replace their markers with:\n" + "\n".join(stale)
    )


def test_templates_do_not_translate_or_reformat_user_data(chinese: None) -> None:
    name = "Print {filename} 中文.raw"
    assert i18n.tr("Export failed for {full_name}: {e}", full_name=name, e="{raw error}") == f"导出 {name} 失败：{{raw error}}"
    assert i18n.tr("{source}", source=name) == name
    assert i18n.tr("Missing source {name}", name=name) == f"Missing source {name}"
    assert count_of(1, "frame") == "1 帧"
    assert count_of(2, "frame") == "2 帧"


@pytest.mark.parametrize("locale,noun", [("ja", "フレーム"), ("ko", "프레임")])
def test_japanese_and_korean_templates_keep_user_text_and_count_forms(locale: str, noun: str) -> None:
    previous = i18n.language()
    try:
        i18n.configure(locale)
        name = "Print {filename} 日本語 한국어.raw"
        result = i18n.tr("Export failed for {full_name}: {e}", full_name=name, e="{raw error}")
        assert name in result
        assert "{raw error}" in result
        assert result != f"Export failed for {name}: {{raw error}}"
        assert count_of(1, "frame") == f"1 {noun}"
        assert count_of(2, "frame") == f"2 {noun}"
    finally:
        i18n.configure(previous)


def test_german_count_forms_use_the_catalog_singular_and_plural() -> None:
    previous = i18n.language()
    try:
        i18n.configure("de")
        assert count_of(1, "frame") == "1 Bild"
        assert count_of(2, "frame") == "2 Bilder"
        assert count_of(2, "strip") == "2 Streifen"
        assert count_of(0, "file") == "0 Dateien"
    finally:
        i18n.configure(previous)


def test_format_signature_keeps_repeated_fields_and_conversions() -> None:
    assert i18n.fields("{name!r} {name}") == i18n.fields("{name} {name!r}")
    assert i18n.fields("{x:.1f}") != i18n.fields("{x:.2f}")
    assert i18n.fields("{x} {x}") != i18n.fields("{x}")


def test_bad_catalog_entries_fall_back_without_discarding_valid_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"Good": "正确", "Bad {name}": "缺少参数", "Broken": "{broken", "Last": "最后"}), encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(i18n, "get_resource_path", lambda _: str(catalog))
        try:
            i18n.configure("zh_CN")
            assert i18n.tr("Good") == "正确"
            assert i18n.tr("Bad {name}", name="test") == "Bad test"
            assert i18n.tr("Broken") == "Broken"
            assert i18n.tr("Last") == "最后"
        finally:
            i18n.configure("en")


def test_missing_catalog_keeps_the_application_usable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(i18n, "get_resource_path", lambda _: str(tmp_path / "missing.json"))
    try:
        i18n.configure("zh_CN")
        assert i18n.tr("Preferences") == "Preferences"
    finally:
        i18n.configure("en")


@pytest.mark.parametrize("locale", CATALOGS)
def test_language_preference_is_saved_and_waits_for_restart(locale: str) -> None:
    from negpy.desktop.view.widgets.preferences_dialog import PreferencesDialog
    from tests.conftest import FakeController, FakeRepo

    repo = FakeRepo()
    dialog = PreferencesDialog(FakeController(repo), None, pinned_keys=set())
    dialog.language_combo.setCurrentIndex(dialog.language_combo.findData(locale))
    assert repo.data["ui_language"] == locale
    assert not dialog._restart_hint.isHidden()
    assert i18n.language() == "en"
    reopened = PreferencesDialog(FakeController(repo), None, pinned_keys=set())
    assert reopened.language_combo.currentData() == locale


@pytest.mark.parametrize("locale,word", [("zh_CN", "密度"), ("ja", "濃度"), ("ko", "농도"), ("de", "Dichte")])
def test_localized_guides_cover_every_panel_and_fall_back(locale: str, word: str) -> None:
    from negpy.desktop.view.widgets.section_help_dialog import _localized_guides, _read_guides, guide_markdown

    english = _localized_guides("en")
    translated = _read_guides(f"negpy/translations/USER_GUIDE.{locale}.md")
    assert translated.keys() == english.keys()
    assert word in translated["tone"]
    text = CATALOGS[locale].with_name(f"USER_GUIDE.{locale}.md").read_text(encoding="utf-8")
    assert "`film:portra iso:>=400`" in text
    assert "`{{ date }}`" in text
    assert "`{{ seq }}`" in text
    previous = i18n.language()
    try:
        i18n.configure(locale)
        assert guide_markdown("tone") == translated["tone"]
    finally:
        i18n.configure(previous)
    assert _localized_guides("missing") == english
    assert _read_guides("negpy/translations/USER_GUIDE.en.md") == english


def test_chinese_command_search_keeps_english_aliases(chinese: None) -> None:
    from negpy.desktop.view.widgets.command_palette import make_entry, rank

    entry = make_entry("slider", i18n.tr("ISO-R Grade"), i18n.tr("Tone"), object())
    assert rank([entry], "对比度") == [entry]
    assert rank([entry], "contrast") == [entry]
    assert rank([entry], "反差") == [entry]


@pytest.mark.parametrize("locale,query", [("ja", "コントラスト"), ("ko", "대비")])
def test_localized_command_search_keeps_english_aliases(locale: str, query: str) -> None:
    from negpy.desktop.view.widgets.command_palette import make_entry, rank

    previous = i18n.language()
    try:
        i18n.configure(locale)
        entry = make_entry("slider", i18n.tr("ISO-R Grade"), i18n.tr("Tone"), object())
        assert rank([entry], query) == [entry]
        assert rank([entry], "contrast") == [entry]
    finally:
        i18n.configure(previous)


def test_fresh_chinese_ui_keeps_enum_and_metadata_keys(tmp_path: Path) -> None:
    # Module constants use the startup language; a fresh process tests the import order.
    result = subprocess.run(
        [sys.executable, "-c", _CHINESE_UI],
        cwd=ROOT,
        env={**os.environ, "NEGPY_USER_DIR": str(tmp_path), "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("locale", CATALOGS)
def test_localized_scan_setup_and_scope_choices_keep_stored_values(tmp_path: Path, locale: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SCAN_SETUP_AND_SCOPE_UI, locale],
        cwd=ROOT,
        env={**os.environ, "NEGPY_USER_DIR": str(tmp_path), "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


_SCAN_SETUP_AND_SCOPE_UI = """
import sys
from negpy.kernel.system.i18n import configure, tr
configure(sys.argv[1])
from PyQt6.QtWidgets import QApplication, QDialog
app = QApplication([])
from unittest.mock import MagicMock
from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.scan_setup import ScanSetupSidebar
from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.domain.models import WorkspaceConfig

controller = MagicMock()
controller.state = AppState()
for capture in ('auto', 'camera', 'scanner'):
    for light in ('white', 'narrowband'):
        controller.session.repo.get_global_setting.return_value = {'capture': capture, 'light': light}
        card = ScanSetupSidebar(controller)
        card.sync_ui()
        assert card.capture_btn.currentData() == ('camera' if capture == 'auto' else capture)
        assert card.light_btn.currentData() == light
        assert card.capture_btn._choices[0][1] == tr('Camera') != 'Camera'
        assert card.light_btn._choices[0][1] == tr('White Light') != 'White Light'
        assert card.hint.text() != ''
        card.light_btn.setCurrentIndex(1 - card.light_btn.currentIndex())
        assert controller.apply_scan_setup.call_args.args[0] == capture

dlg = GranularSettingsDialog(
    None, WorkspaceConfig(), 'scan.raw', show_scope=True, show_current=True,
    show_apply_mode=True, sel_count=2, roll_count=3,
)
assert dlg._scope_choice._choices[1][1] == tr('Selected Frames ({sel_count})', sel_count=2)
assert dlg._scope_choice._choices[1][1] != 'Selected Frames (2)'
for scope in ('current', 'selection', 'roll'):
    dlg._scope_choice.setCurrentIndex(dlg._scope_choice.findData(scope))
    dlg._on_apply()
    assert dlg.scope() == scope
dlg.mode_choice.setCurrentIndex(dlg.mode_choice.findData('replace'))
assert dlg.apply_mode() == 'replace'
"""


_CHINESE_UI = """
from dataclasses import replace
from negpy.kernel.system.i18n import configure, install_qt_translations, tr
configure('zh_CN')
from PyQt6.QtWidgets import QApplication, QDialogButtonBox
app = QApplication([])
install_qt_translations(app)
from negpy.desktop.view.widgets.preferences_dialog import PreferencesDialog
from negpy.desktop.view.widgets.roll_settings_dialog import RollSettingsDialog
from negpy.desktop.view.widgets.export_settings_form import ExportSettingsForm
from negpy.desktop.view.shortcut_registry import REGISTRY
from negpy.desktop.settings_catalog import NON_METADATA_SECTIONS
from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.gear_models import GearLibrary
from tests.conftest import FakeController, FakeRepo
prefs = PreferencesDialog(FakeController(FakeRepo(ui_language='zh_CN')), None, pinned_keys=set())
assert prefs.windowTitle() == '偏好设置'
assert prefs.language_combo.currentData() == 'zh_CN'
assert REGISTRY['open_preferences'].default_key == 'Ctrl+,'
assert REGISTRY['prev_file'].description == '上一个文件'
assert 'Metadata' not in NON_METADATA_SECTIONS
roll = RollSettingsDialog(None, WorkspaceConfig(), GearLibrary(), sel_count=1, roll_count=2)
roll.developer_edit.textEdited.emit('D-76')
assert roll._checks['Process'].isChecked()
roll.format_combo.setCurrentIndex(roll.format_combo.findData('Other'))
assert roll.format_combo.currentText() == '其他'
assert roll._meta.format == 'Other'
assert not roll.format_other_edit.isHidden()
assert roll.push_pull_combo.itemText(0) == '增感 +3'
form = ExportSettingsForm()
form.set_source_space('Adobe RGB')
assert form.ratio_combo.currentData() == 'Original'
assert form.ratio_combo.currentText() == '原始'
assert form.values()['paper_aspect_ratio'] == 'Original'
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
controller = FakeController(FakeRepo())
panel = ControlsPanel(controller)
assert {page['key']: page['label'] for page in panel.pages} == {
    'geometry': '几何', 'tone': '曝光', 'color': '风格', 'finish': '修饰',
}
assert panel.pages[0]['tooltip'] == '几何'
controller.state.config = replace(controller.state.config, geometry=replace(controller.state.config.geometry, autocrop_ratio='6:7'))
panel.autocrop_sidebar.sync_ui()
assert panel.autocrop_sidebar.ratio_combo.currentData() == '6:7'
buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
assert 'Cancel' not in buttons.button(QDialogButtonBox.StandardButton.Cancel).text()
"""


@pytest.mark.parametrize(
    "locale,expected",
    [
        (
            "ja",
            {
                "preferences": "環境設定",
                "previous": "前のファイル",
                "other": "その他",
                "original": "オリジナル",
                "cancel": "キャンセル",
                "tabs": ["ジオメトリ", "露光", "ルック", "仕上げ"],
            },
        ),
        (
            "ko",
            {
                "preferences": "환경 설정",
                "previous": "이전 파일",
                "other": "기타",
                "original": "원본",
                "cancel": "취소",
                "tabs": ["기하", "노출", "룩", "마무리"],
            },
        ),
    ],
)
def test_fresh_japanese_and_korean_ui_keeps_stored_values(tmp_path: Path, locale: str, expected: dict[str, object]) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _JAPANESE_KOREAN_UI, locale, json.dumps(expected)],
        cwd=ROOT,
        env={**os.environ, "NEGPY_USER_DIR": str(tmp_path), "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


_JAPANESE_KOREAN_UI = """
import json
import sys
from negpy.kernel.system.i18n import configure, install_qt_translations
locale, expected_json = sys.argv[1:]
expected = json.loads(expected_json)
configure(locale)
from PyQt6.QtWidgets import QApplication, QDialogButtonBox
app = QApplication([])
install_qt_translations(app)
from negpy.desktop.view.widgets.preferences_dialog import PreferencesDialog
from negpy.desktop.view.widgets.roll_settings_dialog import RollSettingsDialog
from negpy.desktop.view.widgets.export_settings_form import ExportSettingsForm
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.shortcut_registry import REGISTRY
from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.gear_models import GearLibrary
from tests.conftest import FakeController, FakeRepo
prefs = PreferencesDialog(FakeController(FakeRepo(ui_language=locale)), None, pinned_keys=set())
assert prefs.windowTitle() == expected['preferences'], prefs.windowTitle()
assert prefs.language_combo.currentData() == locale
assert REGISTRY['prev_file'].description == expected['previous']
roll = RollSettingsDialog(None, WorkspaceConfig(), GearLibrary(), sel_count=1, roll_count=2)
roll.format_combo.setCurrentIndex(roll.format_combo.findData('Other'))
assert roll.format_combo.currentText() == expected['other']
assert roll._meta.format == 'Other'
roll.developer_edit.textEdited.emit('D-76')
assert roll._checks['Process'].isChecked()
form = ExportSettingsForm()
assert form.ratio_combo.currentData() == 'Original'
assert form.ratio_combo.currentText() == expected['original']
assert form.values()['paper_aspect_ratio'] == 'Original'
panel = ControlsPanel(FakeController(FakeRepo()))
assert [page['label'] for page in panel.pages] == expected['tabs']
buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
assert buttons.button(QDialogButtonBox.StandardButton.Cancel).text().replace('&', '') == expected['cancel']
from negpy.desktop.view.widgets.command_palette import make_entry, rank
from negpy.kernel.system.i18n import tr
entry = make_entry('slider', tr('ISO-R Grade'), tr('Tone'), object())
assert rank([entry], tr('ISO-R Grade')) == [entry]
assert rank([entry], 'contrast') == [entry]
"""
