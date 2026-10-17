"""UI language, message catalogs and formatting; set the language at startup."""

import json
import logging
from pathlib import Path
from string import Formatter
from typing import TYPE_CHECKING, Any

from negpy.kernel.system.paths import get_resource_path

if TYPE_CHECKING:
    from PyQt6.QtCore import QCoreApplication, QTranslator


def mark(source: str) -> str:
    """Mark source text in static catalogs; call tr when displaying it."""
    return source


LANGUAGES = (("system", mark("System Default")), ("en", "English"), ("zh_CN", "简体中文"), ("ja", "日本語"), ("ko", "한국어"))
_language = "en"
_messages: dict[str, str] = {}
_formatter = Formatter()
_qt_translator: "QTranslator | None" = None


def resolve_language(choice: str, system_languages: list[str] | None = None) -> str:
    """Resolve the UI language, with English fallback for unsupported languages."""
    if choice == "system":
        if system_languages is None:
            from PyQt6.QtCore import QLocale

            system_languages = QLocale.system().uiLanguages()
        choice = next(iter(system_languages), "en")
    tag = choice.replace("-", "_").lower()
    if tag in {"zh", "zh_cn", "zh_sg", "zh_hans", "zh_hans_cn", "zh_hans_sg"}:
        return "zh_CN"
    base = tag.split("_", 1)[0]
    supported = {locale.lower(): locale for locale, _ in LANGUAGES if locale != "system"}
    return supported.get(tag, supported.get(base, "en"))


def fields(text: str) -> list[tuple[str, str, str | None]]:
    """Keep placeholder counts, formats and conversions for catalog validation."""
    placeholders = [(name, spec or "", conversion) for _, name, spec, conversion in _formatter.parse(text) if name is not None]
    return sorted(placeholders, key=lambda value: (value[0], value[1], value[2] or ""))


def configure(choice: str = "system") -> str:
    """Load the catalog before UI imports fix the language of module constants."""
    global _language, _messages
    language = resolve_language(choice)
    messages: dict[str, str] = {}
    if language != "en":
        try:
            raw = json.loads(Path(get_resource_path(f"negpy/translations/{language}.json")).read_text(encoding="utf-8"))
            for source, translated in raw.items():
                try:
                    if isinstance(translated, str) and translated and fields(source) == fields(translated):
                        messages[source] = translated
                except ValueError:
                    logging.getLogger(__name__).warning("Invalid translation format: %s", source)
        except (OSError, ValueError, AttributeError, TypeError):
            logging.getLogger(__name__).warning("Translation catalog unavailable: %s", language, exc_info=True)
    _language, _messages = language, messages
    return language


def language() -> str:
    return _language


def tr(source: str, /, **values: Any) -> str:
    """Translate the template first; insert file names, paths and user text as values."""
    text = _messages.get(source, source)
    return text.format(**values) if values else text


def install_qt_translations(app: "QCoreApplication") -> None:
    """Use the application language for Qt's standard buttons and file dialogs."""
    global _qt_translator
    from PyQt6.QtCore import QLibraryInfo, QTranslator

    if _qt_translator is not None:
        app.removeTranslator(_qt_translator)
        _qt_translator = None
    if _language == "en":
        return
    translator = QTranslator(app)
    if translator.load(f"qtbase_{_language}", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
        app.installTranslator(translator)
        _qt_translator = translator
