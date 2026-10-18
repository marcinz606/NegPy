"""Extract marked UI sources, preserve translations and check placeholders."""

import argparse
import ast
import json
from pathlib import Path

from negpy.kernel.system.i18n import LANGUAGES, fields

ROOT = Path(__file__).resolve().parents[1]
TRANSLATIONS = ROOT / "negpy" / "translations"
CATALOGS = {locale: TRANSLATIONS / f"{locale}.json" for locale, _ in LANGUAGES if locale not in {"system", "en"}}
ENGLISH_GUIDE = TRANSLATIONS / "USER_GUIDE.en.md"


def extract() -> set[str]:
    """Marked source texts, found without importing UI modules."""
    messages: set[str] = set()
    for path in sorted((ROOT / "negpy").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            indices = {"tr": (0,), "mark": (0,), "count_of": (1, 2), "plural": (1, 2)}.get(node.func.id, ())
            for index in indices:
                if len(node.args) > index and isinstance(node.args[index], ast.Constant) and isinstance(node.args[index].value, str):
                    messages.add(node.args[index].value)
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check missing translations, obsolete entries and placeholders without writing files"
    )
    args = parser.parse_args()
    messages = extract()
    errors = []
    for locale, path in CATALOGS.items():
        catalog = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        for source in sorted(messages):
            translated = catalog.get(source, "")
            if not translated:
                errors.append(f"{locale} Missing: {source!r}")
            else:
                try:
                    if fields(source) != fields(translated):
                        errors.append(f"{locale} Placeholders: {source!r}")
                except ValueError:
                    errors.append(f"{locale} Invalid format: {source!r}")
        if args.check:
            errors.extend(f"{locale} Obsolete: {source!r}" for source in sorted(catalog.keys() - messages))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({source: catalog.get(source, "") for source in sorted(messages)}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    if args.check:
        if not ENGLISH_GUIDE.exists() or ENGLISH_GUIDE.read_bytes() != (ROOT / "docs" / "USER_GUIDE.md").read_bytes():
            errors.append("Packaged English guide is stale; run make translations")
        if errors:
            raise SystemExit("\n".join(errors))
    else:
        # Wheels lack the repository docs directory; generate the English copy from its source.
        ENGLISH_GUIDE.write_bytes((ROOT / "docs" / "USER_GUIDE.md").read_bytes())
    print(f"{len(CATALOGS)} catalogs; {len(messages)} messages each; {len(errors)} translation errors")


if __name__ == "__main__":
    main()
