"""Every icon name in the source resolves: qtawesome raises on an unknown one only when it is drawn."""

import re
from pathlib import Path

import qtawesome as qta

ROOT = Path(__file__).resolve().parents[1]
ICON = re.compile(r"""["']((?:fa5s|fa5r|fa5b|fa|mdi|mdi6|ph|ri|msc|ei)\.[a-z0-9-]+)["']""")


def test_every_icon_name_resolves(qapp):
    unknown = []
    for path in (ROOT / "negpy").rglob("*.py"):
        for name in set(ICON.findall(path.read_text(encoding="utf-8"))):
            try:
                qta.icon(name)
            except Exception:
                unknown.append(f"{path.relative_to(ROOT)}: {name}")
    assert unknown == []
