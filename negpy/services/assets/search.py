"""Query language shared by the film-strip filter and the library search.

One parser, one matcher: the session filter and any cross-folder search run the
same terms against the same facts dict, so search semantics cannot drift apart.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

FLAG_FIELDS = frozenset({"keeper", "rejected", "edited"})
NUMERIC_FIELDS = frozenset({"iso", "frame", "push"})
# shot is truncated ISO-8601, so the prefix comparison below orders it without parsing a date.
TEXT_FIELDS = frozenset(
    {"name", "path", "ext", "film", "camera", "lens", "developer", "format", "scanning", "roll", "date", "shot", "place"}
)
FIELDS = FLAG_FIELDS | NUMERIC_FIELDS | TEXT_FIELDS

_OPS = (">=", "<=", ">", "<")


@dataclass(frozen=True)
class Term:
    field: str  # "" = bare word, matched against the filename
    op: str  # ":" | ">" | ">=" | "<" | "<="
    value: str
    negate: bool = False


def parse_query(text: str) -> list[Term]:
    """Split a query into ANDed terms. Never raises: an unparseable token is a bare word."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        # Half-typed quote: the box is live, so fall back rather than reject.
        tokens = text.split()
    return [t for t in (_term(tok) for tok in tokens) if t is not None]


def _term(token: str) -> Optional[Term]:
    negate = token.startswith("-")
    if negate:
        token = token[1:]
    if not token:
        return None

    field, sep, value = token.partition(":")
    field = field.lower()
    if not sep or field not in FIELDS:
        return Term("", ":", token.casefold(), negate)

    op = ":"
    for candidate in _OPS:
        if value.startswith(candidate):
            op, value = candidate, value[len(candidate) :]
            break
    return Term(field, op, value.casefold(), negate)


def match(terms: list[Term], facts: dict[str, Any]) -> bool:
    """True when every term holds for these facts (an empty query matches everything)."""
    return all(term.negate != _holds(term, facts) for term in terms)


def _holds(term: Term, facts: dict[str, Any]) -> bool:
    if term.field in FLAG_FIELDS:
        return bool(facts.get(term.field))

    fact = facts.get(term.field or "name")
    if fact is None or fact == "":
        return False

    if term.field in NUMERIC_FIELDS:
        return _compare_numeric(term, fact)
    return _compare_text(term, str(fact).casefold())


def _compare_numeric(term: Term, fact: Any) -> bool:
    try:
        return _apply_op(term.op, float(fact), float(term.value))
    except (TypeError, ValueError):
        return False


def _compare_text(term: Term, fact: str) -> bool:
    if term.op == ":":
        return term.value in fact
    # Ordered comparison on a prefix. ISO dates sort lexicographically, so `date:>=2024-03`
    # works without parsing either side into a date.
    return _apply_op(term.op, fact[: len(term.value)], term.value)


def _apply_op(op: str, left: Any, right: Any) -> bool:
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    return left == right


def facts_for(asset: dict, config: Any = None) -> dict[str, Any]:
    """Searchable facts for one asset: its file identity plus, when it has a saved
    edit, that edit's archival metadata. ``config`` is None for an unedited frame."""
    name = str(asset.get("name") or "")
    facts: dict[str, Any] = {
        "name": name.casefold(),
        "path": str(asset.get("path") or "").casefold(),
        "ext": os.path.splitext(name)[1].lstrip(".").casefold(),
        "date": _date_str(asset.get("mtime")),
        "keeper": bool(asset.get("keeper")),
        "rejected": bool(asset.get("excluded")),
        "edited": config is not None,
    }
    if config is None:
        return facts

    meta = config.metadata
    facts.update(
        {
            "film": " ".join(p for p in (meta.film_manufacturer, meta.film) if p).casefold(),
            "camera": " ".join(p for p in (meta.camera_make, meta.camera_model) if p).casefold(),
            "lens": " ".join(p for p in (meta.lens_make, meta.lens_model) if p).casefold(),
            "developer": meta.developer.casefold(),
            "format": (meta.format_other if meta.format == "Other" else meta.format).casefold(),
            "scanning": meta.scanning.casefold(),
            "roll": " ".join(p for p in (meta.capture_roll, config.process.roll_name or "") if p).casefold(),
            "frame": meta.capture_frame,
            "iso": meta.film_iso,
            "push": meta.push_pull,
            "shot": meta.capture_date.casefold(),
            "place": " ".join(p for p in (meta.location_city, meta.location_state, meta.location_country) if p).casefold(),
        }
    )
    return facts


def _date_str(mtime: Any) -> str:
    try:
        return datetime.fromtimestamp(float(mtime)).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""
