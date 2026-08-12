"""Wording helpers for text the user reads."""


def plural(count: int, singular: str, plural_form: str = "") -> str:
    """The right noun for `count`, without the number.

    `plural_form` for nouns an -s does not cover ("copies", "indices").
    """
    return singular if abs(count) == 1 else (plural_form or f"{singular}s")


def count_of(count: int, singular: str, plural_form: str = "") -> str:
    """`count` and its noun together: "1 file", "3 files".

    Exists because "(s)" is the usual shortcut and it reads as unfinished — "Reopen your
    last session (1 file(s))?" is the report that prompted this. A count is nearly always
    shown with its noun, so one call covers the whole phrase and there is no second place
    to forget.
    """
    return f"{count} {plural(count, singular, plural_form)}"
