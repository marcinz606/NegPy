"""Source-identity tokens: what an assembled source buffer depends on.

A token keys every cache that holds a decoded source, and decides whether a config
change needs a re-decode or only a re-render. It is built from the **whole** config
rather than a hand-picked list of fields, because a field left out of a token is
invisible until a stale buffer is served for the setting the user just changed --
which is exactly how a new HDR field once left the render exposure with no effect.

Imports nothing from the feature packages; they import this.
"""

import hashlib
import os
from dataclasses import astuple
from typing import Any, Sequence


def composite_token(kind: str, config: Any, paths: Sequence[str]) -> str:
    """``|kind:path:mtime:...:digest`` for an assembled source, or "" if one is unreachable.

    The digest covers every field of `config`, so a field added later is included without
    being listed here. `paths` are the extra files the assembly reads; their mtimes make
    an edited source a cache miss.

    "" means "cannot be assembled" — a part has been moved or deleted — and is returned
    rather than raised because a token is built on the way to a cache lookup, not inside
    the decode that will report the missing file properly. It is *not* an inactive
    assembly: callers that key a cache on the token must skip caching when it is empty,
    since two different broken configs share it.
    """
    stamps = []
    for path in paths:
        if not path:
            continue
        try:
            stat = os.stat(path)
        except OSError:
            return ""
        # Size and nanosecond mtime, not getmtime(): whole seconds miss a file rewritten within
        # the same second as the last decode, which is ordinary when a script regenerates a
        # companion exposure.
        stamps.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
    digest = hashlib.sha256(repr(astuple(config)).encode()).hexdigest()[:16]
    return f"|{kind}:{':'.join(stamps)}:{digest}"
