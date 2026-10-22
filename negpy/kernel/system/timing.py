"""Wall time per pipeline step, collected per thread and logged as one line that adds up."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator, List, Optional

# Under this a step is a cache hit or a no-op, and it only clutters the line.
_SHOWN_MS = 0.5


@dataclass
class Step:
    name: str
    ms: float = 0.0
    children: List["Step"] = field(default_factory=list)
    last: float = 0.0  # when the previous lap or child step ended


_current: ContextVar[Optional[Step]] = ContextVar("negpy_timing", default=None)


@contextmanager
def collect(name: str = "total") -> Iterator[Step]:
    """Start a timing tree on this thread. A thread without one records nothing."""
    root = Step(name, last=perf_counter())
    start = root.last
    token = _current.set(root)
    try:
        yield root
    finally:
        _current.reset(token)
        root.ms = (perf_counter() - start) * 1000


@contextmanager
def step(name: str) -> Iterator[None]:
    """A child step whose own laps nest under it."""
    parent = _current.get()
    if parent is None:
        yield
        return
    node = Step(name, last=perf_counter())
    start = node.last
    parent.children.append(node)
    token = _current.set(node)
    try:
        yield
    finally:
        _current.reset(token)
        parent.last = perf_counter()
        node.ms = (parent.last - start) * 1000


def lap(name: str) -> None:
    """Record the time since the previous lap or step as `name`."""
    node = _current.get()
    if node is None:
        return
    now = perf_counter()
    node.children.append(Step(name, (now - node.last) * 1000))
    node.last = now


def _merged(children: List[Step]) -> List[Step]:
    """Same-named steps summed in first-seen order, so a per-tile step reads as one entry."""
    by_name: dict = {}
    for c in children:
        m = by_name.setdefault(c.name, Step(c.name))
        m.ms += c.ms
        m.children += c.children
    return list(by_name.values())


def _parts(node: Step) -> List[str]:
    parts = [
        f"{c.name} {c.ms:.0f}" + (f" [{', '.join(_parts(c))}]" if c.children else "") for c in _merged(node.children) if c.ms >= _SHOWN_MS
    ]
    rest = node.ms - sum(c.ms for c in node.children)
    if node.children and rest >= _SHOWN_MS:
        parts.append(f"other {rest:.0f}")
    return parts


def summary(root: Step) -> str:
    """'60 ms = decode 10 + GPU 40 [upload 30, other 10] + other 8'; all figures in ms."""
    parts = _parts(root)
    return f"{root.ms:.0f} ms" + (" = " + " + ".join(parts) if parts else "")
