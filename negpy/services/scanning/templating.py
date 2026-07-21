import re

from jinja2.sandbox import SandboxedEnvironment


def render_scan_filename(pattern: str, date: str, seq: int) -> str:
    """Render scan filename via Jinja2. Variables: date (str), seq (int)."""
    try:
        env = SandboxedEnvironment()
        template = env.from_string(pattern)
        rendered = template.render(date=date, seq=seq)
        rendered = re.sub(r"[ _-]+", "_", rendered).strip("_")
        return rendered or f"{date}_{seq:03d}"
    except Exception:
        return f"{date}_{seq:03d}"


def require_sequence_varying_scan_filename(pattern: str, date: str, sequence: int = 1) -> None:
    """Reject a scan template that cannot allocate a second basename.

    Collision allocators advance ``seq`` until they find an unused basename. A
    template whose rendered value does not change with ``seq`` would otherwise
    retry the same occupied path forever.
    """
    if type(sequence) is not int or sequence < 1:
        raise ValueError("sequence must be a positive integer")
    current = render_scan_filename(pattern, date, sequence)
    following = render_scan_filename(pattern, date, sequence + 1)
    if current == following:
        raise ValueError("filename pattern must produce a different basename when seq changes")
