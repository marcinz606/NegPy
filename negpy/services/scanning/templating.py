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
