import datetime
from jinja2 import Environment, BaseLoader


class FilenameTemplater:
    """
    Jinja2 filename generator.
    """

    def __init__(self) -> None:
        self.env = Environment(
            loader=BaseLoader(),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, pattern: str, context: dict) -> str:
        try:
            template = self.env.from_string(pattern)
            render_context = {"date": datetime.date.today().isoformat(), **context}
            rendered = template.render(render_context).strip()
            if not rendered:
                raise ValueError("Template rendered to empty string")
            return rendered
        except Exception:
            original = context.get("original_name", "output")
            return f"positive_{original}"
