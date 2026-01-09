import datetime
from jinja2 import Environment, BaseLoader


class FilenameTemplater:
    """
    Handles generation of filenames using Jinja2 templates.
    """

    def __init__(self) -> None:
        # Using a minimal environment for performance and safety
        self.env = Environment(
            loader=BaseLoader(),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, pattern: str, context: dict) -> str:
        """
        Renders the filename pattern with the provided context.
        Falls back to a safe default if rendering fails.
        """
        try:
            template = self.env.from_string(pattern)
            # Add common defaults to context
            render_context = {"date": datetime.date.today().isoformat(), **context}
            rendered = template.render(render_context).strip()
            if not rendered:
                raise ValueError("Template rendered to empty string")
            return rendered
        except Exception:
            original = context.get("original_name", "output")
            return f"positive_{original}"
