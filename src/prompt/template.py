"""Jinja-backed NL2SQL prompt builder."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.data.loader import DataSample


class PromptBuilder:
    """Render prompts from a Jinja2 template."""

    def __init__(self, template_dir: Path | None = None) -> None:
        """Initialize the Jinja2 environment.

        Args:
            template_dir: Optional template directory. Defaults to `src/prompt/templates`.
        """
        resolved_template_dir = template_dir or Path(__file__).resolve().parent / "templates"
        self._environment = Environment(
            loader=FileSystemLoader(str(resolved_template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def build(self, sample: DataSample, template_name: str = "nl2sql.j2") -> str:
        """Render a prompt from a benchmark sample.

        Args:
            sample: Unified benchmark sample.
            template_name: Template filename within the template directory.

        Returns:
            Rendered prompt text.
        """
        template = self._environment.get_template(template_name)
        return template.render(schema=sample.schema, question=sample.question)

    def render(self, sample: DataSample) -> str:
        """Backward-compatible alias for `build()`."""
        return self.build(sample)
