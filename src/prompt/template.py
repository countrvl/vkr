"""Jinja-backed NL2SQL prompt builder."""

from __future__ import annotations

from importlib import resources

from jinja2 import Environment, FileSystemLoader

from src.data.loader import DataSample


class PromptBuilder:
    """Render prompts from a Jinja2 template."""

    def __init__(self, template_name: str = "nl2sql.j2") -> None:
        template_dir = resources.files("src.prompt").joinpath("templates")
        self._environment = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._template = self._environment.get_template(template_name)

    def render(self, sample: DataSample) -> str:
        """Render a prompt for a benchmark sample."""
        return self._template.render(schema=sample.schema, question=sample.question)
