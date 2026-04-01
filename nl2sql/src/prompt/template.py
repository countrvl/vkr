"""Jinja-backed NL2SQL prompt builder."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from nl2sql.src.data.loader import DataSample


_DEFAULT_TEMPLATE = "nl2sql.j2"


class PromptBuilder:
    """Render prompts from a Jinja2 template."""

    def __init__(self, template_dir: Path | None = None) -> None:
        """Initialize the Jinja2 environment and pre-load the default template.

        The default template is loaded once at construction time
        (``auto_reload=False``), avoiding a disk stat on every ``build()``
        call during large benchmark runs.

        Args:
            template_dir: Optional template directory. Defaults to `src/prompt/templates`.
        """
        resolved_template_dir = template_dir or Path(__file__).resolve().parent / "templates"
        self._environment = Environment(
            loader=FileSystemLoader(str(resolved_template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            auto_reload=False,
        )
        self._default_template = self._environment.get_template(_DEFAULT_TEMPLATE)

    def build(self, sample: DataSample, template_name: str = _DEFAULT_TEMPLATE) -> str:
        """Render a prompt from a benchmark sample.

        Uses the pre-loaded default template when ``template_name`` matches the
        default; falls back to ``get_template()`` for non-default names.

        Args:
            sample: Unified benchmark sample.
            template_name: Template filename within the template directory.

        Returns:
            Rendered prompt text.
        """
        if template_name == _DEFAULT_TEMPLATE:
            template = self._default_template
        else:
            template = self._environment.get_template(template_name)
        return template.render(schema=sample.schema, question=sample.question)

    def render(self, sample: DataSample) -> str:
        """Backward-compatible alias for `build()`."""
        return self.build(sample)
