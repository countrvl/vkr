"""Jinja-backed NL2SQL prompt builder."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from nl2sql.src.data.loader import DataSample


_DEFAULT_PROFILE = "nl2sql_json"
_PROFILE_TO_TEMPLATE = {
    "nl2sql_json": "nl2sql.j2",
    "m2_sql_continuation": "m2_sql_continuation.j2",
    "defog_sqlcoder": "defog_sqlcoder.j2",
    "xiyansql_sqlite": "xiyansql_sqlite.j2",
}


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
        self._default_template = self._environment.get_template(_PROFILE_TO_TEMPLATE[_DEFAULT_PROFILE])

    def build(self, sample: DataSample, template_name: str = _DEFAULT_PROFILE) -> str:
        """Render a prompt from a benchmark sample.

        ``template_name`` may be either a prompt profile from
        ``_PROFILE_TO_TEMPLATE`` or a direct template filename.

        Args:
            sample: Unified benchmark sample.
            template_name: Prompt profile or template filename within the template directory.

        Returns:
            Rendered prompt text.
        """
        resolved_template = _PROFILE_TO_TEMPLATE.get(template_name, template_name)
        if resolved_template == _PROFILE_TO_TEMPLATE[_DEFAULT_PROFILE]:
            template = self._default_template
        else:
            template = self._environment.get_template(resolved_template)
        return template.render(
            schema=sample.schema,
            question=sample.question,
            evidence=sample.evidence or "",
            dialect="SQLite",
        )

    def render(self, sample: DataSample) -> str:
        """Backward-compatible alias for `build()`."""
        return self.build(sample)
