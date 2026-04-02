"""Jinja-backed prompt builder for code generation."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from code.src.data.schema import CodeSample


_DEFAULT_TEMPLATE = "codegen.j2"


class PromptBuilder:
    """Render prompts from a Jinja2 template."""

    def __init__(self, template_dir: Path | None = None) -> None:
        resolved_template_dir = template_dir or Path(__file__).resolve().parent / "templates"
        self._environment = Environment(
            loader=FileSystemLoader(str(resolved_template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            auto_reload=False,
        )
        self._default_template = self._environment.get_template(_DEFAULT_TEMPLATE)

    def build(self, sample: CodeSample, template_name: str = _DEFAULT_TEMPLATE) -> str:
        template = self._default_template if template_name == _DEFAULT_TEMPLATE else self._environment.get_template(template_name)
        return template.render(
            prompt_text=sample.prompt_text,
            entry_point=sample.entry_point,
            benchmark=sample.benchmark,
            contract=sample.contract,
        )

    def render(self, sample: CodeSample) -> str:
        return self.build(sample)
