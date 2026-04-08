"""Построитель prompt-ов для генерации кода на базе Jinja."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from code_bench.data.schema import CodeSample


_DEFAULT_PROFILE = "codegen_default"
_TEMPLATE_BY_PROFILE = {
    "codegen_default": "codegen.j2",
    "qwen2_5_coder": "qwen2_5_coder.j2",
    "codegemma_instruct": "codegemma_instruct.j2",
    "deepseek_coder": "deepseek_coder.j2",
    "codellama_instruct": "codellama_instruct.j2",
}


class PromptBuilder:
    """Рендерить prompt-ы из Jinja2-шаблонов."""

    def __init__(self, template_dir: Path | None = None) -> None:
        resolved_template_dir = template_dir or Path(__file__).resolve().parent / "templates"
        self._environment = Environment(
            loader=FileSystemLoader(str(resolved_template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            auto_reload=False,
        )
        self._default_template = self._environment.get_template(_TEMPLATE_BY_PROFILE[_DEFAULT_PROFILE])

    def build(self, sample: CodeSample, prompt_profile: str = _DEFAULT_PROFILE) -> str:
        template_name = _TEMPLATE_BY_PROFILE.get(prompt_profile, prompt_profile)
        template = (
            self._default_template
            if template_name == _TEMPLATE_BY_PROFILE[_DEFAULT_PROFILE]
            else self._environment.get_template(template_name)
        )
        return template.render(
            prompt_text=sample.prompt_text,
            entry_point=sample.entry_point,
            benchmark=sample.benchmark,
            contract=sample.contract,
        )

    def render(self, sample: CodeSample) -> str:
        return self.build(sample)
