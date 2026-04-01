# code

Домен для тестирования моделей-кодеров на `HumanEval+` и `MBPP+`.

Сейчас здесь создан каркас проекта:

- собственные `configs/`
- собственные `scripts/`
- собственные `src/`
- собственные `tests/`
- собственные `notebooks/`

Полный pipeline для code-бенчмарков будет добавляться следующим этапом.

Предполагаемые команды:

```bash
uv run python code/scripts/01_prepare_benchmarks.py
uv run python code/scripts/02_run_inference.py
uv run python code/scripts/03_evaluate.py
uv run python code/scripts/04_archive_results.py
```

