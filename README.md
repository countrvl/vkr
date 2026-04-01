# benchmark-domains

Один репозиторий с двумя независимыми доменами тестирования моделей:

- [`nl2sql/`](/home/count/code/vkr/nl2sql) — Spider/BIRD
- [`code/`](/home/count/code/vkr/code) — HumanEval+/MBPP+

## Структура

```text
benchmark-domains/
├── nl2sql/   # text-to-SQL domain
├── code/     # code-generation domain
├── shared/   # только infra: config/logging/progress
├── data/
│   ├── nl2sql/
│   └── code/
├── results/
│   ├── nl2sql/
│   └── code/
└── pyproject.toml
```

## Что где

- [`nl2sql/README.md`](/home/count/code/vkr/nl2sql/README.md) — все инструкции по Spider/BIRD
- [`code/README.md`](/home/count/code/vkr/code/README.md) — каркас и будущий pipeline для HumanEval+/MBPP+
- [`shared/config.py`](/home/count/code/vkr/shared/config.py) — общий загрузчик YAML-конфигов
- [`shared/logging_utils.py`](/home/count/code/vkr/shared/logging_utils.py) — общий logging/progress UX

## Установка

```bash
uv sync
cp .env.example .env
```

## Запуск

NL2SQL:

```bash
uv run python nl2sql/scripts/01_download_data.py --benchmark all
uv run python nl2sql/scripts/02_run_inference.py --model all --benchmark all --mode ea
uv run python nl2sql/scripts/03_evaluate.py --run-label ea
jupyter lab nl2sql/notebooks/01_report_ea.ipynb
```

Code:

```bash
uv run python code/scripts/01_prepare_benchmarks.py
uv run python code/scripts/02_run_inference.py
uv run python code/scripts/03_evaluate.py
```

Сейчас полноценный production-ready pipeline реализован только для `nl2sql/`.

