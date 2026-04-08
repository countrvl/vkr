# benchmark-domains

Один репозиторий для двух независимых доменов сравнения моделей:

- [`nl2sql/`](/home/count/code/vkr/nl2sql) — генерация SQL для `Spider` и `BIRD`
- [`code/`](/home/count/code/vkr/code) — генерация Python-кода для `HumanEval+` и `MBPP+`

## Назначение

Проект используется для сравнения двух классов моделей:

- `M1` — крупные general-purpose модели
- `M2` — компактные специализированные модели

Домены независимы по данным, prompt-слою, evaluation, метрикам и ноутбукам. Общая инфраструктура вынесена в `shared/`.

## Структура

```text
benchmark-domains/
├── nl2sql/
├── code/
├── shared/
├── data/
│   ├── nl2sql/
│   └── code/
├── results/
│   ├── nl2sql/
│   └── code/
└── pyproject.toml
```

## Общие компоненты

- [`shared/config.py`](/home/count/code/vkr/shared/config.py) — загрузка YAML-конфигов
- [`shared/configs/models.yaml`](/home/count/code/vkr/shared/configs/models.yaml) — единый каталог моделей
- [`shared/logging_utils.py`](/home/count/code/vkr/shared/logging_utils.py) — logging и progress bar
- [`shared/inference/`](/home/count/code/vkr/shared/inference) — общий transport-слой для API и Ollama

Доменные runtime-переопределения задаются через:

- `domain_overrides.sql`
- `domain_overrides.code`

## Установка

```bash
uv sync
cp .env.example .env
```

## Быстрая проверка

Перед длинными прогонами:

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
```

## Основные документы

- [`nl2sql/README.md`](/home/count/code/vkr/nl2sql/README.md) — подробности по NL2SQL-домену
- [`code/README.md`](/home/count/code/vkr/code/README.md) — подробности по code-домену
- [`plan.md`](/home/count/code/vkr/plan.md) — рабочий план и backlog

## Типовые команды

### NL2SQL

```bash
uv run python nl2sql/scripts/01_download_data.py --benchmark all
uv run python nl2sql/scripts/02_run_inference.py --model all --benchmark all --mode ea
uv run python nl2sql/scripts/03_evaluate.py --run-label ea
jupyter lab nl2sql/notebooks/01_report_ea.ipynb
```

### Code

```bash
uv run python code/scripts/01_prepare_benchmarks.py --benchmark all
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode fc
uv run python code/scripts/03_evaluate.py --run-label fc
jupyter lab code/notebooks/01_report_fc_passk.ipynb
```
