# benchmark-domains

Репозиторий для сравнения двух классов моделей в двух независимых прикладных доменах:

- [`nl2sql/`](nl2sql/) — генерация SQL на `Spider` и `BIRD`
- [`code/`](code/) — генерация Python-кода на `HumanEval+` и `MBPP+`

Для `nl2sql` теперь есть два независимых контура:

- benchmark-style evaluation на `Spider`/`BIRD`
- strategy bench для production-like NL2SQL на реальной БД с маршрутами `generate_only`, `generate_validate_retry`, `routing`

## Назначение

Проект сравнивает:

- `M1` — крупные general-purpose модели
- `M2` — компактные специализированные модели

Оба домена используют общий инфраструктурный слой из `shared/`, но имеют собственные данные, промпты, метрики, пайплайны оценки и отчётные ноутбуки.

## Структура

```text
benchmark-domains/
├── nl2sql/
├── code/
├── shared/
├── data/
├── results/
└── pyproject.toml
```

## Общие компоненты

- [`shared/configs/models.yaml`](shared/configs/models.yaml) — единый каталог моделей
- [`shared/config.py`](shared/config.py) — загрузка конфигурации
- [`shared/logging_utils.py`](shared/logging_utils.py) — логирование и индикаторы прогресса
- [`shared/inference/`](shared/inference/) — общий транспортный слой для API и локальных backend-ов

## Установка

```bash
uv sync
cp .env.example .env
```

В `.env` задаются ключи доступа для API-моделей. Для локальных `M2` также должен быть доступен запущенный `Ollama`.

## Проверка окружения

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
```

## Как работать с проектом

Базовый сценарий работы одинаков для обоих доменов:

1. Подготовить данные бенчмарка.
2. Запустить inference для нужных моделей и режима.
3. Запустить evaluation и сохранить агрегированные метрики.
4. Открыть ноутбук с итоговым отчётом.

## Быстрый старт

### NL2SQL

```bash
uv run python nl2sql/scripts/01_download_data.py --benchmark all
uv run python nl2sql/scripts/02_run_inference.py --model all --benchmark all --mode ea
uv run python nl2sql/scripts/03_evaluate.py --run-label ea
jupyter lab nl2sql/notebooks/01_report_ea.ipynb
```

### NL2SQL Strategy Bench

```bash
uv run python -m nl2sql.src.strategy_bench.cli \
  --dataset path/to/cases.yaml \
  --db-dsn-env NL2SQL_STRATEGY_DB_DSN \
  --model m1_chatgpt \
  --strategy all \
  --catalog-path path/to/catalog.yaml
```

`strategy_bench` рассчитан на read-only доступ к production PostgreSQL и отдельный dataset в JSON/JSONL/YAML.

### Code

```bash
uv run python code/scripts/01_prepare_benchmarks.py --benchmark all
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode fc
uv run python code/scripts/03_evaluate.py --run-label fc
jupyter lab code/notebooks/01_report_fc_passk.ipynb
```

## Где смотреть результаты

- raw-результаты сохраняются в `results/<domain>/raw/`
- агрегированные метрики сохраняются в `results/<domain>/metrics/`
- графики и изображения для отчётов сохраняются в `results/<domain>/figures/`
- strategy-bench логи и summary сохраняются в `results/nl2sql/strategy_bench/`

## Доменные README

- [`nl2sql/README.md`](nl2sql/README.md)
- [`code/README.md`](code/README.md)
