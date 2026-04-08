# benchmark-domains

Репозиторий для сравнения двух классов моделей в двух независимых прикладных доменах:

- [`nl2sql/`](/home/count/code/vkr/nl2sql) — генерация SQL на `Spider` и `BIRD`
- [`code/`](/home/count/code/vkr/code) — генерация Python-кода на `HumanEval+` и `MBPP+`

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

- [`shared/configs/models.yaml`](/home/count/code/vkr/shared/configs/models.yaml) — единый каталог моделей
- [`shared/config.py`](/home/count/code/vkr/shared/config.py) — загрузка конфигурации
- [`shared/logging_utils.py`](/home/count/code/vkr/shared/logging_utils.py) — логирование и progress bar
- [`shared/inference/`](/home/count/code/vkr/shared/inference) — общий transport-слой для API и локальных backend-ов

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

Базовый сценарий одинаков для обоих доменов:

1. Подготовить данные benchmark-а.
2. Запустить inference для нужных моделей и режима.
3. Посчитать метрики через evaluation script.
4. Открыть ноутбук с итоговым отчётом.

## Быстрый старт

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

## Где смотреть результаты

- raw-генерации сохраняются в `results/<domain>/raw/`
- агрегированные метрики сохраняются в `results/<domain>/metrics/`
- графики и figures сохраняются в `results/<domain>/figures/`

## Доменные README

- [`nl2sql/README.md`](/home/count/code/vkr/nl2sql/README.md)
- [`code/README.md`](/home/count/code/vkr/code/README.md)
