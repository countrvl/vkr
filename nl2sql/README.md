# nl2sql

Домен для сравнения моделей в задаче NL2SQL:

- на benchmark-ах `Spider` и `BIRD`
- на локальном control mini-benchmark как дополнительном эксперименте основной главы
- в отдельном `strategy_bench` для production-like сценариев на реальной БД во второй части работы

## Назначение

Домен предназначен для сравнения двух классов моделей:

- `M1` — крупные general-purpose LLM через OpenAI-compatible API
- `M2` — компактные специализированные text-to-SQL модели через Ollama

Метрики домена:

- `Execution Accuracy (EA)`
- `Pass@K`
- `Expert Score (ES)`
- `Efficiency (Eff)`

Для `strategy_bench` дополнительно считаются:

- `execution_success_rate`
- `execution_accuracy`
- `pass@3`
- `latency`
- `cost` как число model calls
- `recovery_rate`

## Модели

Модели берутся из [`shared/configs/models.yaml`](../shared/configs/models.yaml) и фильтруются по `supports_sql: true`.

| Ключ | Отображаемое имя | Класс | Бэкенд |
| --- | --- | --- | --- |
| `m1_deepseek` | `DeepSeek V3.2` | `M1` | API |
| `m1_chatgpt` | `ChatGPT 5.2` | `M1` | API |
| `m1_claude` | `Claude Sonnet 4.5` | `M1` | Anthropic |
| `m2_defog` | `Defog-Llama3-SQLCoder-8B q4_0` | `M2` | Ollama |
| `m2_hrida` | `Hrida-T2SQL q8_0` | `M2` | Ollama |
| `m2_arctic` | `Arctic-Text2SQL-R1-7B` | `M2` | Ollama |
| `m2_xiyansql_32b` | `XiYanSQL-QwenCoder-32B-2504` | `M2` | Ollama |

## Бенчмарки

- `Spider` — базовый кросс-доменный бенчмарк для text-to-SQL
- `BIRD` — более сложный и более реалистичный бенчмарк

## Production Strategy Bench

`nl2sql/src/strategy_bench/` — отдельный модуль для второй части эксперимента.

Он сравнивает три стратегии на одном и том же наборе кейсов:

- `generate_only`
- `generate_validate_retry`
- `routing`

Текущие принципы реализации:

- целевая БД — `PostgreSQL` по `read-only DSN`
- dataset хранится отдельно от `Spider`/`BIRD`
- routing детерминированный: `reuse` / `adapt` / `generate`
- SQL catalog хранится во внешнем `YAML`
- schema context берется через live introspection и кэшируется на время прогона

## Local Mini Benchmark

`nl2sql/src/mini_bench/` — отдельный локальный контрольный набор для дополнительного эксперимента в основной главе.

Принципы:

- фиксированная business-like SQLite БД;
- 50 контрольных задач в YAML;
- только базовый режим `EA`;
- те же model profiles и prompt templates, что и в основном `EA`-контуре;
- отдельные summary по категориям, сложности и примерам ошибок.

## Конфигурация

- [`nl2sql/configs/experiment.yaml`](configs/experiment.yaml) — seed, sampling, `k_values`, пути к данным и результатам
- [`nl2sql/configs/metrics.yaml`](configs/metrics.yaml) — веса эффективности и параметры метрик
- [`shared/configs/models.yaml`](../shared/configs/models.yaml) — единый каталог моделей

Для API-моделей ключи доступа читаются из `.env`. Для локальных `M2` требуется запущенный `Ollama` и загруженные модели.

Для `strategy_bench` дополнительно нужны:

- переменная окружения с read-only DSN, по умолчанию `NL2SQL_STRATEGY_DB_DSN`
- драйвер `psycopg` для запуска через PostgreSQL

## Подготовка данных

```bash
uv sync
cp .env.example .env
ollama serve
uv run python nl2sql/scripts/01_download_data.py --benchmark all
```

## Как работать с доменом

Базовый сценарий работы:

1. Подготовить данные бенчмарка.
2. Запустить inference для выбранных моделей.
3. Запустить evaluation для нужного `run_label`.
4. Открыть ноутбук с итоговым отчётом.

Для второй части эксперимента сценарий отдельный:

1. Подготовить strategy-dataset в `JSON`, `JSONL` или `YAML`.
2. Подготовить YAML catalog для `routing`, если стратегия использует `reuse/adapt`.
3. Выдать read-only DSN к production PostgreSQL через env var.
4. Запустить `strategy_bench` CLI и сравнить агрегированные метрики по стратегиям.

Для локального mini-benchmark сценарий тоже отдельный:

1. Пересобрать фиксированный SQLite snapshot.
2. Проверить эталонные SQL.
3. Прогнать выбранные модели в режиме `EA` тем же prompt-контуром, что и в основном benchmark-style запуске.
4. Использовать результаты как дополнительный, а не основной источник интерпретации.

## Основные команды

### Инференс

```bash
uv run python nl2sql/scripts/02_run_inference.py --model all --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model all --benchmark all --mode pass_k
```

Можно запускать и отдельные модели, например:

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m1_chatgpt --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark all --mode ea
```

### Smoke-run

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark spider --mode ea --limit 10
```

### Оценка

```bash
uv run python nl2sql/scripts/03_evaluate.py --run-label ea
uv run python nl2sql/scripts/03_evaluate.py --run-label pass_k
uv run python nl2sql/scripts/03_evaluate.py --run-label all
```

После evaluation итоговые таблицы сохраняются в `results/nl2sql/metrics/<run_label>/`.

### Strategy Bench

Базовый запуск:

```bash
uv run python -m nl2sql.src.strategy_bench.cli \
  --dataset path/to/cases.yaml \
  --db-dsn-env NL2SQL_STRATEGY_DB_DSN \
  --model m1_chatgpt \
  --strategy all \
  --catalog-path path/to/catalog.yaml
```

Только одна стратегия:

```bash
uv run python -m nl2sql.src.strategy_bench.cli \
  --dataset path/to/cases.json \
  --db-dsn-env NL2SQL_STRATEGY_DB_DSN \
  --model m2_defog \
  --strategy generate_validate_retry
```

Полезные флаги:

- `--strategy generate_only|generate_validate_retry|routing|all`
- `--limit N`
- `--output-dir path/to/output`
- `--catalog-path path/to/catalog.yaml`
- `--max-attempts 3`

Contract strategy-dataset:

- `id`
- `natural_language_query`
- `expected_sql` или `expected_result`
- `db_target` опционально
- `metadata` опционально

Contract routing catalog:

- `id`
- `route_type: reuse | adapt`
- `match_rules`
- `sql` для `reuse`
- `template` и `placeholders` для `adapt`

Ограничения v1:

- допускаются только read-only SQL-запросы
- разрешены только `SELECT` и `WITH`
- multi-statement SQL блокируется
- `routing` остается rule-based, без отдельного LLM-router

### Mini Benchmark

Подготовка snapshot БД и проверка эталонных SQL:

```bash
uv run python nl2sql/scripts/05_prepare_mini_bench.py --force
```

Запуск для одной модели:

```bash
uv run python nl2sql/scripts/06_run_mini_bench.py --model m1_chatgpt
```

Полезные флаги:

- `--model <model_key>`
- `--limit N`
- `--dataset path/to/cases.yaml`
- `--db-path path/to/snapshot.sqlite`
- `--output-dir path/to/output`

### Ноутбуки

```bash
jupyter lab nl2sql/notebooks/01_report_ea.ipynb
jupyter lab nl2sql/notebooks/02_report_pass_k.ipynb
```

## Артефакты

- `results/nl2sql/raw/*.jsonl`
- `results/nl2sql/metrics/ea/*.csv`
- `results/nl2sql/metrics/pass_k/*.csv`
- `results/nl2sql/figures/ea/*`
- `results/nl2sql/figures/pass_k/*`
- `results/nl2sql/mini_bench/<model_key>/summary_metrics.json`
- `results/nl2sql/mini_bench/<model_key>/summary_by_category.csv`
- `results/nl2sql/mini_bench/<model_key>/failure_examples.json`
- `results/nl2sql/strategy_bench/per_case_<strategy>.json`
- `results/nl2sql/strategy_bench/summary_metrics.json`
- `results/nl2sql/strategy_bench/summary_metrics.csv`

`summary_metrics.csv` содержит агрегированные метрики по моделям, а `sample_metrics.csv` используется для анализа результатов на уровне отдельных примеров.  

## Полезные файлы

- [`nl2sql/notebooks/01_report_ea.ipynb`](notebooks/01_report_ea.ipynb)
- [`nl2sql/notebooks/02_report_pass_k.ipynb`](notebooks/02_report_pass_k.ipynb)
- [`nl2sql/src/strategy_bench/cli.py`](src/strategy_bench/cli.py)
- [`nl2sql/src/strategy_bench/runner.py`](src/strategy_bench/runner.py)
- [`nl2sql/src/strategy_bench/routing.py`](src/strategy_bench/routing.py)
