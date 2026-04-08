# nl2sql

Домен для сравнения моделей в задаче NL2SQL на бенчмарках `Spider` и `BIRD`.

## Назначение

В домене сравниваются два класса моделей:

- `M1` — крупные general-purpose LLM через OpenAI-compatible API
- `M2` — компактные специализированные text-to-SQL модели через Ollama

Метрики домена:

- `Execution Accuracy (EA)`
- `Pass@K`
- `Expert Score (ES)`
- `Efficiency (Eff)`

## Актуальные модели

Модели берутся из [`shared/configs/models.yaml`](/home/count/code/vkr/shared/configs/models.yaml) и фильтруются по `supports_sql: true`.

| Ключ | Отображаемое имя | Класс | Бэкенд |
| --- | --- | --- | --- |
| `m1_deepseek` | `DeepSeek V3.2` | `M1` | API |
| `m1_chatgpt` | `ChatGPT 5.2` | `M1` | API |
| `m1_claude` | `Claude Sonnet 4.5` | `M1` | Anthropic |
| `m2_defog` | `Defog-Llama3-SQLCoder-8B q4_0` | `M2` | Ollama |
| `m2_hrida` | `Hrida-T2SQL q8_0` | `M2` | Ollama |
| `m2_arctic` | `Arctic-Text2SQL-R1-7B` | `M2` | Ollama |
| `m2_xiyansql_32b` | `XiYanSQL-QwenCoder-32B-2504` | `M2` | Ollama |

Для `m2_xiyansql_32b` используется Ollama tag `Kaiyue/xiyansql-32b:latest`.

## Бенчмарки

- `Spider` — базовый кросс-доменный benchmark для text-to-SQL
- `BIRD` — более сложный и более реалистичный benchmark

Практически `BIRD` заметно тяжелее `Spider`, особенно для компактных `M2`.

## Конфигурация

- [`nl2sql/configs/experiment.yaml`](/home/count/code/vkr/nl2sql/configs/experiment.yaml) — seed, sampling, `k_values`, пути к данным и результатам
- [`nl2sql/configs/metrics.yaml`](/home/count/code/vkr/nl2sql/configs/metrics.yaml) — веса эффективности и параметры метрик
- [`shared/configs/models.yaml`](/home/count/code/vkr/shared/configs/models.yaml) — единый каталог моделей

Для SQL-домена model-specific prompt выбирается из `domain_overrides.sql`.

Для `Claude Sonnet 4.5` batch используется как внутренняя оптимизация transport-слоя.
Снаружи команды инференса и raw-артефакты остаются теми же, что и для других моделей.

## Быстрый старт

```bash
uv sync
cp .env.example .env
ollama serve
uv run python nl2sql/scripts/01_download_data.py --benchmark all
```

Если планируется `Claude`, нужно задать `ANTHROPIC_API_KEY`.

Для локальных `M2` модели Ollama подтягиваются отдельно, например:

```bash
ollama pull mannix/defog-llama3-sqlcoder-8b:q4_0
```

## Основные команды

### Инференс

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m1_deepseek --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m1_chatgpt --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model all --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model all --benchmark all --mode pass_k
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

## Архивация

```bash
.venv/bin/python nl2sql/scripts/04_archive_results.py --dry-run
.venv/bin/python nl2sql/scripts/04_archive_results.py --label before_new_run
.venv/bin/python nl2sql/scripts/04_archive_results.py --scope ea --label before_pass_k
```

Скрипт архивирует текущие `results/nl2sql/*` и копирует snapshot SQL-ноутбуков в архив.

## Проверка статуса

```bash
ls -lh results/nl2sql/raw
wc -l results/nl2sql/raw/*.jsonl
```

Для `ea` число строк в JSONL равно числу уже обработанных sample.

## Полезные файлы

- [`nl2sql/notebooks/01_report_ea.ipynb`](/home/count/code/vkr/nl2sql/notebooks/01_report_ea.ipynb)
- [`nl2sql/notebooks/02_report_pass_k.ipynb`](/home/count/code/vkr/nl2sql/notebooks/02_report_pass_k.ipynb)
- [`plan.md`](/home/count/code/vkr/plan.md)
