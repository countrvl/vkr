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

## Модели

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

## Бенчмарки

- `Spider` — базовый кросс-доменный benchmark для text-to-SQL
- `BIRD` — более сложный и более реалистичный benchmark

## Конфигурация

- [`nl2sql/configs/experiment.yaml`](/home/count/code/vkr/nl2sql/configs/experiment.yaml) — seed, sampling, `k_values`, пути к данным и результатам
- [`nl2sql/configs/metrics.yaml`](/home/count/code/vkr/nl2sql/configs/metrics.yaml) — веса эффективности и параметры метрик
- [`shared/configs/models.yaml`](/home/count/code/vkr/shared/configs/models.yaml) — единый каталог моделей

Для API-моделей ключи доступа читаются из `.env`. Для локальных `M2` требуется запущенный `Ollama` и загруженные модели.

## Быстрый старт

```bash
uv sync
cp .env.example .env
ollama serve
uv run python nl2sql/scripts/01_download_data.py --benchmark all
```

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

После evaluation итоговые таблицы появляются в `results/nl2sql/metrics/<run_label>/`.

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

`summary_metrics.csv` содержит агрегированные метрики по моделям, а `sample_metrics.csv` используется для анализа результатов на уровне отдельных примеров.

## Полезные файлы

- [`nl2sql/notebooks/01_report_ea.ipynb`](/home/count/code/vkr/nl2sql/notebooks/01_report_ea.ipynb)
- [`nl2sql/notebooks/02_report_pass_k.ipynb`](/home/count/code/vkr/nl2sql/notebooks/02_report_pass_k.ipynb)
