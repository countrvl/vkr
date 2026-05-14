# NL2SQL-домен

`nl2sql/` содержит пайплайн для оценки моделей в задаче text-to-SQL:
генерации SQL-запроса по вопросу на естественном языке и схеме базы данных.
Домен объединяет загрузку benchmark-ов, построение prompt-ов, инференс,
execution-based оценку, расчет эффективности и подготовку отчетных
артефактов.

## Benchmark-и и режимы

| Benchmark | Назначение | Данные |
| --- | --- | --- |
| `Spider` | Кросс-доменная text-to-SQL выборка | `data/nl2sql/spider/` |
| `BIRD` | Прикладные SQL-задачи повышенной сложности | `data/nl2sql/bird/` |
| `SEB` | Synthetic e-commerce benchmark на SQLite | `data/nl2sql/synthetic_ecommerce/` |

Основные режимы:

- `ea` - single-shot генерация и оценка Execution Accuracy.
- `pass_k` - многократная генерация и расчет Pass@K для `K = 1, 3, 5`.

## Метрики

| Метрика | Содержание |
| --- | --- |
| `Execution Accuracy` | Совпадение результата выполнения предсказанного SQL с результатом gold SQL |
| `Pass@K` | Доля задач, где хотя бы одна из K генераций проходит execution-based проверку |
| `Efficiency` | Интегральный показатель на основе latency, token usage и cost |
| `latency`, `tokens`, `cost` | Компоненты вычислительного профиля модели |

Настройки метрик находятся в [`configs/metrics.yaml`](configs/metrics.yaml).
Статистические интервалы рассчитываются с параметрами из этого же файла.

## Модели

Модели берутся из [`../shared/configs/models.yaml`](../shared/configs/models.yaml)
и фильтруются по `supports_sql: true`.

### Основной набор для `ea` и `pass_k`

| Класс | Ключ | Отображаемое имя | Backend |
| --- | --- | --- | --- |
| `M1` | `m1_deepseek` | `DeepSeek V3.2` | API |
| `M1` | `m1_chatgpt` | `ChatGPT 5.2` | API |
| `M2` | `m2_arctic` | `Arctic-Text2SQL-R1-7B` | Ollama |
| `M2` | `m2_defog` | `Defog-Llama3-SQLCoder-8B q4_0` | Ollama |
| `M2` | `m2_hrida` | `Hrida-T2SQL q8_0` | Ollama |

### Дополнительная API-модель

| Ключ | Отображаемое имя | Backend | Представление в артефактах |
| --- | --- | --- | --- |
| `m1_claude` | `Claude Sonnet 4.5` | Anthropic | Дополнительные EA-результаты |

`Claude Sonnet 4.5` не включается в основной `pass_k` набор. Поэтому
`results/nl2sql/metrics/pass_k/summary_metrics.csv` соответствует 5 основным
SQL-моделям на 2 benchmark-ах, а
`results/nl2sql/metrics/ea/summary_metrics.csv` дополнительно содержит строки
для `Claude Sonnet 4.5`.

## Структура домена

```text
nl2sql/
├── configs/       # experiment.yaml, metrics.yaml
├── notebooks/     # Отчеты, графики, материалы для приложений
├── scripts/       # Download, inference, evaluation, archive, SEB helpers
├── src/           # Loaders, prompts, inference, evaluation, strategy bench
└── tests/         # Тесты доменного кода
```

Ключевые файлы:

- [`configs/experiment.yaml`](configs/experiment.yaml) - seed, sampling,
  `k_values`, каталоги данных и raw-результатов.
- [`configs/metrics.yaml`](configs/metrics.yaml) - веса эффективности,
  bootstrap и параметры экспертной разметки.
- [`src/prompt/template.py`](src/prompt/template.py) - сборка prompt-ов.
- [`src/inference/runner.py`](src/inference/runner.py) - запуск инференса и
  resume по raw JSONL.
- [`src/evaluation/`](src/evaluation/) - EA, Pass@K, SQL executor,
  efficiency.

## Подготовка окружения

Из корня репозитория:

```bash
uv sync
cp .env.example .env
```

Для API-моделей в `.env` задаются ключи соответствующих провайдеров. Для
локальных `M2` нужен Ollama:

```bash
ollama serve
```

Модели Ollama должны быть доступны по `model_id` из
`../shared/configs/models.yaml`.

## Подготовка данных

```bash
uv run python nl2sql/scripts/01_download_data.py --benchmark all
```

Подготовка и проверка synthetic e-commerce benchmark:

```bash
uv run python nl2sql/scripts/07_prepare_synthetic_ecommerce.py
uv run python nl2sql/scripts/08_validate_synthetic_ecommerce.py
```

## Запуск инференса

### Execution Accuracy

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m1_deepseek --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m1_chatgpt --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_arctic --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_hrida --benchmark all --mode ea
```

Дополнительная EA-модель:

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m1_claude --benchmark all --mode ea
```

### Pass@K

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m1_deepseek --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m1_chatgpt --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m2_arctic --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m2_hrida --benchmark all --mode pass_k
```

Smoke-run:

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark spider --mode ea --limit 10
```

## Оценка метрик

```bash
.venv/bin/python nl2sql/scripts/03_evaluate.py --run-label ea
.venv/bin/python nl2sql/scripts/03_evaluate.py --run-label pass_k
```

Можно пересчитать оба режима одной командой:

```bash
.venv/bin/python nl2sql/scripts/03_evaluate.py --run-label all
```

Выходные файлы:

| Путь | Содержимое |
| --- | --- |
| `results/nl2sql/metrics/ea/summary_metrics.csv` | Агрегированные EA-метрики |
| `results/nl2sql/metrics/ea/sample_metrics.csv` | EA на уровне отдельных примеров |
| `results/nl2sql/metrics/pass_k/summary_metrics.csv` | Агрегированные Pass@K-метрики |
| `results/nl2sql/metrics/pass_k/sample_metrics.csv` | Pass@K на уровне отдельных примеров |
| `results/nl2sql/metrics/*/expert_scores_template.csv` | Шаблон для экспертной разметки |

## Артефакты

| Путь | Назначение |
| --- | --- |
| `results/nl2sql/raw/*.jsonl` | Raw-ответы моделей, один sample на строку |
| `results/nl2sql/batches/` | Metadata batch-запусков |
| `results/nl2sql/figures/ea/` | Графики для EA-отчета |
| `results/nl2sql/figures/pass_k/` | Графики для Pass@K-отчета |
| `results/nl2sql/figures/final/` | Итоговые графики |
| `results/nl2sql/synthetic_benchmark/` | Результаты SEB |
| `results/nl2sql/archive/` | Архивные артефакты, отделенные от основных таблиц |

Для `ea` одна строка JSONL соответствует одному sample. Для `pass_k` одна
строка также соответствует одному sample, а список генераций хранится внутри
записи.

Проверка размера raw-файлов:

```bash
ls -lh results/nl2sql/raw
wc -l results/nl2sql/raw/*.jsonl
```

Размеры полных dev-наборов:

- `Spider`: 1034 sample.
- `BIRD`: 1534 sample.

## Ноутбуки

Основные аналитические ноутбуки:

```bash
jupyter lab nl2sql/notebooks/01_report_ea.ipynb
jupyter lab nl2sql/notebooks/02_report_pass_k.ipynb
jupyter lab final_nl2sql_analysis.ipynb
```

Материалы для приложений:

```bash
jupyter lab nl2sql/notebooks/06_appendix_materials.ipynb
```

SEB-ноутбуки:

```bash
jupyter lab nl2sql/notebooks/03_synthetic_ecommerce_report.ipynb
jupyter lab nl2sql/notebooks/04_synthetic_ecommerce_dataset_description.ipynb
jupyter lab nl2sql/notebooks/05_synthetic_ecommerce_benchmark_analysis.ipynb
```

Пересборка без открытия Jupyter:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/01_report_ea.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/02_report_pass_k.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace final_nl2sql_analysis.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/06_appendix_materials.ipynb
```

## Архивация

```bash
.venv/bin/python nl2sql/scripts/04_archive_results.py --dry-run
.venv/bin/python nl2sql/scripts/04_archive_results.py --label before_new_run
.venv/bin/python nl2sql/scripts/04_archive_results.py --scope ea --label before_pass_k
```

Архивация переносит выбранные артефакты в `results/nl2sql/archive/` и
позволяет отделять новые прогоны от предыдущих наборов raw/metrics/figures.

## Strategy Bench

`nl2sql/src/strategy_bench/` содержит отдельный контур для production-like
сценариев на read-only PostgreSQL. Поддерживаются стратегии:

- `generate_only`;
- `generate_validate_retry`;
- `routing`.

Базовый запуск:

```bash
uv run python -m nl2sql.src.strategy_bench.cli \
  --dataset path/to/cases.yaml \
  --db-dsn-env NL2SQL_STRATEGY_DB_DSN \
  --model m1_chatgpt \
  --strategy all \
  --catalog-path path/to/catalog.yaml
```

Этот контур отделен от Spider/BIRD/SEB и имеет собственные входные данные.
