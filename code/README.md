# Code-domain

`code/` содержит пайплайн для оценки моделей в задаче генерации Python-кода.
Домен использует общую инфраструктуру репозитория для моделей, транспорта,
логирования и статистики, но имеет отдельные benchmark-и, prompt-ы, метрики и
артефакты.

Папка `code/` не является импортируемым Python-пакетом, чтобы не конфликтовать
со стандартным модулем `code`. Импортируемый пакет домена называется
`code_bench`.

## Benchmark-и и режимы

| Benchmark | Назначение | Данные |
| --- | --- | --- |
| `HumanEval+` | Генерация Python-функций с расширенными тестами EvalPlus | `data/code/humaneval_plus/` |
| `MBPP+` | Прикладные задачи программирования с расширенными тестами EvalPlus | `data/code/mbpp_plus/` |

Основные режимы:

- `fc` - Functional Correctness, single-shot генерация и запуск тестов.
- `pass_k` - многократная генерация и расчет Pass@K для `K = 1, 5, 10`.
- `ea` - CLI-alias для `fc`.

## Метрики

| Метрика | Содержание |
| --- | --- |
| `Functional Correctness` | Доля задач, где сгенерированный код проходит тесты |
| `Pass@K` | Доля задач, где хотя бы одна из K генераций проходит тесты |
| `latency`, `tokens`, `cost` | Компоненты вычислительного профиля |
| `Efficiency` | Производный показатель на основе качества и вычислительного профиля |

Оценка корректности выполняется поверх EvalPlus. Настройки эксперимента
находятся в [`configs/experiment.yaml`](configs/experiment.yaml), настройки
метрик - в [`configs/metrics.yaml`](configs/metrics.yaml).

## Модели

Модели берутся из [`../shared/configs/models.yaml`](../shared/configs/models.yaml)
и фильтруются по `supports_code: true`.

| Класс | Ключ | Отображаемое имя | Backend |
| --- | --- | --- | --- |
| `M1` | `m1_deepseek` | `DeepSeek V3.2` | API |
| `M1` | `m1_chatgpt` | `ChatGPT 5.2` | API |
| `M1` | `m1_claude` | `Claude Sonnet 4.5` | Anthropic |
| `M2` | `m2_qwen2_5_coder` | `Qwen2.5-Coder-7B Instruct Q4_K_M` | Ollama |
| `M2` | `m2_qwen2_5_coder_14b` | `Qwen2.5-Coder-14B Instruct` | Ollama |
| `M2` | `m2_deepseek_coder` | `DeepSeek-Coder-V2-Lite 16B Q4_0` | Ollama |

Дополнительные code-модели могут присутствовать в общем конфиге с
`active_by_default: false`; они запускаются по явному ключу модели.

## Структура домена

```text
code/
├── code_bench/    # Импортируемый пакет домена генерации кода
├── configs/       # benchmark, experiment и metrics конфиги
├── notebooks/     # Аналитический ноутбук
├── scripts/       # Prepare, inference, evaluation, archive
└── tests/         # Тесты доменного кода
```

Ключевые файлы:

- [`configs/benchmarks.yaml`](configs/benchmarks.yaml) - описание
  `HumanEval+` и `MBPP+`.
- [`configs/experiment.yaml`](configs/experiment.yaml) - seed, sampling,
  `k_values`, timeout и каталоги.
- [`code_bench/prompt/template.py`](code_bench/prompt/template.py) - сборка
  prompt-ов для генерации кода.
- [`code_bench/inference/runner.py`](code_bench/inference/runner.py) - запуск
  инференса и запись raw JSONL.
- [`code_bench/evaluation/`](code_bench/evaluation/) - Functional Correctness,
  Pass@K и efficiency.

## Подготовка окружения

Из корня репозитория:

```bash
uv sync
cp .env.example .env
```

Для API-моделей в `.env` задаются ключи соответствующих провайдеров. Для
локальных `M2` нужен запущенный Ollama и модели из
`../shared/configs/models.yaml`.

## Подготовка данных

```bash
uv run python code/scripts/01_prepare_benchmarks.py --benchmark all
```

Для HumanEval+ доступен mini-режим:

```bash
uv run python code/scripts/01_prepare_benchmarks.py --benchmark humaneval_plus --mini
```

## Запуск инференса

Functional Correctness:

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode fc
```

Pass@K:

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode pass_k
```

Отдельные модели:

```bash
uv run python code/scripts/02_run_inference.py --model m1_claude --benchmark all --mode fc
uv run python code/scripts/02_run_inference.py --model m2_qwen2_5_coder_14b --benchmark all --mode fc
```

Alias `ea` для Functional Correctness:

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode ea
```

Smoke-run:

```bash
uv run python code/scripts/02_run_inference.py --model m1_deepseek --benchmark humaneval_plus --mode fc --limit 5 --mini
```

## Оценка метрик

```bash
uv run python code/scripts/03_evaluate.py --run-label fc
uv run python code/scripts/03_evaluate.py --run-label pass_k
```

Оба режима можно пересчитать одной командой:

```bash
uv run python code/scripts/03_evaluate.py --run-label all
```

Выходные файлы:

| Путь | Содержимое |
| --- | --- |
| `results/code/metrics/fc/summary_metrics.csv` | Агрегированные FC-метрики |
| `results/code/metrics/fc/sample_metrics.csv` | Метрики на уровне задач |
| `results/code/metrics/fc/candidate_metrics.csv` | Метрики на уровне генераций |
| `results/code/metrics/pass_k/summary_metrics.csv` | Агрегированные Pass@K-метрики |
| `results/code/metrics/pass_k/sample_metrics.csv` | Pass@K на уровне задач |

## Артефакты

| Путь | Назначение |
| --- | --- |
| `results/code/raw/*.jsonl` | Raw-ответы моделей |
| `results/code/batches/` | Metadata batch-запусков |
| `results/code/metrics/fc/` | Functional Correctness и связанные таблицы |
| `results/code/metrics/pass_k/` | Pass@K и связанные таблицы |
| `results/code/figures/fc/` | Графики FC-отчета |
| `results/code/figures/pass_k/` | Графики Pass@K-отчета |
| `results/code/figures/final/` | Итоговые графики code-domain |
| `results/code/archive/` | Архивные артефакты, отделенные от основных таблиц |

`summary_metrics.csv` содержит агрегированные метрики по моделям и benchmark-ам.
`sample_metrics.csv` и `candidate_metrics.csv` используются для анализа на
уровне отдельных задач и генераций.

## Ноутбук

```bash
jupyter lab code/notebooks/01_report_fc_passk.ipynb
```

Пересборка без открытия Jupyter:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace code/notebooks/01_report_fc_passk.ipynb
```

## Архивация

```bash
.venv/bin/python code/scripts/04_archive_results.py --dry-run
.venv/bin/python code/scripts/04_archive_results.py --label before_new_run
```

Архивные каталоги используются для отделения новых запусков от предыдущих
наборов raw/metrics/figures.
