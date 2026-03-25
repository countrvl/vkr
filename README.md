# nl2sql-bench

NL2SQL.
Сравниваются модели на бенчмарках **Spider 1.0** и **BIRD**:

| Модель | Тип | Бэкенд |
| --- | --- | --- |
| **M1** DeepSeek-V3.2 | Frontier API | OpenAI-compatible API (`api.deepseek.com`) |
| **M2** Defog-Llama3-SQLCoder-8B | Compact local | Ollama (`localhost:11434`) |

Метрики: Execution Accuracy (EA), Pass@K (K=1,5,10), Expert Score (ES), Efficiency (Eff).

---

## Бенчмарки

- **Spider 1.0** — классический кросс-доменный benchmark для text-to-SQL. Содержит вопросы и SQL-запросы по множеству разных баз данных и используется как базовый стандарт для оценки обобщающей способности моделей.
- **BIRD** — более сложный и более современный benchmark для text-to-SQL с большим числом реалистичных баз данных и запросов. Обычно он заметно тяжелее для моделей, чем Spider, особенно по качеству генерации на сложных схемах.

Ключевые метрики в проекте:

- **EA (Execution Accuracy)** — доля примеров, где предсказанный SQL после исполнения дает тот же результат, что и эталонный запрос.
- **Pass@K** — вероятность того, что среди `K` сгенерированных кандидатов есть хотя бы один корректный запрос. Эта метрика особенно полезна для оценки качества модели в режиме множественной генерации.
- **Expert Score** — экспертная оценка качества SQL по ручной разметке. Используется как дополнительная качественная метрика там, где одной execution-based оценки недостаточно.
- **Efficiency** — агрегированная метрика эффективности, которая учитывает время инференса, память, число токенов и стоимость генерации.

---

## Быстрый старт

```bash
# 1. Установить зависимости
uv sync

# 2. Настроить переменные окружения
echo "DEEPSEEK_API_KEY=sk-..." > .env

# 3. Поднять Ollama и загрузить модель (для M2)
ollama serve
ollama pull mannix/defog-llama3-sqlcoder-8b:q4_0

# 4. Скачать данные
uv run python scripts/01_download_data.py --benchmark all

# 5. Тестовый запуск (--limit для быстрой проверки)
uv run python scripts/02_run_inference.py --model m2_compact --benchmark spider --mode ea --limit 10

# 6. Полный запуск
uv run python scripts/02_run_inference.py --model all --benchmark all --mode ea
uv run python scripts/02_run_inference.py --model all --benchmark all --mode pass_k

# 7. Вычислить метрики
uv run python scripts/03_evaluate.py

# 8. Открыть ноутбук с отчетом
jupyter lab notebooks/01_report.ipynb
```

---

## Структура проекта

```text
nl2sql-bench/
├── configs/
│   ├── experiment.yaml   # seed, temperature, top_p, k_values, data_dir, results_dir
│   ├── models.yaml       # M1 (DeepSeek) и M2 (SQLCoder) конфигурации
│   └── metrics.yaml      # веса Eff, pricing, statistical_tests, параметры ES
│
├── scripts/
│   ├── 01_download_data.py   # загрузка и подготовка Spider/BIRD
│   ├── 02_run_inference.py   # запуск инференса → results/raw/*.jsonl
│   └── 03_evaluate.py        # EA + Pass@K + Eff → results/metrics/summary_metrics.csv
│
├── src/
│   ├── data/
│   │   ├── loader.py     # DataSample, load_spider(), load_bird()
│   │   ├── schema.py     # serialize_schema() — CREATE TABLE statements
│   │   └── download.py   # httpx-загрузка Spider и BIRD
│   ├── prompt/
│   │   ├── template.py   # PromptBuilder с кешированным Jinja2 шаблоном
│   │   └── templates/nl2sql.j2
│   ├── inference/
│   │   ├── base.py           # GenerationResult, InferenceBackend, extract_sql()
│   │   ├── api_backend.py    # APIBackend (DeepSeek, retry + exponential backoff)
│   │   ├── ollama_backend.py # OllamaBackend (local M2 via /api/generate)
│   │   └── runner.py         # ExperimentRunner — batch + resume по sample_id
│   └── evaluation/
│       ├── executor.py    # execute_sql() — SQLite + timeout + нормализация строк
│       ├── ea.py          # execution_accuracy()
│       ├── pass_at_k.py   # pass_at_k(), compute_all_pass_at_k()
│       ├── expert_score.py # expert_score(), ExpertEvaluation, cohens_kappa()
│       └── efficiency.py  # compute_efficiency(), normalize_efficiency_rows()
│
├── notebooks/
│   ├── 01_report.ipynb        # единый отчет по экспериментам
│   └── analysis_utils.py      # helper-функции для ноутбука
│
├── results/
│   ├── raw/      # JSONL по benchmark и mode (ea / pass_k)
│   ├── metrics/  # summary_metrics.csv
│   └── figures/  # графики (DPI=300)
│
└── tests/        # pytest suite
```

---

## Скрипты

### `02_run_inference.py`

```text
--model     m1_frontier | m2_compact | all   (обязательный)
--benchmark spider | bird | all               (default: all)
--mode      ea | pass_k                       (ea: temp=0, n=1 / pass_k: temp из конфига, n=max(k_values))
--config-dir путь к конфигам                  (default: configs)
--data-dir   корень данных                    (default: из experiment.yaml)
--results-dir директория raw JSONL            (default: из experiment.yaml)
--limit N   ограничить количество samples     (для smoke-test)
```

### `03_evaluate.py`

```text
--config-dir путь к конфигам                  (default: configs)
--raw-dir    путь к директории с JSONL        (default: из experiment.yaml -> results/raw)
--data-dir   корень данных для db_path        (default: из experiment.yaml -> data)
--output-dir путь для CSV                     (default: results/metrics)
```

---

## Конфигурация

Все параметры в YAML-файлах, magic numbers в коде отсутствуют.

**`configs/experiment.yaml`** — seed, temperature, top_p, k_values, max_tokens, `data_dir`, `results_dir`
**`configs/models.yaml`** — модели, бэкенды, API-ключи (через env vars), параметры, pricing
**`configs/metrics.yaml`** — веса Eff (α+β+γ+δ = 1.0), pricing reference values, statistical tests, параметры ES

---

## Проверка статуса

### Проверить, какие raw-файлы уже создаются

```bash
ls -lh results/raw
```

### Посмотреть, сколько sample уже записано

```bash
wc -l results/raw/*.jsonl
```

Для `ea` число строк в JSONL соответствует числу уже обработанных sample.

### Проверить прогресс по конкретной модели и режиму

```bash
wc -l results/raw/DeepSeek-V3.2_*_ea_*.jsonl
wc -l results/raw/Defog-Llama3-SQLCoder-8B_*_ea_*.jsonl
wc -l results/raw/DeepSeek-V3.2_*_pass_k_*.jsonl
wc -l results/raw/Defog-Llama3-SQLCoder-8B_*_pass_k_*.jsonl
```

### Как понять, что происходит во время выполнения

- если растет число строк в `results/raw/*.jsonl`, выполнение идет;
- если для `ea` файл дошел до `1034` строк на `Spider` или `1534` строк на `BIRD`, соответствующий benchmark завершен;
- если файл уже существует, повторный запуск той же команды продолжит выполнение через `resume`, а не начнет его заново.

### Если выполнение было остановлено

Можно просто повторно запустить ту же команду:

```bash
uv run python scripts/02_run_inference.py --model m2_compact --benchmark all --mode ea
```

`ExperimentRunner` автоматически подхватит последний JSONL для того же `model + benchmark + mode` и продолжит запись.

### Быстрая проверка метрик после завершения

```bash
uv run python scripts/03_evaluate.py
```

Итоговый CSV записывается в `results/metrics/summary_metrics.csv`.

---

## Отчеты

Основной ноутбук проекта — **`notebooks/01_report.ipynb`**.

Содержит:
- обзор данных
- основные метрики `EA` и `Pass@K`
- эффективность (`Tinf`, `Tok`, `Cost`, `Eff`)
- сравнение моделей на уровне sample
- error analysis
- блок под expert score

Ноутбук использует `notebooks/analysis_utils.py`, автоматически находит доступный каталог результатов и сохраняет графики в `results/figures/`.

---

## Метрики

| Метрика | Формула | Реализация |
| --- | --- | --- |
| **EA** | `(1/N)·Σ I(exec(ŝ) = exec(s*))` | `src/evaluation/ea.py` |
| **Pass@K** | `1 − C(n−c,k) / C(n,k)` (unbiased, Chen et al. 2021) | `src/evaluation/pass_at_k.py` |
| **ES** | `(C + E + R) / 3`, каждый критерий 1–5 | `src/evaluation/expert_score.py` |
| **Eff** | `α·Tinf + β·Mem + γ·Tok + δ·Cost` (min-max нормализация) | `src/evaluation/efficiency.py` |

Сравнение EA: результаты нормализуются (ORDER BY игнорируется — set equality, стандарт Spider/BIRD).

---

## Ключевые детали реализации

- **Resume**: `ExperimentRunner` продолжает запись в последний JSONL для конкретного `model + benchmark + mode`.
- **Raw results**: для `ea` и `pass_k` создаются отдельные JSONL-файлы. Внутри записи сохраняется `run_label`.
- **db_path**: в raw JSONL путь к БД сохраняется относительно `data_dir`, когда это возможно.
- **fsync**: сброс на диск каждые 50 записей + финальный fsync (не после каждой записи).
- **Retry**: 3 попытки с exponential backoff (1 s, 2 s, 4 s) для обоих бэкендов.
- **Токены при n > 1**: prompt-токены не делятся (один вход для всех choices); completion-токены распределяются без потери остатка.
- **seed/top_p**: читаются из `experiment.yaml` и передаются в оба бэкенда.
- **Pricing**: для API-моделей pricing из `models.yaml` сохраняется в metadata generation results и используется в `Efficiency`.
- **Шаблон**: Jinja2 загружается один раз при создании `PromptBuilder` (`auto_reload=False`).

---

## Тесты

```bash
uv run pytest tests/ -v
# текущий набор тестов: loader, download, runner, executor, base, prompt, metrics
```

---

## Зависимости

```bash
uv sync              # основные зависимости
uv sync --extra dev  # + pytest, ruff
```

Основные: `openai`, `httpx`, `jinja2`, `pyyaml`, `python-dotenv`, `tqdm`
Notebooks: `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `jupyter`
