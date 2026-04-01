# nl2sql

NL2SQL.
Сравниваются модели на бенчмарках **Spider 1.0** и **BIRD**.

Проект поддерживает несколько моделей в двух классах:

- `M1` — крупные general-purpose LLM через OpenAI-compatible API
- `M2` — компактные специализированные text-to-SQL модели через Ollama

Текущий набор моделей задается в [`nl2sql/configs/models.yaml`](/home/count/code/vkr/nl2sql/configs/models.yaml):

| Модель | Тип | Бэкенд |
| --- | --- | --- |
| `m1_deepseek` / **DeepSeek** | Frontier API | OpenAI-compatible API |
| `m1_chatgpt` / **ChatGPT** | Frontier API | OpenAI-compatible API |
| `m2_defog` / **Defog-Llama3-SQLCoder-8B** | Compact local | Ollama |
| `m2_hrida` / **Hrida-T2SQL** | Compact local | Ollama |
| `m2_arctic` / **Arctic-Text2SQL-R1-7B** | Compact local | Ollama |

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
cp .env.example .env

# 3. Поднять Ollama и загрузить модель (для M2)
ollama serve
ollama pull mannix/defog-llama3-sqlcoder-8b:q4_0

# 4. Скачать данные
uv run python nl2sql/scripts/01_download_data.py --benchmark all

# 5. Тестовый запуск (--limit для быстрой проверки)
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark spider --mode ea --limit 10

# 6. Запуск нескольких моделей
uv run python nl2sql/scripts/02_run_inference.py --model m1_deepseek --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m1_chatgpt --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark all --mode ea

# 7. Полный запуск всех моделей
uv run python nl2sql/scripts/02_run_inference.py --model all --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model all --benchmark all --mode pass_k

# 8. Вычислить метрики
uv run python nl2sql/scripts/03_evaluate.py

# 9. Открыть ноутбук с отчетом
jupyter lab nl2sql/notebooks/01_report_ea.ipynb
```

---

## Структура проекта

```text
nl2sql/
├── configs/
│   ├── experiment.yaml   # seed, temperature, top_p, k_values, data_dir, results_dir
│   ├── models.yaml       # конфигурации доступных M1 и M2
│   └── metrics.yaml      # веса Eff, pricing, statistical_tests, параметры ES
│
├── scripts/
│   ├── 01_download_data.py   # загрузка и подготовка Spider/BIRD
│   ├── 02_run_inference.py   # запуск инференса → results/nl2sql/raw/*.jsonl
│   ├── 03_evaluate.py        # EA + Pass@K + Eff → results/nl2sql/metrics/{ea,pass_k}/summary_metrics.csv
│   └── 04_archive_results.py # архивировать текущие results/{raw,metrics,figures}
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
│   │   ├── api_backend.py    # APIBackend для M1 через OpenAI-compatible API
│   │   ├── ollama_backend.py # OllamaBackend для локальных M2 через /api/generate
│   │   └── runner.py         # ExperimentRunner — batch + resume по sample_id
│   └── evaluation/
│       ├── executor.py    # execute_sql() — SQLite + timeout + нормализация строк
│       ├── ea.py          # execution_accuracy()
│       ├── pass_at_k.py   # pass_at_k(), compute_all_pass_at_k()
│       ├── expert_score.py # expert_score(), ExpertEvaluation, cohens_kappa()
│       └── efficiency.py  # compute_efficiency(), normalize_efficiency_rows()
│
├── notebooks/
│   ├── 01_report.ipynb        # совместимый alias для отчета по ea
│   ├── 01_report_ea.ipynb     # отчет по single-shot режиму ea
│   ├── 02_report_pass_k.ipynb # отчет по режиму pass_k
│   └── analysis_utils.py      # helper-функции для ноутбуков
│
├── results/
│   ├── raw/      # JSONL по benchmark и mode (ea / pass_k)
│   ├── metrics/  # ea/summary_metrics.csv и pass_k/summary_metrics.csv
│   └── figures/  # ea/* и pass_k/* (DPI=300)
│
└── tests/        # pytest suite
```

---

## Скрипты

### `02_run_inference.py`

```text
--model     ключ из nl2sql/configs/models.yaml | all | m1 | m2 | список через запятую   (обязательный)
--benchmark spider | bird | all               (default: all)
--mode      ea | pass_k                       (ea: temp=0, n=1 / pass_k: temp из конфига, n=max(k_values))
--config-dir путь к конфигам                  (default: configs)
--data-dir   корень данных                    (default: из experiment.yaml)
--results-dir директория raw JSONL            (default: из experiment.yaml)
--limit N   ограничить количество samples     (для smoke-test)
```

Примеры:

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m1_deepseek --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m1 --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2 --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model all --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m1,m2_defog --benchmark all --mode ea
```

### `03_evaluate.py`

```text
--config-dir путь к конфигам                  (default: configs)
--raw-dir    путь к директории с JSONL        (default: из experiment.yaml -> results/nl2sql/raw)
--data-dir   корень данных для db_path        (default: из experiment.yaml -> data)
--output-dir путь для CSV                     (default: results/nl2sql/metrics)
--run-label  ea | pass_k | all               (default: all)
```

Примеры:

```bash
uv run python nl2sql/scripts/03_evaluate.py --run-label ea
uv run python nl2sql/scripts/03_evaluate.py --run-label pass_k
uv run python nl2sql/scripts/03_evaluate.py --run-label all
```

### `04_archive_results.py`

Архивирует текущие артефакты из `results/` и дополнительно копирует в архив оба отчетных ноутбука:

- `nl2sql/notebooks/01_report_ea.ipynb`
- `nl2sql/notebooks/02_report_pass_k.ipynb`

```text
--results-dir корень каталога results         (default: results)
--label      суффикс имени архива             (default: artifacts)
--scope      all | ea | pass_k                (default: all)
--dry-run    только показать, что будет сделано
```

Примеры:

```bash
.venv/bin/python nl2sql/scripts/04_archive_results.py --dry-run
.venv/bin/python nl2sql/scripts/04_archive_results.py --label limit_50_run
.venv/bin/python nl2sql/scripts/04_archive_results.py --scope ea --label before_pass_k
```

---

## Конфигурация

Все параметры в YAML-файлах, magic numbers в коде отсутствуют.

**`nl2sql/configs/experiment.yaml`** — seed, temperature, top_p, k_values, max_tokens, `data_dir`, `results_dir`
**`nl2sql/configs/models.yaml`** — модели, бэкенды, `model_id`, URL и API-ключи (через env vars), параметры, pricing
**`nl2sql/configs/metrics.yaml`** — веса Eff (α+β+γ+δ = 1.0), pricing reference values, statistical tests, параметры ES

### Переменные окружения

Пример лежит в [`.env.example`](/home/count/code/vkr/.env.example).

Основные переменные:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_API_URL`
- `DEEPSEEK_MODEL_ID`
- `OPENAI_API_KEY`
- `OPENAI_API_URL`
- `OPENAI_MODEL_ID`
- `OLLAMA_API_URL`

---

## Проверка статуса

### Проверить, какие raw-файлы уже создаются

```bash
ls -lh results/nl2sql/raw
```

### Посмотреть, сколько sample уже записано

```bash
wc -l results/nl2sql/raw/*.jsonl
```

Для `ea` число строк в JSONL соответствует числу уже обработанных sample.

### Проверить прогресс по конкретной модели и режиму

```bash
wc -l results/nl2sql/raw/DeepSeek_*_ea_*.jsonl
wc -l results/nl2sql/raw/ChatGPT_*_ea_*.jsonl
wc -l results/nl2sql/raw/Defog-Llama3-SQLCoder-8B_*_ea_*.jsonl
wc -l results/nl2sql/raw/Hrida-T2SQL_*_ea_*.jsonl
wc -l results/nl2sql/raw/Arctic-Text2SQL-R1-7B_*_ea_*.jsonl
wc -l results/nl2sql/raw/DeepSeek_*_pass_k_*.jsonl
wc -l results/nl2sql/raw/ChatGPT_*_pass_k_*.jsonl
wc -l results/nl2sql/raw/Defog-Llama3-SQLCoder-8B_*_pass_k_*.jsonl
wc -l results/nl2sql/raw/Hrida-T2SQL_*_pass_k_*.jsonl
wc -l results/nl2sql/raw/Arctic-Text2SQL-R1-7B_*_pass_k_*.jsonl
```

### Как понять, что происходит во время выполнения

- если растет число строк в `results/nl2sql/raw/*.jsonl`, выполнение идет;
- если для `ea` файл дошел до `1034` строк на `Spider` или `1534` строк на `BIRD`, соответствующий benchmark завершен;
- если файл уже существует, повторный запуск той же команды продолжит выполнение через `resume`, а не начнет его заново.

### Если выполнение было остановлено

Можно просто повторно запустить ту же команду:

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark all --mode ea
```

`ExperimentRunner` автоматически подхватит последний JSONL для того же `model + benchmark + mode` и продолжит запись.

### Быстрая проверка метрик после завершения

```bash
uv run python nl2sql/scripts/03_evaluate.py --run-label ea
```

Итоговый CSV записывается в `results/nl2sql/metrics/<run_label>/summary_metrics.csv`.

### Очистить рабочие артефакты без удаления истории

Если нужно начать новый запуск с чистого `results/`, не удаляя старые файлы, используй архивирование:

```bash
.venv/bin/python nl2sql/scripts/04_archive_results.py --label before_new_run
```

Скрипт переместит текущие артефакты в новую папку внутри `results/archive/`, а затем создаст нужные рабочие каталоги заново. Для `--scope ea` и `--scope pass_k` архивируются только соответствующие raw-файлы и подпапки `metrics/<run_label>`, `figures/<run_label>`.

---

## Отчеты

В проекте два отдельных отчетных ноутбука:

- **`nl2sql/notebooks/01_report_ea.ipynb`** — отчет по single-shot режиму `ea`
- **`nl2sql/notebooks/02_report_pass_k.ipynb`** — отчет по режиму `pass_k`

Оба ноутбука содержат:
- обзор данных
- основные метрики `EA` и `Pass@K`
- эффективность (`Tinf`, `Tok`, `Cost`, `Eff`)
- сравнение моделей на уровне sample
- error analysis
- блок под expert score

Ноутбуки используют `nl2sql/notebooks/analysis_utils.py`, автоматически находят доступный каталог результатов и сохраняют графики в `results/nl2sql/figures/<run_label>/`.

Если нужно открыть отчет по конкретному набору результатов, можно явно указать каталог:

```bash
NL2SQL_RESULTS_DIR=/путь/к/results/nl2sql jupyter lab nl2sql/notebooks/01_report_ea.ipynb
NL2SQL_RESULTS_DIR=/путь/к/results/nl2sql jupyter lab nl2sql/notebooks/02_report_pass_k.ipynb
```

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
