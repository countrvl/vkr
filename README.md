# nl2sql-bench

NL2SQL.
Сравниваются две модели на бенчмарках **Spider 1.0** и **BIRD**:

| Модель | Тип | Бэкенд |
| --- | --- | --- |
| **M1** DeepSeek-V3.2 | Frontier API | OpenAI-compatible API (`api.deepseek.com`) |
| **M2** SQLCoder-7B | Compact local | Ollama (`localhost:11434`) |

Метрики: Execution Accuracy (EA), Pass@K (K=1,5,10), Expert Score (ES), Efficiency (Eff).

---

## Быстрый старт

```bash
# 1. Установить зависимости
uv sync

# 2. Настроить переменные окружения
echo "DEEPSEEK_API_KEY=sk-..." > .env

# 3. Поднять Ollama и загрузить модель (для M2)
ollama serve
ollama pull sqlcoder:7b

# 4. Скачать данные
uv run python scripts/01_download_data.py --benchmark all

# 5. Тестовый прогон (--limit для быстрой проверки)
uv run python scripts/02_run_inference.py --model m2_compact --benchmark spider --mode ea --limit 10

# 6. Полный прогон
uv run python scripts/02_run_inference.py --model all --benchmark all --mode ea
uv run python scripts/02_run_inference.py --model all --benchmark all --mode pass_k

# 7. Вычислить метрики
uv run python scripts/03_evaluate.py

# 8. Анализ в ноутбуках
jupyter lab notebooks/
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
│   ├── 01_download_data.py   # скачать и подготовить Spider/BIRD
│   ├── 02_run_inference.py   # запустить инференс → results/raw/*.jsonl
│   └── 03_evaluate.py        # EA + Pass@K + Eff → results/metrics/summary_metrics.csv
│
├── src/
│   ├── data/
│   │   ├── loader.py     # DataSample, load_spider(), load_bird()
│   │   ├── schema.py     # serialize_schema() — CREATE TABLE statements
│   │   └── download.py   # httpx-загрузка Spider и BIRD (прямые URL)
│   ├── prompt/
│   │   ├── template.py   # PromptBuilder с кешированным Jinja2 шаблоном
│   │   └── templates/nl2sql.j2
│   ├── inference/
│   │   ├── base.py           # GenerationResult, InferenceBackend, extract_sql()
│   │   ├── api_backend.py    # APIBackend (DeepSeek, retry + exponential backoff)
│   │   ├── ollama_backend.py # OllamaBackend (SQLCoder, /api/generate)
│   │   └── runner.py         # ExperimentRunner — batch + resume по sample_id
│   └── evaluation/
│       ├── executor.py    # execute_sql() — SQLite + timeout + нормализация строк
│       ├── ea.py          # execution_accuracy()
│       ├── pass_at_k.py   # pass_at_k(), compute_all_pass_at_k()
│       ├── expert_score.py # expert_score(), ExpertEvaluation, cohens_kappa()
│       └── efficiency.py  # compute_efficiency(), normalize_efficiency_rows()
│
├── notebooks/
│   ├── 01_eda.ipynb           # распределение запросов Spider/BIRD
│   ├── 02_results.ipynb       # EA и Pass@K: основные результаты
│   ├── 03_expert_score.ipynb  # ES, Cohen's κ
│   ├── 04_efficiency.ipynb    # Tinf, Mem, Tok, Cost, Eff
│   ├── 05_hypothesis.ipynb    # McNemar test, bootstrap CI, H0/H1
│   └── 06_error_analysis.ipynb # категоризация ошибок, Venn-диаграмма
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

Все параметры эксперимента — в YAML-файлах, magic numbers в коде отсутствуют.

**`configs/experiment.yaml`** — seed, temperature, top_p, k_values, max_tokens, `data_dir`, `results_dir`
**`configs/models.yaml`** — модели, бэкенды, API-ключи (через env vars), параметры, pricing
**`configs/metrics.yaml`** — веса Eff (α+β+γ+δ = 1.0), pricing reference values, statistical tests, параметры ES

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
