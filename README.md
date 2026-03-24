# nl2sql-bench

Стенд сравнительного эксперимента NL2SQL для магистерской ВКР.
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
cp .env.example .env
# Вставить DEEPSEEK_API_KEY в .env

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
│   ├── experiment.yaml   # seed, temperature, k_values, max_tokens, top_p
│   ├── models.yaml       # M1 (DeepSeek) и M2 (SQLCoder) конфигурации
│   └── metrics.yaml      # веса Eff, pricing DeepSeek, параметры ES
│
├── scripts/
│   ├── 01_download_data.py   # скачать Spider/BIRD через gdown
│   ├── 02_run_inference.py   # запустить инференс → results/raw/*.jsonl
│   └── 03_evaluate.py        # EA + Pass@K + Eff → results/metrics/summary_metrics.csv
│
├── src/
│   ├── data/
│   │   ├── loader.py     # DataSample, load_spider(), load_bird()
│   │   ├── schema.py     # serialize_schema() — CREATE TABLE statements
│   │   └── download.py   # gdown-обёртки для Spider и BIRD
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
│   ├── raw/      # append-only JSONL (не перезаписывать!)
│   ├── metrics/  # summary_metrics.csv
│   └── figures/  # графики для ВКР (DPI=300)
│
└── tests/        # 37 тестов, pytest
```

---

## Скрипты

### `02_run_inference.py`

```text
--model     m1_frontier | m2_compact | all   (обязательный)
--benchmark spider | bird | all               (default: all)
--mode      ea | pass_k                       (ea: temp=0, n=1 / pass_k: temp из конфига, n=max(k_values))
--limit N   ограничить количество samples     (для smoke-test)
```

### `03_evaluate.py`

```text
--raw-dir    путь к директории с JSONL        (default: results/raw)
--output-dir путь для CSV                     (default: results/metrics)
--config-dir путь к конфигам                  (default: configs)
```

---

## Конфигурация

Все параметры эксперимента — в YAML-файлах, magic numbers в коде отсутствуют.

**`configs/experiment.yaml`** — seed, temperature, top_p, k_values, max_tokens
**`configs/models.yaml`** — модели, бэкенды, API-ключи (через env vars), параметры
**`configs/metrics.yaml`** — веса Eff (α+β+γ+δ = 1.0), pricing DeepSeek, параметры ES

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

- **Resume**: `ExperimentRunner` при старте читает уже записанные `sample_id` из JSONL и пропускает их. Безопасно прерывать и перезапускать.
- **fsync**: сброс на диск каждые 50 записей + финальный fsync (не после каждой записи).
- **Retry**: 3 попытки с exponential backoff (1 s, 2 s, 4 s) для обоих бэкендов.
- **Токены при n > 1**: prompt-токены не делятся (один вход для всех choices); completion-токены распределяются без потери остатка.
- **seed/top_p**: читаются из `experiment.yaml` и передаются в оба бэкенда.
- **Шаблон**: Jinja2 загружается один раз при создании `PromptBuilder` (`auto_reload=False`).

---

## Тесты

```bash
uv run pytest tests/ -v
# 37 тестов: loader, executor, base (extract_sql), prompt, metrics
```

---

## Зависимости

```bash
uv sync              # основные зависимости
uv sync --extra dev  # + pytest, ruff
```

Основные: `openai`, `httpx`, `jinja2`, `pyyaml`, `python-dotenv`, `tqdm`, `gdown`
Notebooks: `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `jupyter`
