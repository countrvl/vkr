# NL2SQL-домен

`nl2sql/` - основной экспериментальный домен ВКР. Он содержит пайплайн
генерации, execution-based оценки и анализа SQL-запросов для benchmark-ов
`Spider`, `BIRD` и локального synthetic e-commerce benchmark (`SEB`).

## Назначение

Домен сравнивает два класса моделей:

- `M1` - крупные general-purpose LLM через API;
- `M2` - компактные специализированные text-to-SQL модели через Ollama.

Основные метрики:

- `Execution Accuracy (EA)` - корректность результата выполнения SQL;
- `Pass@K` - вероятность получить хотя бы один корректный SQL среди K
  генераций;
- `Efficiency (Eff)` - интегральная метрика вычислительного профиля;
- latency, tokens, cost - компоненты эффективности.

`Expert Score` в текущем состоянии представлен только шаблоном
`expert_scores_template.csv`. Это не отдельный количественный результат без
протокола экспертной разметки.

## Актуальные модели

Модели задаются в [`../shared/configs/models.yaml`](../shared/configs/models.yaml)
и фильтруются по `supports_sql: true`.

| Ключ | Отображаемое имя | Класс | Backend | Роль в текущем анализе |
| --- | --- | --- | --- | --- |
| `m1_deepseek` | `DeepSeek V3.2` | `M1` | API | основной |
| `m1_chatgpt` | `ChatGPT 5.2` | `M1` | API | основной |
| `m1_claude` | `Claude Sonnet 4.5` | `M1` | Anthropic | дополнительный EA |
| `m2_arctic` | `Arctic-Text2SQL-R1-7B` | `M2` | Ollama | основной |
| `m2_defog` | `Defog-Llama3-SQLCoder-8B q4_0` | `M2` | Ollama | основной |
| `m2_hrida` | `Hrida-T2SQL q8_0` | `M2` | Ollama | основной |

Неактивные или экспериментальные модели, например `Qwen`, `XiYanSQL` и старый
`SQLCoder-7B`, не используются в основном сравнении без отдельного решения.

## Benchmark-и

- `Spider` - кросс-доменный text-to-SQL benchmark.
- `BIRD` - более сложный и реалистичный benchmark с большим числом прикладных
  SQL-задач.
- `SEB` - локальный synthetic e-commerce benchmark на SQLite. Он нужен для
  контролируемой проверки поведения моделей и не заменяет Spider/BIRD.

## Структура домена

```text
nl2sql/
├── configs/       # experiment.yaml, metrics.yaml
├── scripts/       # download, inference, evaluation, archive, SEB helpers
├── src/           # data loaders, inference, prompts, evaluation
├── notebooks/     # аналитические ноутбуки и материалы для приложений
└── tests/         # тесты доменного кода
```

Входные данные находятся в `../data/nl2sql/`, результаты - в
`../results/nl2sql/`.

## Подготовка окружения

```bash
uv sync
cp .env.example .env
```

Для API-моделей в `.env` должны быть заданы соответствующие ключи. Для
локальных `M2` нужен запущенный Ollama:

```bash
ollama serve
```

## Подготовка данных

```bash
uv run python nl2sql/scripts/01_download_data.py --benchmark all
```

Локальный SEB уже хранится в `data/nl2sql/synthetic_ecommerce/`. При
необходимости его можно пересобрать и проверить:

```bash
uv run python nl2sql/scripts/07_prepare_synthetic_ecommerce.py
uv run python nl2sql/scripts/08_validate_synthetic_ecommerce.py
```

## Запуск inference

Полный запуск может занимать много времени, особенно для `BIRD` и `pass_k`.
Для длинных прогонов рекомендуется запускать модели отдельно.

### EA

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m1_deepseek --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m1_chatgpt --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_arctic --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_hrida --benchmark all --mode ea
```

### Pass@K

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m1_deepseek --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m1_chatgpt --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m2_arctic --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m2_hrida --benchmark all --mode pass_k
```

Быстрый smoke-run:

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark spider --mode ea --limit 10
```

## Оценка метрик

Если raw JSONL уже лежат в `results/nl2sql/raw/`, метрики пересчитываются так:

```bash
.venv/bin/python nl2sql/scripts/03_evaluate.py --run-label ea
.venv/bin/python nl2sql/scripts/03_evaluate.py --run-label pass_k
```

Результаты сохраняются в:

- `results/nl2sql/metrics/ea/summary_metrics.csv`;
- `results/nl2sql/metrics/ea/sample_metrics.csv`;
- `results/nl2sql/metrics/pass_k/summary_metrics.csv`;
- `results/nl2sql/metrics/pass_k/sample_metrics.csv`.

Текущий `pass_k` собран как полный активный набор для 5 основных SQL-моделей:
`5 моделей x 2 benchmark-а = 10 строк` в `summary_metrics.csv`.

## Ноутбуки

Основные:

```bash
jupyter lab nl2sql/notebooks/01_report_ea.ipynb
jupyter lab nl2sql/notebooks/02_report_pass_k.ipynb
jupyter lab final_nl2sql_analysis.ipynb
```

Материалы для приложений ВКР:

```bash
jupyter lab nl2sql/notebooks/06_appendix_materials.ipynb
```

SEB-ноутбуки:

```bash
jupyter lab nl2sql/notebooks/03_synthetic_ecommerce_report.ipynb
jupyter lab nl2sql/notebooks/04_synthetic_ecommerce_dataset_description.ipynb
jupyter lab nl2sql/notebooks/05_synthetic_ecommerce_benchmark_analysis.ipynb
```

Пересборка ноутбуков без ручного открытия:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/01_report_ea.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/02_report_pass_k.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace final_nl2sql_analysis.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/06_appendix_materials.ipynb
```

## Артефакты

Активные:

- `results/nl2sql/raw/*.jsonl` - raw inference;
- `results/nl2sql/metrics/ea/*.csv` - EA-метрики;
- `results/nl2sql/metrics/pass_k/*.csv` - Pass@K-метрики;
- `results/nl2sql/figures/ea/*.png` - графики EA;
- `results/nl2sql/figures/pass_k/*.png` - графики Pass@K;
- `results/nl2sql/figures/final/*.png` - итоговые графики;
- `results/nl2sql/synthetic_benchmark/` - результаты SEB.

Архивные и неполные результаты находятся в `results/nl2sql/archive/`. Их не
следует смешивать с активными таблицами без явного указания.

## Проверка статуса raw-прогонов

```bash
ls -lh results/nl2sql/raw
wc -l results/nl2sql/raw/*.jsonl
```

Для `ea` одна строка JSONL соответствует одному sample. Для `pass_k` одна строка
также соответствует одному sample, но внутри записи хранится список из K
генераций.

Ожидаемые размеры активных полных прогонов:

- `Spider`: 1034 sample;
- `BIRD`: 1534 sample.

## Strategy Bench

`nl2sql/src/strategy_bench/` - отдельный экспериментальный модуль для
production-like сценариев на read-only PostgreSQL. Он сравнивает стратегии:

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

Этот контур отделен от Spider/BIRD и не участвует в основных таблицах NL2SQL,
если не указано отдельно.

## Важные ограничения интерпретации

- Сравнивать модели корректно только при одинаковом benchmark-е и полном
  покрытии выборки.
- Smoke-run и неполные raw-файлы не используются для итоговых выводов.
- Старый `SQLCoder-7B` не является активным baseline.
- `expert_scores_template.csv` - только заготовка для экспертной разметки.
- Архивные результаты могут быть полезны для истории экспериментов, но не
  являются источником текущих таблиц ВКР.
